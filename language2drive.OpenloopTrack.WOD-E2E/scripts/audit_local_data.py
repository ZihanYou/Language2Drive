"""Read-only metadata inventory of uncompressed E2ED TFRecords; no JPEG decoding.
Wire inspection skips large payloads. CRCs are NOT verified; this is not a
replacement for full record integrity validation or the official protobuf reader.
"""
import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import struct


def varint(f):
    n = 0
    for shift in range(0, 70, 7):
        b = f.read(1)
        if not b:
            raise EOFError('truncated varint')
        n |= (b[0] & 127) << shift
        if b[0] < 128:
            return n
    raise ValueError('invalid varint')


def fields(f, end):
    while f.tell() < end:
        tag = varint(f)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError('zero field number')
        if wire == 0:
            yield number, wire, varint(f), None
        elif wire in (1, 2, 5):
            size = varint(f) if wire == 2 else (8 if wire == 1 else 4)
            start = f.tell()
            if start + size > end:
                raise ValueError('field exceeds message boundary')
            yield number, wire, start, size
            f.seek(start + size)
        else:
            raise ValueError(f'unsupported wire type {wire}')
    if f.tell() != end:
        raise ValueError('message boundary mismatch')


def trajectory(f, end):
    out = {}
    names = {1:'x', 2:'y', 3:'z', 4:'vx', 5:'vy', 6:'ax', 7:'ay'}
    for n, w, start, size in fields(f, end):
        if n in names and w in (2, 5):
            raw = f.read(size)
            out.setdefault(names[n], []).extend(struct.unpack('<' + 'f' * (size // 4), raw))
        elif n == 8 and w == 5:
            out['score'] = struct.unpack('<f', f.read(4))[0]
    return out


def frame(f, end, detailed=True):
    out = {'camera_ids':[], 'calibration_ids':[], 'frame_fields':[]}
    for n,w,start,size in fields(f,end):
        out['frame_fields'].append(n)
        if n == 1 and w == 2:
            for cn,cw,cs,cz in fields(f,start+size):
                if cn == 1 and cw == 2:
                    out['frame_name'] = f.read(cz).decode()
                elif cn == 2 and cw == 2:
                    for kn,kw,ks,kz in fields(f,cs+cz):
                        if kn == 1 and kw == 0:
                            out['calibration_ids'].append(ks)
        elif n == 2 and w == 0:
            out['timestamp_us'] = start
            if not detailed:
                f.seek(end)
                break
        elif n == 4 and w == 2:
            for imn,imw,ims,imz in fields(f,start+size):
                if imn == 1 and imw == 0:
                    out['camera_ids'].append(ims)
    return out


def records(path):
    size = path.stat().st_size
    with path.open('rb') as f:
        while f.tell() < size:
            offset = f.tell()
            head = f.read(12)
            if len(head) != 12:
                raise EOFError('truncated record header')
            length = struct.unpack('<Q',head[:8])[0]
            end = f.tell()+length
            if end+4 > size:
                raise EOFError('truncated record payload')
            row = {'file':str(path),'offset':offset,'record_bytes':length,'preferences':[]}
            for n,w,start,z in fields(f,end):
                if n == 1 and w == 2:
                    frame_start, frame_end = start, start+z
                    row.update(frame(f,start+z,detailed=False))
                elif n in (5,6,8) and w == 2:
                    t = trajectory(f,start+z)
                    if n == 8:
                        row['preferences'].append(t)
                    else:
                        row['future' if n == 5 else 'past'] = t
                elif n == 7 and w == 0:
                    row['intent'] = start
            if any(valid(t) for t in row['preferences']):
                f.seek(frame_start)
                row.update(frame(f,frame_end))
            f.seek(end+4)
            yield row


def valid(t):
    return (0 <= t.get('score',-1) <= 10 and
            len(t.get('x',[])) == len(t.get('y',[])) and len(t.get('x',[])) >= 2 and
            all(math.isfinite(v) for k in ('x','y') for v in t[k]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scan-aggregate', action='store_true', help='Also scan the concatenated aggregate record by record')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    clusters=json.loads((args.root/'val_sequence_name_to_scenario_cluster.json').read_text())
    paths=sorted(args.root.glob('*.tfrecord-*'))
    aggregate=args.root/'val_40pct.tfrecord'
    aggregate_check={'path':str(aggregate),'scope':'size and 3 x 4096 byte samples per shard; not full equality proof'}
    if aggregate.exists():
        total=sum(p.stat().st_size for p in paths)
        aggregate_check.update(bytes=aggregate.stat().st_size,shard_bytes=total)
        matches=0; checked=0; offset=0
        if total==aggregate.stat().st_size:
            with aggregate.open('rb') as af:
                for shard in paths:
                    size=shard.stat().st_size
                    with shard.open('rb') as sf:
                        for rel in (0,size//2,max(0,size-4096)):
                            sf.seek(rel); af.seek(offset+rel); checked+=1
                            matches+=sf.read(4096)==af.read(4096)
                    offset+=size
        aggregate_check.update(checked_blocks=checked,matched_blocks=matches)
        if args.scan_aggregate or total!=aggregate.stat().st_size or matches!=checked or not checked:
            paths.append(aggregate)
            aggregate_check['record_scan']='included'
        else:
            aggregate_check['record_scan']='not scanned; sampled as concatenation of numbered shards'
    else:
        aggregate_check['record_scan']='absent'
    seen={}; rated={}; inventory=[]; discrepancies=[]
    camera_counts=collections.Counter(); field_counts=collections.Counter(); past_lengths=collections.Counter()
    for path in paths:
        count=nrated=duplicates=0
        for r in records(path):
            count+=1
            key=(r.get('frame_name'),r.get('timestamp_us'))
            selected={k:r.get(k) for k in ('frame_name','timestamp_us','preferences','past','future','intent','camera_ids','calibration_ids','frame_fields')}
            digest=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
            if key in seen:
                duplicates+=1
                if seen[key] != digest: discrepancies.append({'key':key,'file':str(path)})
            else:
                seen[key]=digest
                past_lengths[len(r.get('past',{}).get('x',[]))]+=1
            good=[i for i,t in enumerate(r['preferences']) if valid(t)]
            if good:
                nrated+=1
                name=r.get('frame_name','')
                matches=[s for s in re.split('[-_/]',name) if s in clusters]
                segment=matches[0] if len(matches)==1 else (name if name in clusters else None)
                r.update(valid_preference_indices=good,sequence_id=segment,
                         scenario_cluster=clusters[segment]['scenario_cluster'] if segment else None)
                rated.setdefault(key,r)
        inventory.append({'path':str(path),'bytes':path.stat().st_size,'records':count,'rated_records':nrated,'duplicate_records':duplicates})
        print(path.name,count,nrated,duplicates,flush=True)
    rs=list(rated.values()); pairs=[]
    for r in rs:
        camera_counts[len(r['camera_ids'])]+=1
        field_counts.update(set(r['frame_fields']))
    for r in rs:
        for a,b in itertools.combinations(r['valid_preference_indices'],2):
            ta,tb=r['preferences'][a],r['preferences'][b]
            pairs.append({'frame_name':r['frame_name'],'sequence_id':r['sequence_id'],'scenario_cluster':r['scenario_cluster'],
                'indices':[a,b],'scores':[ta['score'],tb['score']],
                'point_counts':[len(ta['x']),len(tb['x'])],
                'dx_endpoint':abs(ta['x'][-1]-tb['x'][-1]),'dy_endpoint':abs(ta['y'][-1]-tb['y'][-1]),
                'time_axis_status':'requires_official_loader_confirmation',
                'status':'unreviewed_geometric_pair'})
    summary={'scope':'local validation numbered shards; aggregate inclusion recorded separately; metadata only; JPEG bytes and CRC not validated',
             'aggregate_check':aggregate_check,
             'files':inventory,'raw_records':sum(x['records'] for x in inventory),'unique_frames':len(seen),
             'duplicate_metadata_conflicts':discrepancies,'unique_rated_frames':len(rs),
             'preference_point_counts':dict(collections.Counter(len(t['x']) for r in rs for t in r['preferences'] if valid(t))),
             'rated_zero_timestamps':sum(r.get('timestamp_us') == 0 for r in rs),
             'valid_preferences_distribution':dict(collections.Counter(len(r['valid_preference_indices']) for r in rs)),
             'rated_unique_sequences':len({r['sequence_id'] for r in rs if r['sequence_id']}),
             'rated_unmatched_sequence_ids':sum(r['sequence_id'] is None for r in rs),
             'metadata_sequences':len(clusters),'rated_clusters':dict(collections.Counter(r['scenario_cluster'] for r in rs)),
             'rated_camera_count_distribution':dict(camera_counts),'past_position_length_distribution':dict(past_lengths),
             'rated_populated_frame_field_counts':dict(field_counts),'raw_pairs':len(pairs),
             'pairs_both_scores_ge6':sum(min(x['scores'])>=6 for x in pairs),
             'frames_with_two_scores_ge6':sum(sum(r['preferences'][i]['score']>=6 for i in r['valid_preference_indices'])>=2 for r in rs),
             'rated_example':{k:rs[0][k] for k in ('frame_name','sequence_id','timestamp_us','camera_ids','calibration_ids','intent')} if rs else None}
    (args.out/'local_inventory.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    for name,rows in [('rated_frames.jsonl',rs),('raw_pairs.jsonl',pairs)]:
        with (args.out/name).open('w') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='files'},indent=2),flush=True)

if __name__=='__main__': main()

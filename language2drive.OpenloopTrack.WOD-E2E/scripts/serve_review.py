"""Private WOD candidate review server, intended for access through an SSH tunnel.

Binds to 127.0.0.1 only. Human PASS records a review decision; it does not
certify an executable contract or freeze an evaluation PCU.
"""
import argparse
import csv
import io
import json
import os
import re
from pathlib import Path
import threading
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'review'
QUEUE = REVIEW / 'review_queue.csv'
DECISIONS = REVIEW / 'decisions.json'
EVENTS = REVIEW / 'events.jsonl'
DRAFTS = REVIEW / 'instruction_drafts.json'
LOCK = threading.RLock()
MAX_BODY = 16 * 1024
TASK_FAMILIES = {'TURN_WAIT', 'TURN_CHOICE', 'TURN_STRAIGHT', 'STOP_GO',
                 'LATERAL_CHOICE', 'SPEED_CHOICE', 'OTHER', 'UNRESOLVED'}


def read_queue():
    with QUEUE.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    ids = [r['candidate_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate candidate IDs in review queue')
    return rows, set(ids)


def read_decisions():
    if not DECISIONS.exists():
        return {}
    data = json.loads(DECISIONS.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('decisions.json must contain an object')
    return data


def read_drafts():
    data = json.loads(DRAFTS.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('instruction_drafts.json must contain an object')
    return data


def candidate_view(row, decisions, drafts):
    key = row['candidate_id']
    draft = drafts.get(key, {})
    saved = decisions.get(key, {})
    result = {'candidate_id': key, 'frame_name': row['frame_name'],
              'reference_a': row['reference_a'], 'reference_b': row['reference_b'],
              'decision': saved.get('decision', 'PENDING'),
              'reviewer': saved.get('reviewer', ''), 'note': saved.get('note', ''),
              'updated_at': saved.get('updated_at', ''),
              'task_confirmed': saved.get('task_confirmed', False),
              'instruction_language': 'en',
              'task_source': saved.get('task_source', 'ego_motion_draft'),
              'review_checks': draft.get('review_checks', '')}
    for field in ('task_family', 'task_summary', 'instruction_a', 'instruction_b'):
        result[field] = saved.get(field, draft.get(field, ''))
    return result


def save_decisions(data):
    fd, temp = tempfile.mkstemp(prefix='.decisions-', suffix='.json', dir=REVIEW)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, DECISIONS)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class Handler(BaseHTTPRequestHandler):
    server_version = 'WODReview/1.0'

    def respond(self, status, body, mime, disposition=None):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self' 'unsafe-inline' data:; frame-ancestors 'none'")
        if disposition:
            self.send_header('Content-Disposition', disposition)
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, status, data):
        self.respond(status, json.dumps(data, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/candidates':
            with LOCK:
                rows, _ = read_queue()
                decisions = read_decisions()
                drafts = read_drafts()
                payload = [candidate_view(r, decisions, drafts) for r in rows]
            return self.json_response(200, {'candidates': payload})
        if path == '/api/review_queue.csv':
            with LOCK:
                rows, _ = read_queue()
                decisions = read_decisions()
                drafts = read_drafts()
                out = io.StringIO()
                extra = ['task_family', 'task_summary', 'instruction_language',
                         'instruction_a', 'instruction_b',
                         'task_confirmed', 'task_source', 'review_checks', 'updated_at']
                writer = csv.DictWriter(out, fieldnames=list(rows[0]) + extra)
                writer.writeheader()
                for r in rows:
                    row = dict(r)
                    view = candidate_view(r, decisions, drafts)
                    for name in extra:
                        row[name] = view[name]
                    d = decisions.get(r['candidate_id'])
                    if d:
                        row['status'] = 'REVIEWED' if d['decision'] != 'PENDING' else 'DISCOVERED'
                        row['decision'] = d['decision']
                        row['reviewer'] = d['reviewer']
                        row['reason'] = d['note']
                    writer.writerow(row)
            return self.respond(200, out.getvalue().encode('utf-8'), 'text/csv; charset=utf-8',
                                'attachment; filename="wod_review_queue_current.csv"')
        if path == '/api/model_review_priority.csv':
            target = REVIEW / 'model_review_priority.csv'
            if not target.is_file():
                return self.json_response(404, {'error': 'model review worklist not ready'})
            return self.respond(200, target.read_bytes(), 'text/csv; charset=utf-8',
                                'attachment; filename="wod_model_review_priority.csv"')
        if path == '/api/model_predictions':
            target = REVIEW / 'model_predictions.json'
            if not target.is_file():
                return self.json_response(404, {'error': 'model trajectories not ready'})
            return self.respond(200, target.read_bytes(), 'application/json; charset=utf-8')
        if path == '/':
            path = '/index.html'
        if path in ('/index.html', '/build_report.json') or (path.startswith('/assets/') and '/' not in path[8:]):
            name = path.removeprefix('/')
            target = REVIEW / name
            if not target.is_file():
                return self.json_response(404, {'error': 'not found'})
            mime = 'image/jpeg' if name.endswith('.jpg') else ('text/html; charset=utf-8' if name.endswith('.html') else 'application/json')
            return self.respond(200, target.read_bytes(), mime)
        return self.json_response(404, {'error': 'not found'})

    def do_POST(self):
        if urlsplit(self.path).path != '/api/decision':
            return self.json_response(404, {'error': 'not found'})
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            return self.json_response(415, {'error': 'send application/json'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > MAX_BODY:
                return self.json_response(413, {'error': 'invalid request size'})
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('JSON object required')
            candidate_id = body.get('candidate_id')
            decision = body.get('decision')
            reviewer = body.get('reviewer', '')
            note = body.get('note', '')
            if not isinstance(candidate_id, str) or not isinstance(decision, str):
                raise ValueError('candidate_id and decision must be strings')
            if decision not in ('PENDING', 'PASS', 'FAIL'):
                raise ValueError('decision must be PENDING, PASS or FAIL')
            if body.get('instruction_language', 'en') != 'en':
                raise ValueError('benchmark instructions must use English (en)')
            if not isinstance(reviewer, str) or len(reviewer) > 80:
                raise ValueError('reviewer must be at most 80 characters')
            if not isinstance(note, str) or len(note) > 2000:
                raise ValueError('note must be at most 2000 characters')
            if any(ord(ch) < 32 and ch not in '\n\t' for ch in reviewer + note):
                raise ValueError('invalid control character')
            with LOCK:
                _, ids = read_queue()
                if candidate_id not in ids:
                    return self.json_response(404, {'error': 'unknown candidate'})
                decisions = read_decisions()
                drafts = read_drafts()
                old = decisions.get(candidate_id, {})
                proposed = drafts.get(candidate_id, {})
                task_fields = {}
                for field, limit in (('task_family', 40), ('task_summary', 500),
                                     ('instruction_a', 500), ('instruction_b', 500)):
                    value = body.get(field, old.get(field, proposed.get(field, '')))
                    if not isinstance(value, str) or len(value) > limit:
                        raise ValueError(f'{field} must be text of at most {limit} characters')
                    task_fields[field] = value.strip()
                if task_fields['task_family'] not in TASK_FAMILIES:
                    raise ValueError('invalid task_family')
                task_confirmed = body.get('task_confirmed', old.get('task_confirmed', False))
                if not isinstance(task_confirmed, bool):
                    raise ValueError('task_confirmed must be a boolean')
                if decision == 'PASS' and (not task_confirmed or
                                           task_fields['task_family'] == 'UNRESOLVED' or
                                           not all(task_fields[f] for f in ('task_summary', 'instruction_a', 'instruction_b')) or
                                           task_fields['instruction_a'] == task_fields['instruction_b']):
                    raise ValueError('通过前须确认明确任务、两条不同的英文指令以及已核对场景')
                if decision == 'PASS' and any(
                    not re.search(r'[A-Za-z]', task_fields[field]) or
                    not task_fields[field].isascii()
                    for field in ('instruction_a', 'instruction_b')
                ):
                    raise ValueError('通过前两条指令必须使用英文 ASCII 文本')
                edited = any(task_fields[f] != proposed.get(f, '') for f in task_fields)
                record = {'decision': decision, 'reviewer': reviewer.strip(),
                          'note': note.strip(), 'updated_at': datetime.now(timezone.utc).isoformat(),
                          **task_fields, 'instruction_language': 'en',
                          'task_confirmed': task_confirmed,
                          'task_source': 'human_confirmed' if task_confirmed else
                                         ('human_draft' if edited else 'ego_motion_draft')}
                decisions[candidate_id] = record
                save_decisions(decisions)
                with EVENTS.open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'candidate_id': candidate_id, **record}, ensure_ascii=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
            return self.json_response(200, {'candidate_id': candidate_id, **record})
        except (ValueError, json.JSONDecodeError) as e:
            return self.json_response(400, {'error': str(e)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8787)
    args = parser.parse_args()
    read_queue()
    read_drafts()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'WOD private review: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()

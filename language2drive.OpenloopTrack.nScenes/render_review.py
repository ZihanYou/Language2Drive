#!/usr/bin/env python3
"""Render local map geometry, paired witnesses and camera evidence for review."""
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw
from shapely.geometry import Point
from build_lane_pcus import MapGeometry
from paths import ARTIFACT_ROOT,DATA_ROOT


def polygon_parts(shape):
    return [shape] if shape.geom_type=='Polygon' else list(shape.geoms)


def main():
    audit=ARTIFACT_ROOT/'audit'
    for folder in ('map_overlays','cameras'):(audit/folder).mkdir(parents=True,exist_ok=True)
    source=ARTIFACT_ROOT/'candidates/all_review_queue.jsonl'
    maps={};cards=[];diagnostics={}
    for name in ('lidar_diagnostics','footprint_swept_diagnostics'):
        path=audit/(name+'.jsonl')
        diagnostics[name]={r['pcu_id']:r for r in
                           (json.loads(s) for s in path.read_text().splitlines() if s.strip())} if path.exists() else {}
    rows=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    for index,p in enumerate(rows,1):
        loc=p['public_input']['location']
        if loc not in maps:maps[loc]=MapGeometry(loc)
        geom=maps[loc];anchor=p['public_input']['ego_pose']['xy_m']
        tile=Image.new('RGB',(780,780),'#142033');draw=ImageDraw.Draw(tile)
        def pixel(xy):return (390+(xy[0]-anchor[0])*6,390-(xy[1]-anchor[1])*6)
        query=Point(anchor).buffer(65)
        for shape in geom.drivable_tree.query(query):
            for polygon in polygon_parts(shape):
                draw.polygon([pixel(xy) for xy in polygon.exterior.coords],fill='#36475d')
                for ring in polygon.interiors:draw.polygon([pixel(xy) for xy in ring.coords],fill='#142033')
        for shape in geom.tree.query(query):
            for polygon in polygon_parts(shape):draw.line([pixel(xy) for xy in polygon.exterior.coords],fill='#758799',width=1)
        keys=('witness_change','witness_keep') if p['family']=='lane_change_keep' else ('witness_low','witness_high')
        for key,color in zip(keys,('#ff4f87','#ffffff')):
            points=[pixel(xy) for xy in p['private_sidecar'][key]['global_xy_m']]
            draw.line(points,fill=color,width=4)
        draw.ellipse((384,384,396,396),fill='#54cfff')
        filename='{:03d}.png'.format(index);tile.save(audit/'map_overlays'/filename)
        cameras=[]
        for channel in ('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'):
            ref=p['public_input']['sensor_refs'][channel]
            with Image.open(DATA_ROOT/ref['filename']) as image:
                image=image.convert('RGB');image.thumbnail((800,450))
                name='{:03d}_{}.jpg'.format(index,channel);image.save(audit/'cameras'/name,quality=85)
            cameras.append('<figure><figcaption>{}</figcaption><img src="cameras/{}"></figure>'.format(channel,name))
        evidence=''.join('<details><summary>{}</summary><pre>{}</pre></details>'.format(
            name,html.escape(json.dumps(records.get(p['pcu_id'],{}),indent=2)))
            for name,records in diagnostics.items())
        cards.append('<article id="candidate-{:03d}"><h2>{}</h2><p>Pink: arm 0; white: arm 1; blue: anchor.</p><img src="map_overlays/{}"><div class="cameras">{}</div>{}</article>'.format(index,html.escape(p['pcu_id']),filename,''.join(cameras),evidence))
    document='<!doctype html><meta charset="utf-8"><title>PCU review</title><style>body{font:16px system-ui;background:#142033;color:white}article{max-width:1100px;margin:30px auto}img{max-width:100%}.cameras{display:grid;grid-template-columns:repeat(3,1fr)}figure{margin:8px}</style><h1>PCU review queue</h1>'+''.join(cards)
    (audit/'index.html').write_text(document)
    print('Rendered {} candidates'.format(len(rows)))


if __name__=='__main__':main()

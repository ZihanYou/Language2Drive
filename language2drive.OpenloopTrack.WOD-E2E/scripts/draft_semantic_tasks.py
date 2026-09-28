"""Draft pair-specific language for human review from rated ego motion only.

The drafts are proposals, not semantic ground truth: images, legality, referents,
route neutrality and both-arm feasibility must be checked by a human reviewer.
"""
import csv
import json
import math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'review'


def motion(t):
    points = t['xy_m']
    speeds = [math.dist(p, points[i-1] if i else [0, 0]) * 4 for i, p in enumerate(points)]
    dx = points[19][0] - points[15][0]
    dy = points[19][1] - points[15][1]
    heading = math.degrees(math.atan2(dy, dx)) if math.hypot(dx, dy) > 0.3 else 0
    return {'x5':points[19][0], 'y5':points[19][1],
            'last_speed':sum(speeds[-4:])/4, 'path':sum(speeds)/4,
            'heading':heading, 'stop':sum(speeds[-4:])/4 < 0.6}


def turn_side(m):
    if m['path'] >= 5 and abs(m['heading']) >= 30 and abs(m['y5']) >= 3:
        return '左' if m['heading'] > 0 else '右'
    return None


def draft(a, b):
    ma, mb = motion(a), motion(b)
    motions = [ma, mb]
    side = [turn_side(ma), turn_side(mb)]
    instructions = ['', '']
    family = 'UNRESOLVED'
    summary = '当前两条轨迹尚未形成可稳定描述的互斥驾驶任务。请结合场景决定是否剔除或重写指令。'
    checks = '检查是否存在可观察且都合理的两种选择；仅有几何差异不能判为语言任务。'
    if ma['stop'] != mb['stop'] and abs(ma['path']-mb['path']) >= 3:
        stopped = 0 if ma['stop'] else 1
        moving = 1-stopped
        if side[moving]:
            family = 'TURN_WAIT'
            summary = f'同一路口的等待与向{side[moving]}转向选择（需核查是否真的存在可见路口）。'
            instructions[stopped] = 'Stop before this intersection and remain there.'
            instructions[moving] = f'Turn {"left" if side[moving] == "左" else "right"} at this intersection and continue driving.'
            checks = '核对路口、转向入口和停车位置是否可见，转向是否已获允许；停车和转向能否在同一初始状态下独立完成。'
        else:
            family = 'STOP_GO'
            summary = '在当前场景中停车等待，或继续沿前方可通行路线行驶。'
            instructions[stopped] = 'Stop smoothly at a safe position ahead and wait.'
            instructions[moving] = 'Continue driving along the drivable route ahead.'
            checks = '核对停车位置、交通控制与前方通道；继续通行必须确实安全合法，不能仅靠高评分推断。'
    elif side[0] and side[1] and side[0] != side[1]:
        family = 'TURN_CHOICE'
        summary = '在可见分叉或路口选择不同转向分支。'
        for i in (0,1): instructions[i] = f'Turn {"left" if side[i] == "左" else "right"} at the intersection ahead and continue driving.'
        checks = '核对两处合法出口都可见且5秒内可判定；原始route command不可预先指定某一分支。'
    elif bool(side[0]) != bool(side[1]) and min(ma['path'],mb['path']) >= 5:
        turning = 0 if side[0] else 1
        straight = 1-turning
        if abs(motions[straight]['heading']) < 15:
            family = 'TURN_STRAIGHT'
            summary = f'前方路口向{side[turning]}转，或直行通过。'
            instructions[turning] = f'Turn {"left" if side[turning] == "左" else "right"} at the intersection ahead and continue driving.'
            instructions[straight] = 'Go straight through the intersection ahead and continue driving.'
            checks = '核对两条合法分支与路口是否可见；必须移除泄漏转向答案的原始route command。'
    if family == 'UNRESOLVED' and abs(ma['y5']-mb['y5']) >= 2:
        family = 'LATERAL_CHOICE'
        summary = '沿不同横向通行位置继续行驶；需确认是真实可选路径，而非弯道坐标差。'
        if abs(ma['y5']) < 1.2 and abs(mb['y5']) >= 2.5:
            stay, shift = 0,1
        elif abs(mb['y5']) < 1.2 and abs(ma['y5']) >= 2.5:
            stay, shift = 1,0
        else:
            stay = shift = None
        if stay is not None:
            instructions[stay] = 'Keep your current lateral position and continue along the drivable route ahead.'
            instructions[shift] = f'Move smoothly into the drivable space on the {"left" if motions[shift]["y5"]>0 else "right"} and continue driving.'
        else:
            left = 0 if ma['y5'] > mb['y5'] else 1
            instructions[left] = 'Take the drivable path farther to the left ahead and continue driving.'
            instructions[1-left] = 'Take the drivable path farther to the right ahead and continue driving.'
        checks = '核对通行空间、车道/通道边界、转向和route是否一致；ego-frame横移本身不能证明存在合法左右选择。'
    if family == 'UNRESOLVED' and abs(ma['x5']-mb['x5']) >= 4:
        family = 'SPEED_CHOICE'
        summary = '在同一路径上选择较慢或较快的前进节奏。'
        slow = 0 if ma['path'] < mb['path'] else 1
        instructions[slow] = 'Slow down and proceed cautiously along the current drivable route.'
        instructions[1-slow] = 'Maintain a normal driving speed and continue along the current drivable route.'
        checks = '核对两条轨迹确实沿同一路径，均符合当前交通状况；冻结不重叠的速度范围后才可成为契约。'
    return {'task_family':family, 'task_summary':summary,
            'instruction_language':'en',
            'instruction_a':instructions[0], 'instruction_b':instructions[1],
            'review_checks':checks, 'source':'ego_motion_draft',
            'motion_evidence':{'a':ma,'b':mb}}


def main():
    rows={r['frame_name']:r for line in (ROOT/'artifacts/normalized_rated_frames.jsonl').open()
          if (r:=json.loads(line))}
    with (REVIEW/'review_queue.csv').open(newline='') as f: queue=list(csv.DictReader(f))
    result={}
    for row in queue:
        source=rows[row['frame_name']]
        refs={str(t['reference_index']):t for t in source['preferences']}
        item=draft(refs[row['reference_a']],refs[row['reference_b']])
        item.update(frame_name=row['frame_name'],reference_a=row['reference_a'],reference_b=row['reference_b'])
        result[row['candidate_id']]=item
    (REVIEW/'instruction_drafts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(dict(Counter(r['task_family'] for r in result.values())))

if __name__ == '__main__': main()

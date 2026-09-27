"""Private review labels bound to the executed pair, not inferred qualification."""
import math


def pair_review_timing(sidecar):
    """One explicit video horizon, keeping legacy seven-second preview times."""
    if sidecar.get("hypothesis_kind") == "speed" and "speed_timing" in sidecar:
        from .speed_timing import speed_timing
        duration = speed_timing(sidecar)["rollout_duration_s"]
        if len(sidecar["sessions"]) != 2 or any(p.get("rollout_duration_s", duration) != duration
                                               for p in sidecar["sessions"].values()):
            raise ValueError("speed review horizons differ from the explicit timing")
        # B3 authoring also requires a three-second pre-scope view. Preserve it
        # while showing the true end of a longer run, not only its first seven seconds.
        return duration, ([1.5, 3., 4., 5., 6., 7.] if duration == 7 else
                          [1.5, 3., duration-3, duration-2, duration-1, duration])
    values = [p.get("rollout_duration_s", 7.) for p in sidecar["sessions"].values()]
    if (len(values) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 7 for v in values)
        or values[0] != values[1]):
        raise ValueError("review requires equal finite horizons of at least seven seconds")
    duration = values[0]
    times = [1.5, 3., 4., 5., 6., 7.] if duration == 7 else [1.5, duration / 2, duration - 3, duration - 2, duration - 1, duration]
    if len(set(times)) != 6 or sorted(times) != times:
        raise ValueError("review preview schedule must be distinct and ordered")
    return duration, times


def pair_review_labels(launch, sidecar, runtime_return, pair_kind):
    if pair_kind not in {"lane", "speed", "branch", "scoped_lane"} or sidecar.get("hypothesis_kind", "lane") != pair_kind:
        raise ValueError("review pair kind mismatch")
    sessions = sidecar["sessions"]
    if len(sessions) != 2 or sorted(p["arm_id"] for p in sessions.values()) != ["R0", "R1"]:
        raise ValueError("review requires both actual arms")
    returns = runtime_return["rollout_returns"]
    if (len(returns) != 2 or {r["session_uuid"] for r in returns} != set(sessions)
        or any(r.get("success") is not True for r in returns)):
        raise ValueError("review requires successful returns for the exact pair")
    if pair_kind == "scoped_lane":
        selected = launch["candidate"]["selected_arm_pair"]
        hypothesis = launch["scoped_lane_hypothesis"]
        duration, _ = pair_review_timing(sidecar)
        side = {"keep_vs_left": ("change_left", "左侧"), "keep_vs_right": ("change_right", "右侧")}.get(selected["kind"])
        if (side is None or [(a["arm"], a["role"]) for a in selected["arms"]] != [("R0", "keep"), ("R1", side[0])]
            or len({str(a["lane_id"]) for a in selected["arms"]}) != 2
            or hypothesis.get("qualified") is not False or hypothesis.get("human_reviewed") is not False
            or hypothesis.get("formal_eligibility_authorized") is not False
            or hypothesis.get("native_metrics_preserved") is not True
            or hypothesis["rollout_duration_s"] != duration):
            raise ValueError("bound exploratory B2 roles, duration and preserved native metrics required")
        return {"R0": "范围内保持原始车道", "R1": "范围内选择" + side[1] + "车道"}, (
            "这是 B2 同范围车道偏好，不是 A5 短时车道驻留；请检查共同起止边界、同一实体终点、"
            "整个范围内的车道选择及同一主要路线。原生指标和逆向告警保留，规划诊断不等于实际合法性批准。"
            "视频仅覆盖已执行的 %g 秒，后续六秒预测不代表已经执行；请独立检查交通信号、目标可见性和重建质量。" % duration)
    if pair_kind == "lane":
        selected = launch["candidate"]["selected_arm_pair"]
        if selected["kind"] == "left_vs_right":
            if (sidecar.get("lane_pair_kind") != "left_vs_right"
                or [(a["arm"], a["role"]) for a in selected["arms"]] != [
                    ("R0", "change_left"), ("R1", "change_right")]
                or len({str(a["lane_id"]) for a in selected["arms"]}) != 2):
                raise ValueError("bilateral review labels require exact left/right binding")
            return {"R0": "左换道", "R1": "右换道"}, (
                "本项为原始中间车道出发的左换道／右换道，不含保持车道臂。"
                "请核对初始三车道、双侧标线和动作合法性、交通信号及完整渲染；自动预筛不代表人工批准。")
        change = {"keep_vs_left": "左换道", "keep_vs_right": "右换道"}[
            launch["candidate"]["selected_arm_pair"]["kind"]]
        return {"R0": "保持车道", "R1": change}, "请核对车道线与两种动作的合法性。"
    if pair_kind == "branch":
        selected = launch["candidate"]["selected_arm_pair"]
        hypothesis = launch["branch_hypothesis"]
        duration, _ = pair_review_timing(sidecar)
        if (selected["kind"] != "recorded_vs_counterfactual_branch"
            or sorted(a["arm"] for a in selected["arms"]) != ["R0", "R1"]
            or len({a["lane_id"] for a in selected["arms"]}) != 2
            or hypothesis.get("qualified") is not False
            or hypothesis.get("human_reviewed") is not False
            or hypothesis["rollout_duration_s"] != duration):
            raise ValueError("reviewed branch identities and exploratory duration required")
        ids = {a["arm"]: str(a["lane_id"]) for a in selected["arms"]}
        mode = hypothesis.get("plan_mode", "yield_after_branch")
        if mode not in {"continuing", "yield_after_branch"}:
            raise ValueError("unknown branch plan mode")
        future_notice = (
            "本方案为曲率限速路径，可能在已确认路径终端前减速；视频只覆盖已执行的 %g 秒，"
            "不证明预测未来已执行或已经完全停车。" % duration if mode == "continuing" else
            "完整预测计划还包含驶出所选分支后的减速等待；视频只覆盖已执行的 %g 秒，"
            "不证明预测未来已执行或已经完全停车。等待位置是否允许停车须另行核实，不能从静止 actor 推断；" % duration)
        return {"R0": "所选分支 R0（lane " + ids["R0"] + "）",
                "R1": "所选分支 R1（lane " + ids["R1"] + "）"}, (
            "这是新分支轨迹与减速方案，原时间窗初审不覆盖本方案；lane 编号不是正式出口语义。"
            "原发现的 recorded 标注不证明新窗口的记录车辆经过该出口，也不表示本次轨迹是记录重放。"
            "请独立检查两臂是否进入不同出口、转向/交通信号/让行是否合法、渲染和遮挡是否可接受。"
            + future_notice +
            "没有足够依据时请选待定，在备注注明秒数、相机和缺失证据。"
            "派生地图连接、非黑图或无碰撞均不替代人工审核与连续重建支持。")
    hypothesis = launch["speed_hypothesis"]
    if hypothesis.get("initial_lane_pair_review_does_not_cover_speed_pair") is not True:
        raise ValueError("new speed hypothesis must not inherit lane approval")
    by_arm = {p["arm_id"]: p for p in sessions.values()}
    targets = [by_arm[arm]["target_speed_kph"] for arm in ("R0", "R1")]
    if targets != hypothesis["targets_kph"]:
        raise ValueError("launch and executed speed labels mismatch")
    bands = []
    labels = {}
    for arm in ("R0", "R1"):
        private = by_arm[arm]
        target = private["target_speed_kph"]
        band = private["diagnostic_speed_band_mps"]
        if (len(band) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                                for v in [target, *band])
            or not 0 <= band[0] < target / 3.6 < band[1]):
            raise ValueError("invalid speed review band")
        bands.append(band)
        labels[arm] = "目标 %g km/h（诊断区间 %g–%g km/h）" % (target, band[0] * 3.6, band[1] * 3.6)
    if not (bands[0][1] < bands[1][0] or bands[1][1] < bands[0][0]):
        raise ValueError("review speed bands overlap")
    scope_notice = ("5–7 秒保留探索性速度诊断，原横道范围在实跑前已绑定，仍须核验实际完整范围。"
                    if sidecar.get("crosswalk_scope_reference") else
                    "5–7 秒仅为探索性速度诊断区间，共享空间 scope 尚未冻结。")
    return labels, (
        "这是新速度方案，不是保持/换道任务，原换道初审不覆盖本方案。"
        + scope_notice +
        "请独立核对两臂适用限速、交通信号与动作合法性，并在备注注明依据；"
        "地图原始限速字段为 0 且单位未声明，不能据此认定合法。"
        "仅凭视频无法确认限速时，请选待定；速度数值诊断不替代人工合法性审核。")

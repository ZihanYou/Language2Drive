"""Human review of real paired evidence, separate from initial curation/qualification."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import re
import threading

from .io import atomic_json, load_json


SCHEMA = "l2d.realsim.feasibility_reviews.v0.1"
ARM_FIELDS = ("legal_action", "render_quality", "observable")
CAMERAS = {"camera_front_wide_120fov", "camera_cross_left_120fov",
           "camera_cross_right_120fov", "camera_front_tele_30fov"}


class FeasibilityReviewStore:
    def __init__(self, initial_store, packet_root, decisions_path):
        self.initial_store = initial_store
        self.packet_root = Path(packet_root).resolve()
        self.decisions_path = Path(decisions_path)
        self._lock = threading.RLock()

    def _authority_binding(self, proposal, inventory):
        if inventory.get("curation_authority") == "agent_under_user_delegation_pending_final_review":
            raise ValueError("自主编制证据必须使用独立审核入口")
        initial = self.initial_store.snapshot()["decisions"].get(proposal, {})
        if initial.get("decision") != "keep" or initial.get("revision") != inventory.get("review_revision"):
            raise ValueError("初审已变更或不是 keep；请重新核对证据包")
        return {"initial_review_revision": initial["revision"]}

    def _packet(self, proposal, packet):
        if any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,160}", v)
               for v in (proposal, packet)):
            raise ValueError("无效证据包编号")
        directory = (self.packet_root / proposal / packet).resolve()
        if not directory.is_relative_to(self.packet_root):
            raise ValueError("证据包路径越界")
        inventory_path = directory / "inventory.private.json"
        inventory = load_json(inventory_path)
        if (inventory.get("scope") != "human_review_packet_not_qualification"
            or inventory.get("proposal_key") != proposal):
            raise ValueError("不是受支持的真实双臂审核包")
        authority = self._authority_binding(proposal, inventory)
        videos = inventory.get("videos", [])
        if (len(videos) != 8 or {(v["arm"], v["camera"]) for v in videos}
            != {(arm, camera) for arm in ("R0", "R1") for camera in CAMERAS}):
            raise ValueError("缺少双臂四相机视频")
        for video in videos:
            name = video["file"]
            path = (directory / name).resolve()
            if (Path(name).name != name or not path.is_relative_to(directory)
                or not path.is_file() or path.stat().st_size != video["size_bytes"]):
                raise ValueError("视频缺失或大小已改变")
        stat = inventory_path.stat()
        # Nanoseconds exceed JavaScript's safe integer range. Keep the exact
        # stat binding across browser JSON parse/stringify without rounding or
        # weakening the stale-evidence check to milliseconds/a tolerance.
        return {"proposal_key": proposal, "packet_name": packet, "seed": inventory["seed"],
                **authority,
                "inventory_size_bytes": stat.st_size, "inventory_mtime_ns": str(stat.st_mtime_ns)}

    def _log(self):
        if not self.decisions_path.exists():
            return {"schema_version": SCHEMA, "decisions": {}, "audit_log": []}
        log = load_json(self.decisions_path)
        if (log.get("schema_version") != SCHEMA or not isinstance(log.get("decisions"), dict)
            or not isinstance(log.get("audit_log"), list)):
            raise ValueError("补充审核日志格式无效")
        return log

    def snapshot(self, proposal, packet):
        with self._lock:
            binding = self._packet(proposal, packet)
            record = self._log()["decisions"].get(proposal + "/" + packet)
            return {"binding": binding, "record": record, "qualified": False,
                    "revision": record["revision"] if record else 0}

    def submit(self, proposal, packet, payload):
        with self._lock:
            snapshot = self.snapshot(proposal, packet)
            if not isinstance(payload, dict):
                raise ValueError("提交内容必须是对象")
            allowed = {"binding", "expected_revision", "reviewer", "decision", "notes",
                       "arms", "semantically_distinct", "watched_all_camera_videos"}
            if set(payload) - allowed:
                raise ValueError("包含不受支持的字段；不能通过页面设置正式资格")
            if payload.get("binding") != snapshot["binding"]:
                raise ValueError("证据包已变化，请刷新后重新审核")
            if type(payload.get("expected_revision")) is not int or payload["expected_revision"] != snapshot["revision"]:
                raise ValueError("审核结果已被其他页面更新，请刷新后再提交")
            reviewer, notes = payload.get("reviewer"), payload.get("notes", "")
            if not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 100:
                raise ValueError("审核人必须为 1–100 个字符")
            if not isinstance(notes, str) or len(notes) > 4000:
                raise ValueError("备注必须是最多 4000 字符的文本")
            decision = payload.get("decision")
            if not isinstance(decision, str) or decision not in {"keep", "pending", "reject"}:
                raise ValueError("结论必须为 keep、pending 或 reject")
            if decision != "keep" and not notes.strip():
                raise ValueError("待定或拒绝必须注明原因及相机/秒数")
            arms = payload.get("arms")
            if not isinstance(arms, dict) or set(arms) != {"R0", "R1"}:
                raise ValueError("必须分别审核两个 arm")
            for arm in arms.values():
                if not isinstance(arm, dict) or set(arm) != set(ARM_FIELDS):
                    raise ValueError("每臂需要合法性、渲染质量和可观察性三项判断")
            values = [v for arm in arms.values() for v in arm.values()] + [payload.get("semantically_distinct")]
            if any(v is not None and type(v) is not bool for v in values):
                raise ValueError("判断必须为 true、false 或 null")
            watched = payload.get("watched_all_camera_videos")
            if type(watched) is not bool:
                raise ValueError("请明确是否看完双臂四相机视频")
            if decision == "keep" and (not watched or any(v is not True for v in values)):
                raise ValueError("保留前必须看完视频并明确通过全部判断；不确定请选待定")
            timestamp = datetime.now(timezone.utc).isoformat()
            record = {"binding": snapshot["binding"], "revision": snapshot["revision"] + 1,
                      "reviewer": reviewer.strip(), "reviewed_at": timestamp, "decision": decision,
                      "notes": notes.strip(), "arms": copy.deepcopy(arms),
                      "semantically_distinct": payload.get("semantically_distinct"),
                      "watched_all_camera_videos": watched,
                      "scope": "human_visual_legal_review_not_formal_qualification",
                      "qualified": False, "continuous_support_verified": False}
            log = self._log()
            log["decisions"][proposal + "/" + packet] = record
            log["audit_log"].append(copy.deepcopy(record))
            log["updated_at"] = timestamp
            atomic_json(self.decisions_path, log)
            return {"record": record, "qualified": False}


class AgentFeasibilityReviewStore(FeasibilityReviewStore):
    """Human review of delegated pilot videos, NOT approval of a PCU source.

    The immutable packet is the review target, not any source draft that may
    later reuse its videos. No initial human decision is invented or required.
    """

    def _authority_binding(self, proposal, inventory):
        from .agent_curation import AUTHORITY
        if (inventory.get("curation_authority") != AUTHORITY
            or inventory.get("final_human_review_pending") is not True
            or inventory.get("review_revision") is not None
            or inventory.get("review_batch") is not None):
            raise ValueError("需要独立的自主编制待审证据包，不能混用初审授权")
        return {"curation_authority": AUTHORITY,
                "review_target": "paired_pilot_videos_not_pcu_source_approval"}

    def _packet(self, proposal, packet):
        binding = super()._packet(proposal, packet)
        directory = self.packet_root / proposal / packet
        inventory = load_json(directory / "inventory.private.json")
        # Stat-only stale checks: no repeated hashing or decoding of large videos.
        binding["video_stats"] = [
            {"file": video["file"],
             "size_bytes": (directory / video["file"]).stat().st_size,
             "mtime_ns": str((directory / video["file"]).stat().st_mtime_ns)}
            for video in inventory["videos"]
        ]
        return binding

    def snapshot(self, proposal, packet):
        result = super().snapshot(proposal, packet)
        record = result["record"]
        result["previous_review_stale"] = bool(record and record["binding"] != result["binding"])
        if result["previous_review_stale"]:
            # Retain revision/history for concurrency, but never prefill approval
            # from a previous version of the evidence now shown in the browser.
            result["record"] = None
        return result

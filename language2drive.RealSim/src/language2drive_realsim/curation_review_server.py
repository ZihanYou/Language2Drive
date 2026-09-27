"""Loopback-only HTTP service for durable private curation decisions."""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlsplit

from .curation_review import (
    REJECTION_REASONS,
    TRISTATE_REVIEW_FIELDS,
    arm_pair_candidates,
)
from .io import atomic_json, load_json
from .feasibility_review import FeasibilityReviewStore, AgentFeasibilityReviewStore
from .feasibility_review_ui import REVIEW_SCRIPT


SCHEMA_VERSION = "l2d.realsim.curation_decisions.v0.1"
DECISIONS = {"keep", "pending", "reject"}
RECONSTRUCTION_STATES = {"unverified", "verified", "failed"}
MAX_BODY_BYTES = 32 * 1024


class ReviewValidationError(ValueError):
    """A submitted review is invalid or not backed by the inventory."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStore:
    """Validate and atomically persist current decisions plus revision history."""

    def __init__(self, inventory_path: Path, decisions_path: Path) -> None:
        self.inventory_path = Path(inventory_path).resolve()
        self.decisions_path = Path(decisions_path).resolve()
        self._lock = threading.RLock()
        self._inventory_mtime_ns: Optional[int] = None
        self._proposals: Dict[str, Dict[str, Any]] = {}
        self._refresh_inventory(force=True)

    def _refresh_inventory(self, force: bool = False) -> None:
        try:
            mtime_ns = self.inventory_path.stat().st_mtime_ns
        except FileNotFoundError as error:
            raise ReviewValidationError(
                "审核库存不存在：%s；请先运行 prepare-curation" % self.inventory_path
            ) from error
        if not force and mtime_ns == self._inventory_mtime_ns:
            return
        inventory = load_json(self.inventory_path)
        proposals: Dict[str, Dict[str, Any]] = {}
        for scene in inventory.get("scenes", []):
            for proposal in scene.get("proposals", []):
                if not proposal.get("timing_review", {}).get(
                    "automatic_timing_prefilter_pass"
                ):
                    continue
                key = proposal.get("proposal_key")
                if not isinstance(key, str) or not key:
                    raise ReviewValidationError("审核库存包含无效 proposal_key")
                if key in proposals:
                    raise ReviewValidationError("审核库存包含重复 proposal_key：" + key)
                pairs = arm_pair_candidates(proposal)
                proposals[key] = {
                    "artifact_uuid": scene.get("artifact_uuid"),
                    "scene_id_private": scene.get("scene_id_private"),
                    "proposal": proposal,
                    "arm_pairs": {row["arm_pair_id"]: row for row in pairs},
                }
        self._proposals = proposals
        self._inventory_mtime_ns = mtime_ns

    def _empty_log(self) -> Dict[str, Any]:
        timestamp = _now()
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": timestamp,
            "updated_at": timestamp,
            "decisions": {},
            "audit_log": [],
        }

    def _load_log(self) -> Dict[str, Any]:
        if not self.decisions_path.is_file():
            return self._empty_log()
        payload = load_json(self.decisions_path)
        if not isinstance(payload, dict):
            raise ReviewValidationError("审核日志根节点必须是对象")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ReviewValidationError("审核日志 schema_version 不受支持")
        if not isinstance(payload.get("decisions"), dict):
            raise ReviewValidationError("审核日志 decisions 必须是对象")
        audit_log = payload.setdefault("audit_log", [])
        if not isinstance(audit_log, list):
            raise ReviewValidationError("审核日志 audit_log 必须是数组")
        payload.setdefault("created_at", _now())
        payload.setdefault("updated_at", payload["created_at"])
        return payload

    @staticmethod
    def _nullable_boolean(payload: Dict[str, Any], field: str) -> Optional[bool]:
        value = payload.get(field)
        if value is not None and not isinstance(value, bool):
            raise ReviewValidationError("%s 必须是 true、false 或 null" % field)
        return value

    def validate_review(
        self, proposal_key: str, payload: Any
    ) -> Dict[str, Any]:
        if proposal_key not in self._proposals:
            raise ReviewValidationError("proposal_key 不存在或未通过自动时序筛选")
        if not isinstance(payload, dict):
            raise ReviewValidationError("提交内容必须是 JSON 对象")

        reviewer = payload.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ReviewValidationError("审核人不能为空")
        reviewer = reviewer.strip()
        if len(reviewer) > 100:
            raise ReviewValidationError("审核人不得超过 100 个字符")

        decision = payload.get("decision")
        if decision not in DECISIONS:
            raise ReviewValidationError("审核结论必须是 keep、pending 或 reject")

        notes = payload.get("notes", "")
        if not isinstance(notes, str):
            raise ReviewValidationError("备注必须是字符串")
        notes = notes.strip()
        if len(notes) > 4000:
            raise ReviewValidationError("备注不得超过 4000 个字符")
        if decision == "pending" and not notes:
            raise ReviewValidationError("待定时必须在备注中写明缺少的证据")

        reconstruction = payload.get("reconstruction_support_status", "unverified")
        if reconstruction not in RECONSTRUCTION_STATES:
            raise ReviewValidationError(
                "重建支持状态必须是 unverified、verified 或 failed"
            )

        values = {
            field: self._nullable_boolean(payload, field)
            for field in TRISTATE_REVIEW_FIELDS
        }
        reasons = payload.get("rejection_reasons", [])
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) for reason in reasons
        ):
            raise ReviewValidationError("rejection_reasons 必须是字符串数组")
        if len(reasons) != len(set(reasons)):
            raise ReviewValidationError("拒绝原因不能重复")
        known_reasons = {reason for reason, _label in REJECTION_REASONS}
        for reason in reasons:
            if reason in known_reasons:
                continue
            if reason.startswith("other:") and 0 < len(reason[6:].strip()) <= 500:
                continue
            raise ReviewValidationError("未知或无效的拒绝原因：" + reason)
        if decision == "reject" and not reasons:
            raise ReviewValidationError("拒绝时至少选择一个拒绝原因")
        if decision == "keep" and reasons:
            raise ReviewValidationError("保留项不能同时填写拒绝原因")

        proposal_row = self._proposals[proposal_key]
        if (proposal_row["proposal"].get("review_stage") == "offline_prefilter_before_nre"
            and reconstruction == "verified"):
            raise ReviewValidationError("本批只有离线预筛证据，重建支持须保持未验证，不能代替实际 NRE 检查")
        pair_id = payload.get("arm_pair_id")
        if pair_id is not None and not isinstance(pair_id, str):
            raise ReviewValidationError("arm_pair_id 必须是字符串或 null")
        pair = proposal_row["arm_pairs"].get(pair_id) if pair_id else None
        if pair_id and pair is None:
            raise ReviewValidationError("所选双臂组合不属于该候选")

        if decision == "keep":
            if pair is None:
                raise ReviewValidationError("保留时必须选择并审核明确的两个选择")
            proposal = proposal_row["proposal"]
            if (
                proposal.get("proposal_type") == "route_branch"
                and not pair.get("recorded_arm_identified")
            ):
                raise ReviewValidationError(
                    "该路口候选尚未识别记录臂，不能判为保留；请先标待定并补证据"
                )
            required_true = TRISTATE_REVIEW_FIELDS[:4]
            missing = [field for field in required_true if values[field] is not True]
            if missing:
                raise ReviewValidationError(
                    "保留项的以下字段必须明确通过：" + ", ".join(missing)
                )
            if values["semantic_referents_verified"] is False:
                raise ReviewValidationError("语义指代不通过时不能判为保留")
            if reconstruction == "failed":
                raise ReviewValidationError("重建支持失败时不能判为保留")

        return {
            "reviewer": reviewer,
            "decision": decision,
            "arm_pair_id": pair_id,
            "reviewed_arm_pair": copy.deepcopy(pair),
            **values,
            "reconstruction_support_status": reconstruction,
            "rejection_reasons": reasons,
            "notes": notes,
        }

    def _summary(self, log: Dict[str, Any]) -> Dict[str, int]:
        decisions = log["decisions"]
        counts = {"keep": 0, "pending": 0, "reject": 0}
        for key in self._proposals:
            record = decisions.get(key)
            if isinstance(record, dict) and record.get("decision") in counts:
                counts[record["decision"]] += 1
        reviewed = sum(counts.values())
        return {
            "total": len(self._proposals),
            **counts,
            "unreviewed": len(self._proposals) - reviewed,
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._refresh_inventory()
            log = self._load_log()
            active = {
                key: value
                for key, value in log["decisions"].items()
                if key in self._proposals
            }
            return {
                "schema_version": SCHEMA_VERSION,
                "decisions": active,
                "summary": self._summary(log),
                "updated_at": log.get("updated_at"),
            }

    def submit(self, proposal_key: str, payload: Any) -> Dict[str, Any]:
        with self._lock:
            self._refresh_inventory()
            validated = self.validate_review(proposal_key, payload)
            log = self._load_log()
            previous = log["decisions"].get(proposal_key, {})
            revision = int(previous.get("revision", 0)) + 1
            timestamp = _now()
            proposal_row = self._proposals[proposal_key]
            proposal = proposal_row["proposal"]
            record = {
                **validated,
                "proposal_key": proposal_key,
                "artifact_uuid": proposal_row["artifact_uuid"],
                "scene_id_private": proposal_row["scene_id_private"],
                "proposal_type": proposal.get("proposal_type"),
                "candidate_families_not_qualified": list(
                    proposal.get("candidate_families_not_qualified", [])
                ),
                "evidence_sheet": proposal.get("evidence_sheet"),
                "map_evidence_svg": proposal.get("map_evidence_svg"),
                "reviewed_at": timestamp,
                "revision": revision,
            }
            log["decisions"][proposal_key] = record
            log["audit_log"].append(copy.deepcopy(record))
            log["updated_at"] = timestamp
            atomic_json(self.decisions_path, log)
            return {"record": record, "summary": self._summary(log)}


def additional_review_inputs(root: Path) -> Dict[str, Dict[str, Path]]:
    """Read a local registry; namespace logs never alias the original logs."""
    root = Path(root).resolve()
    registry = root / "config/additional_curation_reviews.private.json"
    if not registry.exists():
        return {}
    payload = load_json(registry)
    if not isinstance(payload, dict):
        raise ValueError("additional review registry must be an object")
    inputs = {}
    used_logs = {(root / "data/private/curation_decisions.private.json").resolve(),
                 (root / "data/private/feasibility_decisions.private.json").resolve()}
    for name, entry in payload.items():
        if (not re.fullmatch(r"[a-z0-9_-]{1,64}", name) or not isinstance(entry, dict)
            or not {"inventory", "decisions"} <= set(entry)
            or set(entry) - {"inventory", "decisions", "acquisition"}):
            raise ValueError("invalid additional review registry entry")
        inventory = (root / entry["inventory"]).resolve()
        decisions = (root / entry["decisions"]).resolve()
        if (not inventory.is_relative_to(root) or not decisions.is_relative_to(root / "data/private")
            or decisions in used_logs or inventory == decisions):
            raise ValueError("additional review paths must be private, isolated and project-local")
        used_logs.add(decisions)
        inputs[name] = {"inventory_path": inventory, "decisions_path": decisions}
        if "acquisition" in entry:
            acquisition = (root / entry["acquisition"]).resolve()
            if not acquisition.is_relative_to(root) or acquisition in {inventory, decisions}:
                raise ValueError("additional acquisition report must be isolated and project-local")
            inputs[name]["acquisition_path"] = acquisition
    return inputs


def load_additional_review_stores(root: Path) -> Dict[str, ReviewStore]:
    return {name: ReviewStore(paths["inventory_path"], paths["decisions_path"])
            for name, paths in additional_review_inputs(root).items()}


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: Any,
        handler: Any,
        store: ReviewStore,
        static_root: Path,
        additional_stores: Optional[Dict[str, ReviewStore]] = None,
        agent_review_only: bool = False,
    ) -> None:
        super().__init__(address, handler)
        self.review_store = store
        self.static_root = Path(static_root).resolve()
        self.agent_review_only = agent_review_only
        self.additional_review_stores = dict(additional_stores or {})
        self.feasibility_store = FeasibilityReviewStore(
            store, self.static_root / "feasibility",
            store.decisions_path.with_name("feasibility_decisions.private.json"))
        self.agent_feasibility_store = AgentFeasibilityReviewStore(
            None, self.static_root / "agent-feasibility",
            store.decisions_path.with_name("agent_feasibility_decisions.private.json"))
        self.additional_feasibility_stores = {
            name: FeasibilityReviewStore(
                initial, self.static_root / "additional-feasibility" / name,
                initial.decisions_path.parent / "additional_feasibility" / name / "decisions.private.json")
            for name, initial in self.additional_review_stores.items()
        }


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    server: ReviewHTTPServer

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        directory = str(args[2].static_root) if len(args) >= 3 else None
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; object-src 'self'; connect-src 'self'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        super().end_headers()

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _same_origin_or_no_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        return urlsplit(origin).netloc == self.headers.get("Host", "")

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        try:
            if path.startswith("/api/agent-feasibility-reviews/"):
                identifiers = unquote(path[len("/api/agent-feasibility-reviews/"):]).split("/")
                if len(identifiers) != 2:
                    raise ValueError("需要候选与证据包编号")
                self._json(200, self.server.agent_feasibility_store.snapshot(*identifiers))
                return
            if path.startswith("/agent-feasibility/") and path.endswith("/index.private.html"):
                identifiers = unquote(path).strip("/").split("/")
                if len(identifiers) != 4:
                    raise ValueError("无效自主编制审核页面路径")
                # Validate before reading/injecting. Historical packet files are
                # kept byte-for-byte; existing pages gain the form at serve time.
                self.server.agent_feasibility_store.snapshot(*identifiers[1:3])
                page = (self.server.static_root / unquote(path).lstrip("/")).resolve()
                if not page.is_relative_to(self.server.static_root / "agent-feasibility"):
                    raise ValueError("审核页面路径越界")
                content = page.read_text(encoding="utf-8")
                if '/feasibility-review.private.js' not in content:
                    content = content.replace('</html>', '<script src="/feasibility-review.private.js" defer></script></html>')
                encoded = content.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if path.startswith("/api/additional-feasibility-reviews/"):
                identifiers = unquote(path[len("/api/additional-feasibility-reviews/"):]).split("/")
                if len(identifiers) != 3 or identifiers[0] not in self.server.additional_feasibility_stores:
                    raise ValueError("unknown additional feasibility batch or invalid packet path")
                self._json(200, self.server.additional_feasibility_stores[identifiers[0]].snapshot(*identifiers[1:]))
                return
            if path.startswith("/api/additional-reviews/"):
                name = unquote(path[len("/api/additional-reviews/"):])
                store = self.server.additional_review_stores.get(name)
                if store is None:
                    self._json(404, {"error": "unknown additional review batch"})
                else:
                    self._json(200, store.snapshot())
                return
            if path == "/feasibility-review.private.js":
                encoded = REVIEW_SCRIPT.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if path.startswith("/api/feasibility-reviews/"):
                identifiers = unquote(path[len("/api/feasibility-reviews/"):]).split("/")
                if len(identifiers) != 2:
                    raise ValueError("需要候选与证据包编号")
                self._json(200, self.server.feasibility_store.snapshot(*identifiers))
                return
            if path == "/api/reviews":
                self._json(200, self.server.review_store.snapshot())
                return
            if path == "/api/health":
                self._json(
                    200,
                    {
                        "ok": True,
                        "loopback_only": True,
                        "feasibility_review_available": True,
                        "agent_feasibility_review_available": True,
                        "agent_review_only": self.server.agent_review_only,
                        "additional_review_batches": sorted(self.server.additional_review_stores),
                        "decision_log": str(
                            self.server.review_store.decisions_path
                        ),
                    },
                )
                return
            if path == "/":
                self.send_response(302)
                self.send_header("Location", "/index.private.html")
                self.end_headers()
                return
            super().do_GET()
        except (OSError, ValueError, ReviewValidationError) as error:
            self._json(500, {"error": str(error)})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        if self.server.agent_review_only and not path.startswith("/api/agent-feasibility-reviews/"):
            self._json(405, {"error": "本端口只保存自主编制双臂视频审核；旧审核请使用原服务"})
            return
        supplementary = path.startswith("/api/feasibility-reviews/")
        agent_feasibility = path.startswith("/api/agent-feasibility-reviews/")
        additional_feasibility = path.startswith("/api/additional-feasibility-reviews/")
        additional = path.startswith("/api/additional-reviews/")
        prefix = ("/api/agent-feasibility-reviews/" if agent_feasibility else
                  "/api/additional-feasibility-reviews/" if additional_feasibility else
                  "/api/additional-reviews/" if additional else
                  "/api/feasibility-reviews/" if supplementary else "/api/reviews/")
        if not path.startswith(prefix) or len(path) <= len(prefix):
            self._json(404, {"error": "not found"})
            return
        if not self._same_origin_or_no_origin():
            self._json(403, {"error": "origin 不匹配"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Content-Type 必须是 application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(400, {"error": "无效 Content-Length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(413, {"error": "提交内容为空或超过 32 KiB"})
            return
        try:
            payload = json.loads(
                self.rfile.read(length).decode("utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError("non-standard JSON constant: " + value)
                ),
            )
            proposal_key = unquote(path[len(prefix) :])
            if agent_feasibility:
                identifiers = proposal_key.split("/")
                if len(identifiers) != 2:
                    raise ValueError("需要候选与证据包编号")
                result = self.server.agent_feasibility_store.submit(*identifiers, payload)
            elif additional_feasibility:
                identifiers = proposal_key.split("/")
                if len(identifiers) != 3 or identifiers[0] not in self.server.additional_feasibility_stores:
                    raise ValueError("unknown additional feasibility batch or invalid packet path")
                result = self.server.additional_feasibility_stores[identifiers[0]].submit(*identifiers[1:], payload)
            elif additional:
                identifiers = proposal_key.split("/")
                if len(identifiers) != 2 or identifiers[0] not in self.server.additional_review_stores:
                    raise ValueError("unknown additional review batch or invalid proposal path")
                result = self.server.additional_review_stores[identifiers[0]].submit(identifiers[1], payload)
            elif supplementary:
                identifiers = proposal_key.split("/")
                if len(identifiers) != 2:
                    raise ValueError("需要候选与证据包编号")
                result = self.server.feasibility_store.submit(*identifiers, payload)
            else:
                result = self.server.review_store.submit(proposal_key, payload)
            self._json(200, {"ok": True, **result})
        except (UnicodeDecodeError, json.JSONDecodeError, ReviewValidationError, ValueError) as error:
            self._json(400, {"error": str(error)})
        except OSError as error:
            self._json(500, {"error": "写入审核日志失败：%s" % error})

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler API
        self._json(405, {"error": "cross-origin requests are not allowed"})

    def list_directory(self, path: str) -> None:
        self.send_error(404, "directory listing disabled")
        return None


def serve_curation_review(
    project_root: Path,
    bind: str = "127.0.0.1",
    port: int = 8765,
    inventory_path: Optional[Path] = None,
    decisions_path: Optional[Path] = None,
    static_root: Optional[Path] = None,
    agent_review_only: bool = False,
) -> None:
    """Serve the review UI and API on loopback for SSH forwarding."""

    if bind != "localhost":
        try:
            if not ipaddress.ip_address(bind).is_loopback:
                raise ValueError("审核服务只允许绑定回环地址；请通过 SSH 隧道访问")
        except ValueError as error:
            if "只允许" in str(error):
                raise
            raise ValueError("bind 必须是回环 IP 或 localhost") from error
    if not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port 必须在 0..65535 范围内")
    root = Path(project_root).resolve()
    inventory = inventory_path or root / "generated" / "curation_review_inventory.private.json"
    decisions = decisions_path or root / "data" / "private" / "curation_decisions.private.json"
    review_root = static_root or root / ".runtime" / "curation_review"
    if not Path(review_root).joinpath("index.private.html").is_file():
        raise ValueError("审核页面不存在；请先运行 prepare-curation")
    Path(decisions).parent.mkdir(parents=True, exist_ok=True)
    if not os.access(Path(decisions).parent, os.W_OK):
        raise ValueError("审核日志目录不可写：%s" % Path(decisions).parent)
    store = ReviewStore(Path(inventory), Path(decisions))
    server = ReviewHTTPServer(
        (bind, port), ReviewRequestHandler, store, Path(review_root),
        additional_stores=load_additional_review_stores(root),
        agent_review_only=agent_review_only,
    )
    print(
        "Language2Drive review service: http://%s:%d/index.private.html\n"
        "Writing decisions to: %s"
        % (bind, server.server_port, Path(decisions).resolve()),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()

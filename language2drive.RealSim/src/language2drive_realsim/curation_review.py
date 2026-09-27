"""Build private, timestamp-aligned curator review packets.

The packets join route-reachable topology hypotheses with real preview frames.
They are review aids only: every scientific decision remains explicitly unset,
and no packet can be consumed as a qualified scene annotation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from html import escape
from pathlib import Path
from typing import Any, Dict, List

from .io import load_json, stable_hash
from .preview_mining import _video_path


MODEL_INPUT_RESOLUTION = [320, 576]
MODEL_INPUT_DIMENSIONS = {"width": 576, "height": 320}
MINIMUM_PRE_DECISION_CONTEXT_S = 3.0
MINIMUM_POST_DECISION_CONTEXT_S = 2.0
TRISTATE_REVIEW_FIELDS = (
    "camera_observable_at_576x320",
    "both_arms_topologically_legal",
    "both_arms_topologically_reachable",
    "balanced_opportunity",
    "semantic_referents_verified",
)
REJECTION_REASONS = (
    ("camera_target_not_visible", "关键道路结构或语义目标不可见"),
    ("text_unreadable_at_576x320", "任务依赖的文字在 576×320 下不可读"),
    ("pre_decision_choice_already_committed", "窗口开始时选择已经不可逆"),
    ("insufficient_post_decision_commitment", "后部帧不足以观察记录臂承诺"),
    ("frame_or_map_misaligned", "帧时间与地图机会不一致"),
    ("arms_not_distinct", "两条臂没有形成不同驾驶语义"),
    ("arms_not_both_legal", "至少一条臂不合法"),
    ("arms_not_both_reachable", "至少一条臂在拓扑上不连通"),
    ("route_or_command_leakage_risk", "非语言输入可能泄露目标臂"),
    ("unbalanced_arm_difficulty", "两臂难度、安全或时间余量严重失衡"),
    ("semantic_referent_ambiguous", "语义指代对象不唯一或不可辨认"),
    ("actor_identity_unstable", "关键交通参与者跨帧身份不稳定"),
    ("reconstruction_support_failed", "NRE 验证明某一臂无重建支持"),
    ("duplicate_or_near_duplicate", "与已保留机会重复且无新增覆盖"),
)


def arm_pair_candidates(proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return stable, explicitly paired arm choices for human review.

    These are still topology-only suggestions.  Naming two arms prevents an
    opportunity-level review from being mistaken for review of an unspecified
    counterfactual choice.
    """

    topology = proposal.get("topology_private", {})
    current_lane = str(proposal.get("lane_id_private", ""))
    rows: List[Dict[str, Any]] = []

    def add(kind: str, first_role: str, first_lane: str, second_role: str, second_lane: str) -> None:
        arms = [
            {"arm": "R0", "role": first_role, "lane_id": str(first_lane)},
            {"arm": "R1", "role": second_role, "lane_id": str(second_lane)},
        ]
        rows.append(
            {
                "arm_pair_id": "arm-pair-" + stable_hash(
                    {
                        "proposal_key": proposal.get("proposal_key"),
                        "kind": kind,
                        "arms": arms,
                    }
                )[:16],
                "kind": kind,
                "arms": arms,
                "recorded_arm_identified": any(
                    arm["role"] == "recorded" for arm in arms
                ),
            }
        )

    if proposal.get("proposal_type") == "route_branch":
        legal = [str(value) for value in topology.get("legal_next_lanes", [])]
        observed = topology.get("observed_recorded_next_lane")
        observed = str(observed) if observed is not None else None
        if observed in legal:
            for alternative in legal:
                if alternative != observed:
                    add(
                        "recorded_vs_counterfactual_branch",
                        "recorded",
                        observed,
                        "counterfactual_branch",
                        alternative,
                    )
        else:
            for index, first in enumerate(legal):
                for second in legal[index + 1 :]:
                    add("unobserved_branch_pair", "branch", first, "branch", second)
    else:
        for lane in topology.get("adjacent_lanes_left", []):
            add("keep_vs_left", "keep", current_lane, "change_left", str(lane))
        for lane in topology.get("adjacent_lanes_right", []):
            add("keep_vs_right", "keep", current_lane, "change_right", str(lane))
        for left in topology.get("adjacent_lanes_left", []):
            for right in topology.get("adjacent_lanes_right", []):
                add("left_vs_right", "change_left", str(left), "change_right", str(right))
    if "review_arm_pair_ids" in proposal:
        allowed = proposal["review_arm_pair_ids"]
        ids = {row["arm_pair_id"] for row in rows}
        if (not isinstance(allowed, list) or not allowed
            or any(not isinstance(value, str) for value in allowed)
            or len(allowed) != len(set(allowed)) or not set(allowed) <= ids):
            raise ValueError("invalid explicitly selected review arm pairs")
        rows = [row for row in rows if row["arm_pair_id"] in allowed]
    return rows


def _arm_pair_label(pair: Dict[str, Any]) -> str:
    role_labels = {
        "recorded": "记录轨迹",
        "counterfactual_branch": "反事实分支",
        "branch": "分支",
        "keep": "保持当前车道",
        "change_left": "向左换道",
        "change_right": "向右换道",
    }
    rendered = []
    for arm in pair["arms"]:
        rendered.append(
            "%s %s（lane %s）"
            % (
                arm["arm"],
                role_labels.get(arm["role"], arm["role"]),
                arm["lane_id"],
            )
        )
    suffix = "；记录臂尚未识别" if pair["kind"] == "unobserved_branch_pair" else ""
    return " ↔ ".join(rendered) + suffix


def decision_sample_times(decision_s: float, duration_s: float) -> List[float]:
    """Return five bounded frame times centered on a decision opportunity."""

    end = max(0.0, float(duration_s) - 0.05)
    center = min(max(0.0, float(decision_s)), end)
    start = max(0.0, center - 3.0)
    stop = min(end, center + 3.0)
    if stop <= start:
        return [round(start, 3)] * 5
    step = (stop - start) / 4.0
    return [round(start + index * step, 3) for index in range(5)]


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _render_sheet(video: Path, sample_times_s: List[float], output: Path) -> None:
    if not sample_times_s:
        raise ValueError("at_least_one_preview_seek_required")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise OSError("ffmpeg_not_available")
    command = [ffmpeg, "-loglevel", "error", "-y"]
    for sample_time in sample_times_s:
        command.extend(("-ss", "%.3f" % sample_time, "-i", str(video)))
    filters = []
    labels = []
    for index in range(len(sample_times_s)):
        label = "v%d" % index
        filters.append(
            "[%d:v]scale=576:320:force_original_aspect_ratio=decrease,"
            "pad=576:320:(ow-iw)/2:(oh-ih)/2[%s]" % (index, label)
        )
        labels.append("[%s]" % label)
    if len(labels) == 1:
        filters.append("%snull[out]" % labels[0])
    else:
        filters.append("%shstack=inputs=%d[out]" % ("".join(labels), len(labels)))
    output.parent.mkdir(parents=True, exist_ok=True)
    command.extend(
        (
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-frames:v",
            "1",
            str(output),
        )
    )
    subprocess.run(command, check=True, capture_output=True, text=True)


def _render_map_svg(
    hypothesis: Dict[str, Any], trajectory: List[List[float]], output: Path
) -> None:
    geometry = hypothesis.get("lane_geometry_private_not_qualified", {})
    polylines = {
        lane_id: row.get("centerline_xyz", [])
        for lane_id, row in geometry.items()
        if row.get("centerline_xyz")
    }
    if not polylines:
        raise ValueError("map_geometry_unavailable")
    decision = [float(value) for value in hypothesis["position_xyz"][:2]]
    local_trajectory = [
        point
        for point in trajectory
        if (float(point[0]) - decision[0]) ** 2
        + (float(point[1]) - decision[1]) ** 2
        <= 60.0**2
    ]
    all_points = [point for points in polylines.values() for point in points]
    all_points.extend(local_trajectory)
    all_points.append(decision)
    min_x = min(float(point[0]) for point in all_points)
    max_x = max(float(point[0]) for point in all_points)
    min_y = min(float(point[1]) for point in all_points)
    max_y = max(float(point[1]) for point in all_points)
    span_x, span_y = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)
    scale = min(740.0 / span_x, 740.0 / span_y)
    x_margin = (800.0 - span_x * scale) / 2.0
    y_margin = (800.0 - span_y * scale) / 2.0

    def xy(point: List[float]) -> str:
        x = x_margin + (float(point[0]) - min_x) * scale
        y = 800.0 - (y_margin + (float(point[1]) - min_y) * scale)
        return "%.2f,%.2f" % (x, y)

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800" viewBox="0 0 800 800">',
        '<rect width="800" height="800" fill="#f8fafc"/>',
        "<title>%s</title>" % escape(str(hypothesis.get("lane_id", "lane review"))),
        '<text x="16" y="24" font-family="sans-serif" font-size="14" fill="#0f172a">'
        "dashed: recorded ego · magenta: decision · amber: gates</text>",
    ]
    if local_trajectory:
        elements.append(
            '<polyline points="%s" fill="none" stroke="#0f172a" '
            'stroke-width="3" stroke-dasharray="8 6" opacity="0.65"/>'
            % " ".join(xy(point) for point in local_trajectory)
        )
    palette = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#0891b2")
    current_lane = str(hypothesis["lane_id"])
    for index, (lane_id, points) in enumerate(sorted(polylines.items())):
        color = "#475569" if lane_id == current_lane else palette[index % len(palette)]
        elements.append(
            '<polyline points="%s" fill="none" stroke="%s" stroke-width="6" opacity="0.8"/>'
            % (" ".join(xy(point) for point in points), color)
        )
        label_xy = xy(points[len(points) // 2]).split(",")
        elements.append(
            '<text x="%s" y="%s" font-family="sans-serif" font-size="16" '
            'font-weight="bold" fill="%s" stroke="white" stroke-width="3" '
            'paint-order="stroke">lane %s</text>'
            % (label_xy[0], label_xy[1], color, escape(str(lane_id)))
        )
        gate = geometry[lane_id].get("commitment_gate_xy_not_qualified")
        if gate and len(gate) == 2:
            elements.append(
                '<line x1="%s" y1="%s" x2="%s" y2="%s" '
                'stroke="#f59e0b" stroke-width="4"/>'
                % tuple((xy(gate[0]) + "," + xy(gate[1])).replace(",", " ").split())
            )
    decision_xy = xy(decision).split(",")
    elements.append(
        '<circle cx="%s" cy="%s" r="8" fill="#db2777" stroke="white" stroke-width="2"/>'
        % (decision_xy[0], decision_xy[1])
    )
    elements.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _curator_decisions() -> Dict[str, Any]:
    return {
        "camera_observable_at_model_resolution": None,
        "both_arms_legal": None,
        "both_arms_reachable": None,
        "both_arms_reconstruction_supported": None,
        "balanced_opportunity": None,
        "semantic_referents_verified": None,
        "reviewer": None,
        "reviewed_at": None,
        "notes": "",
    }


REVIEW_JAVASCRIPT = r"""
"use strict";

const triStateFields = [
  "camera_observable_at_576x320",
  "both_arms_topologically_legal",
  "both_arms_topologically_reachable",
  "balanced_opportunity",
  "semantic_referents_verified",
];
const reviewerInput = document.getElementById("reviewer");
const filterInput = document.getElementById("filter-state");
const searchInput = document.getElementById("search");
const summaryNode = document.getElementById("review-summary");

function parseTriState(value) {
  if (value === "true") return true;
  if (value === "false") return false;
  return null;
}

function showSummary(summary) {
  const reviewed = summary.keep + summary.pending + summary.reject;
  summaryNode.textContent =
    `已提交 ${reviewed}/${summary.total} · 保留 ${summary.keep} · ` +
    `待定 ${summary.pending} · 拒绝 ${summary.reject} · 未审核 ${summary.unreviewed}`;
}

function setFormRecord(form, record) {
  form.elements.decision.value = record.decision || "";
  form.elements.arm_pair_id.value = record.arm_pair_id || "";
  for (const field of triStateFields) {
    const value = record[field];
    form.elements[field].value = value === true ? "true" : value === false ? "false" : "";
  }
  form.elements.reconstruction_support_status.value =
    record.reconstruction_support_status || "unverified";
  form.elements.notes.value = record.notes || "";
  const reasons = new Set(record.rejection_reasons || []);
  for (const checkbox of form.querySelectorAll('input[name="rejection_reason"]')) {
    checkbox.checked = reasons.has(checkbox.value);
  }
  const other = [...reasons].find((value) => value.startsWith("other:"));
  form.elements.other_reason.value = other ? other.slice(6) : "";
  const card = form.closest("article");
  card.dataset.savedDecision = record.decision;
  card.classList.remove("dirty", "state-keep", "state-pending", "state-reject");
  card.classList.add(`state-${record.decision}`);
  form.querySelector(".save-status").textContent =
    `已保存 · 第 ${record.revision} 版 · ${record.reviewed_at}`;
  if (!reviewerInput.value && record.reviewer) reviewerInput.value = record.reviewer;
  applyFilters();
}

function collectPayload(form) {
  const reasons = [...form.querySelectorAll('input[name="rejection_reason"]:checked')]
    .map((input) => input.value);
  const other = form.elements.other_reason.value.trim();
  if (other) reasons.push(`other:${other}`);
  const payload = {
    reviewer: reviewerInput.value.trim(),
    decision: form.elements.decision.value,
    arm_pair_id: form.elements.arm_pair_id.value || null,
    reconstruction_support_status: form.elements.reconstruction_support_status.value,
    rejection_reasons: reasons,
    notes: form.elements.notes.value.trim(),
  };
  for (const field of triStateFields) {
    payload[field] = parseTriState(form.elements[field].value);
  }
  return payload;
}

async function submitReview(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const card = form.closest("article");
  const button = form.querySelector('button[type="submit"]');
  const status = form.querySelector(".save-status");
  const key = card.dataset.proposalKey;
  button.disabled = true;
  status.textContent = "正在提交……";
  status.classList.remove("error");
  try {
    const response = await fetch(`/api/reviews/${encodeURIComponent(key)}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(collectPayload(form)),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    localStorage.setItem("l2d-reviewer", reviewerInput.value.trim());
    setFormRecord(form, result.record);
    showSummary(result.summary);
  } catch (error) {
    status.textContent = `提交失败：${error.message}`;
    status.classList.add("error");
  } finally {
    button.disabled = false;
  }
}

function applyFilters() {
  const wanted = filterInput.value;
  const query = searchInput.value.trim().toLowerCase();
  for (const card of document.querySelectorAll("article.proposal")) {
    const decision = card.dataset.savedDecision || "unreviewed";
    const stateMatches = wanted === "all" || wanted === decision;
    const textMatches = !query || card.textContent.toLowerCase().includes(query);
    card.hidden = !(stateMatches && textMatches);
  }
}

function goToNextUnreviewed() {
  const card = [...document.querySelectorAll("article.proposal")]
    .find((node) => !node.dataset.savedDecision);
  if (card) {
    filterInput.value = "all";
    searchInput.value = "";
    applyFilters();
    card.scrollIntoView({behavior: "smooth", block: "start"});
  }
}

async function loadReviews() {
  reviewerInput.value = localStorage.getItem("l2d-reviewer") || "";
  try {
    const response = await fetch("/api/reviews", {cache: "no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    for (const [key, record] of Object.entries(result.decisions || {})) {
      const card = document.querySelector(`[data-proposal-key="${CSS.escape(key)}"]`);
      if (card) setFormRecord(card.querySelector("form"), record);
    }
    showSummary(result.summary);
  } catch (error) {
    summaryNode.textContent = `无法读取已保存审核：${error.message}`;
    summaryNode.classList.add("error");
  }
}

for (const form of document.querySelectorAll("form.review-form")) {
  form.addEventListener("submit", submitReview);
  form.addEventListener("change", () => {
    form.closest("article").classList.add("dirty");
    form.querySelector(".save-status").textContent = "有尚未提交的修改";
  });
  form.addEventListener("input", () => {
    form.closest("article").classList.add("dirty");
    form.querySelector(".save-status").textContent = "有尚未提交的修改";
  });
}
filterInput.addEventListener("change", applyFilters);
searchInput.addEventListener("input", applyFilters);
document.getElementById("next-unreviewed").addEventListener("click", goToNextUnreviewed);
window.addEventListener("beforeunload", (event) => {
  if (document.querySelector("article.dirty")) {
    event.preventDefault();
    event.returnValue = "";
  }
});
loadReviews();
"""


def _render_review_index(
    scenes: List[Dict[str, Any]], output: Path, project_root: Path,
    api_base: str = "/api/reviews",
) -> int:
    """Write the private HTML client for the loopback review service."""

    cards = []
    survivor_count = 0
    tri_state_labels = (
        ("camera_observable_at_576x320", "576×320 下可观察"),
        ("both_arms_topologically_legal", "两个选择在拓扑上都合法"),
        ("both_arms_topologically_reachable", "两个选择在拓扑上都可达"),
        ("balanced_opportunity", "两个选择的难度与时间余量平衡"),
        ("semantic_referents_verified", "语义指代已核实（不适用可留空）"),
    )
    for scene in scenes:
        for proposal in scene["proposals"]:
            if not proposal["timing_review"]["automatic_timing_prefilter_pass"]:
                continue
            survivor_count += 1

            def href(value: str) -> str:
                path = Path(value)
                absolute = path if path.is_absolute() else Path(project_root) / path
                return os.path.relpath(absolute, output.parent).replace(os.sep, "/")

            families = ", ".join(proposal["candidate_families_not_qualified"])
            timing = proposal["timing_review"]
            topology = proposal["topology_private"]
            if proposal["proposal_type"] == "route_branch":
                topology_summary = "recorded next lane: %s · legal next lanes: %s" % (
                    topology.get("observed_recorded_next_lane", "unknown"),
                    ", ".join(topology.get("legal_next_lanes", [])) or "none",
                )
            else:
                topology_summary = "left adjacent: %s · right adjacent: %s" % (
                    ", ".join(topology.get("adjacent_lanes_left", [])) or "none",
                    ", ".join(topology.get("adjacent_lanes_right", [])) or "none",
                )
            pair_options = ['<option value="">请选择本 PCU 的两个选择</option>']
            for pair in arm_pair_candidates(proposal):
                pair_options.append(
                    '<option value="%s">%s</option>'
                    % (
                        escape(pair["arm_pair_id"], quote=True),
                        escape(_arm_pair_label(pair)),
                    )
                )
            tri_state_controls = []
            for field, label in tri_state_labels:
                tri_state_controls.append(
                    """<label>{label}
  <select name="{field}">
    <option value="">未判断 / 不适用</option>
    <option value="true">通过</option>
    <option value="false">不通过</option>
  </select>
</label>""".format(label=escape(label), field=escape(field, quote=True))
                )
            reason_controls = []
            for reason, label in REJECTION_REASONS:
                reason_controls.append(
                    '<label><input type="checkbox" name="rejection_reason" value="%s"> %s</label>'
                    % (escape(reason, quote=True), escape(label))
                )
            cards.append(
                """
<article class="proposal" data-proposal-key="{proposal_key}">
  <h2>{proposal_id} · {proposal_type}</h2>
  <p><code>{proposal_key}</code></p>
  <p>scene: <code>{scene_id}</code> · artifact: <code>{artifact}</code></p>
  <p>lane: <code>{lane}</code> · decision: {decision:.3f}s · context: {pre:.2f}s / {post:.2f}s</p>
  <p>unqualified family suggestions: {families}</p>
  <p>unqualified topology choice: {topology_summary}</p>
  <div class="evidence">
    <a href="{sheet}"><img src="{sheet}" alt="five time-aligned model-resolution frames"></a>
    <a class="map-link" href="{map_svg}"><img class="map" src="{map_svg}" alt="authoritative map review"></a>
  </div>
  <form class="review-form">
    <fieldset>
      <legend>审核并直接保存到私有工作日志</legend>
      <label class="wide"><strong>本 PCU 的两个选择（保留时必选并审核两者）</strong>
        <select name="arm_pair_id">{pair_options}</select>
      </label>
      {tri_state_controls}
      <label>重建支持状态
        <select name="reconstruction_support_status">
          <option value="unverified">尚未验证（本轮默认）</option>
          <option value="verified">已由完整 NRE 证据确认</option>
          <option value="failed">NRE 验证失败</option>
        </select>
      </label>
      <label><strong>审核结论</strong>
        <select name="decision" required>
          <option value="">请选择</option>
          <option value="keep">保留，进入 NRE/oracle 验证</option>
          <option value="pending">待定，需要补充证据</option>
          <option value="reject">拒绝</option>
        </select>
      </label>
    </fieldset>
    <details>
      <summary>拒绝原因（判为 reject 时至少选择一项）</summary>
      <div class="reason-grid">{reason_controls}</div>
      <label>其他原因 <input type="text" name="other_reason" maxlength="500" placeholder="简短说明"></label>
    </details>
    <label class="notes">备注
      <textarea name="notes" maxlength="4000" rows="3" placeholder="待定时必须写明缺少什么证据；也可记录另一选择的风险。"></textarea>
    </label>
    <div class="actions">
      <button type="submit">提交 / 更新本项审核</button>
      <span class="save-status">尚未提交</span>
    </div>
  </form>
</article>
""".format(
                    proposal_id=escape(proposal["proposal_id"]),
                    proposal_type=escape(proposal["proposal_type"]),
                    proposal_key=escape(proposal["proposal_key"]),
                    scene_id=escape(scene["scene_id_private"]),
                    artifact=escape(scene["artifact_uuid"]),
                    lane=escape(proposal["lane_id_private"]),
                    decision=proposal["decision_offset_us"] / 1_000_000.0,
                    pre=timing["pre_decision_context_s"],
                    post=timing["post_decision_context_s"],
                    families=escape(families or "none"),
                    topology_summary=escape(topology_summary),
                    sheet=escape(href(proposal["evidence_sheet"]), quote=True),
                    map_svg=escape(
                        href(proposal["map_evidence_svg"]), quote=True
                    ),
                    pair_options="".join(pair_options),
                    tri_state_controls="".join(tri_state_controls),
                    reason_controls="".join(reason_controls),
                )
            )
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Language2Drive 私有时间窗审核</title>
<style>
body{font:15px system-ui,sans-serif;margin:24px;background:#f1f5f9;color:#0f172a}
.warning{padding:14px;background:#fef3c7;border:1px solid #f59e0b}
.toolbar{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:10px;align-items:end;padding:12px;background:#e2e8f0;border:1px solid #94a3b8}
.toolbar label{display:grid;gap:4px}.toolbar input,.toolbar select{min-width:180px}
.proposal{margin:22px 0;padding:18px;background:white;border:2px solid #cbd5e1;border-radius:10px;scroll-margin-top:105px}
.proposal.state-keep{border-color:#16a34a}.proposal.state-pending{border-color:#d97706}.proposal.state-reject{border-color:#dc2626}.proposal.dirty{box-shadow:0 0 0 4px #fde68a}
.evidence{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,1fr);gap:12px;align-items:start}
img{width:100%;height:auto}.map-link{display:block}.map{display:block;min-height:420px;border:1px solid #e2e8f0;background:#f8fafc;object-fit:contain}
fieldset{display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:10px;margin-top:14px}
fieldset label{display:grid;gap:4px}.wide{grid-column:1/-1}select,input,textarea,button{font:inherit;padding:7px}textarea{width:100%;box-sizing:border-box}
details{margin:12px 0;padding:10px;background:#f8fafc}.reason-grid{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:6px;margin:10px 0}.reason-grid label{display:block}
.notes{display:grid;gap:4px}.actions{display:flex;gap:12px;align-items:center;margin-top:10px}button{cursor:pointer}.save-status{color:#475569}.error{color:#b91c1c;font-weight:bold}
code{overflow-wrap:anywhere}@media(max-width:900px){.evidence,fieldset,.reason-grid{grid-template-columns:1fr}.wide{grid-column:auto}.toolbar{position:static}}
</style></head><body>
<h1>Language2Drive 私有时间窗候选审核</h1>
<p class="warning"><strong>Fail closed：</strong>保留只是进入后续 NRE/oracle 验证，不会自动写入正式 scene annotations，也不会自动成为合格 PCU。每个 PCU 都必须明确并审核两个语言选择。</p>
<div class="toolbar">
  <label>审核人<input id="reviewer" maxlength="100" placeholder="必填"></label>
  <label>状态筛选<select id="filter-state"><option value="all">全部</option><option value="unreviewed">未审核</option><option value="keep">保留</option><option value="pending">待定</option><option value="reject">拒绝</option></select></label>
  <label>搜索<input id="search" placeholder="proposal / scene / lane"></label>
  <button id="next-unreviewed" type="button">跳到下一个未审核</button>
  <strong id="review-summary">正在读取审核进度……</strong>
</div>
<p>共 __COUNT__ 个自动时序保留候选。提交会直接、原子地写入 evaluator-private 工作日志。</p>
__CARDS__
<script src="review.private.js"></script>
</body></html>
""".replace("__COUNT__", str(survivor_count)).replace("__CARDS__", "".join(cards))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    (output.parent / "review.private.js").write_text(
        REVIEW_JAVASCRIPT.strip().replace("/api/reviews", api_base) + "\n", encoding="utf-8"
    )
    return survivor_count


def build_curation_review(
    topology_report_path: Path,
    preview_inventory_path: Path,
    preview_root: Path,
    output_dir: Path,
    project_root: Path,
    render_evidence: bool = True,
) -> Dict[str, Any]:
    """Create fail-closed private review proposals from real mined evidence."""

    topology = load_json(topology_report_path)
    preview_inventory = load_json(preview_inventory_path)
    previews = {row["clip_id"]: row for row in preview_inventory.get("scenes", [])}
    output_dir = Path(output_dir)
    scenes: List[Dict[str, Any]] = []
    proposal_index = 0

    for mined_scene in topology.get("scenes", []):
        route = mined_scene.get("route_aligned_topology", {})
        if (
            mined_scene.get("full_artifact_verified") is not True
            or route.get("available") is not True
        ):
            continue
        scene_id = str(mined_scene["scene_id_private"])
        clip_id = scene_id[len("clipgt-") :] if scene_id.startswith("clipgt-") else scene_id
        preview = previews.get(clip_id, {})
        video = _video_path(Path(preview_root), clip_id)
        duration_s = float(preview.get("duration_s") or 20.0)
        proposal_inputs = [
            ("route_branch", row)
            for row in route.get("route_branch_hypotheses", [])
        ] + [
            ("lane_choice", row)
            for row in route.get("lane_choice_hypotheses", [])
        ]
        proposal_inputs.sort(
            key=lambda item: (int(item[1].get("offset_us", 0)), item[0])
        )
        proposals = []
        for proposal_type, hypothesis in proposal_inputs:
            proposal_index += 1
            proposal_id = "review-%06d" % proposal_index
            proposal_key = "opportunity-" + stable_hash(
                {
                    "artifact_uuid": mined_scene["artifact_uuid"],
                    "proposal_type": proposal_type,
                    "offset_us": int(hypothesis["offset_us"]),
                    "lane_id": str(hypothesis["lane_id"]),
                    "legal_next_lanes": list(
                        hypothesis.get("legal_next_lanes", [])
                    ),
                    "observed_recorded_next_lane": hypothesis.get(
                        "observed_recorded_next_lane"
                    ),
                    "adjacent_lanes_left": list(
                        hypothesis.get("adjacent_lanes_left", [])
                    ),
                    "adjacent_lanes_right": list(
                        hypothesis.get("adjacent_lanes_right", [])
                    ),
                }
            )[:20]
            decision_s = float(hypothesis["offset_us"]) / 1_000_000.0
            pre_context_s = max(0.0, decision_s)
            post_context_s = max(0.0, duration_s - decision_s)
            timing_rejections = []
            if pre_context_s < MINIMUM_PRE_DECISION_CONTEXT_S:
                timing_rejections.append("insufficient_pre_decision_context")
            if post_context_s < MINIMUM_POST_DECISION_CONTEXT_S:
                timing_rejections.append("insufficient_post_decision_context")
            sample_times_s = decision_sample_times(decision_s, duration_s)
            sheet = output_dir / clip_id / (proposal_id + ".jpg")
            map_svg = output_dir / clip_id / (proposal_id + ".map.svg")
            render_error = None
            rendered = False
            map_error = None
            map_rendered = False
            if render_evidence:
                if not video.is_file():
                    render_error = "preview_video_not_found"
                else:
                    try:
                        _render_sheet(video, sample_times_s, sheet)
                        rendered = True
                    except (OSError, subprocess.CalledProcessError) as error:
                        render_error = "evidence_render_failed:%s" % type(error).__name__
                try:
                    _render_map_svg(
                        hypothesis,
                        route.get("recorded_trajectory_xyz_private", []),
                        map_svg,
                    )
                    map_rendered = True
                except (OSError, TypeError, ValueError) as error:
                    map_error = "map_render_failed:%s" % type(error).__name__
            topology_private = {
                key: hypothesis[key]
                for key in (
                    "legal_next_lanes",
                    "observed_recorded_next_lane",
                    "adjacent_lanes_left",
                    "adjacent_lanes_right",
                )
                if key in hypothesis
            }
            topology_private["lane_geometry_private_not_qualified"] = hypothesis.get(
                "lane_geometry_private_not_qualified", {}
            )
            topology_private["lane_geometry_errors"] = hypothesis.get(
                "lane_geometry_errors", []
            )
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "proposal_key": proposal_key,
                    "proposal_type": proposal_type,
                    "candidate_families_not_qualified": list(
                        hypothesis.get("hypotheses_not_qualified", [])
                    ),
                    "decision_offset_us": int(hypothesis["offset_us"]),
                    "timing_observation_private_not_qualified": hypothesis.get(
                        "timing_observation_not_qualified",
                        {"marker_semantics": "lane_sample_marker_not_verified_commitment_time",
                         "exact_gate_crossing_verified": False}),
                    "suggested_start_time_offset_us_not_qualified": int(
                        hypothesis["suggested_start_time_offset_us_not_qualified"]
                    ),
                    "position_xyz_private": list(hypothesis["position_xyz"]),
                    "lane_id_private": str(hypothesis["lane_id"]),
                    "topology_private": topology_private,
                    "sample_times_s": sample_times_s,
                    "timing_review": {
                        "pre_decision_context_s": pre_context_s,
                        "post_decision_context_s": post_context_s,
                        "minimum_pre_decision_context_s": MINIMUM_PRE_DECISION_CONTEXT_S,
                        "minimum_post_decision_context_s": MINIMUM_POST_DECISION_CONTEXT_S,
                        "automatic_timing_prefilter_pass": not timing_rejections,
                        "automatic_rejection_reasons": timing_rejections,
                    },
                    "model_input_resolution": MODEL_INPUT_RESOLUTION,
                    "model_input_dimensions": MODEL_INPUT_DIMENSIONS,
                    "evidence_sheet": _portable_path(sheet, project_root),
                    "evidence_rendered": rendered,
                    "evidence_error": render_error,
                    "map_evidence_svg": _portable_path(map_svg, project_root),
                    "map_evidence_rendered": map_rendered,
                    "map_evidence_error": map_error,
                    "curator_decisions": _curator_decisions(),
                    "qualification_state": "CURATOR_REVIEW_REQUIRED",
                }
            )
        scenes.append(
            {
                "scene_id_private": scene_id,
                "artifact_uuid": mined_scene["artifact_uuid"],
                "preview_video": _portable_path(video, project_root),
                "preview_video_present": video.is_file(),
                "preview_stratum": preview.get("stratum"),
                "preview_labels": preview.get("labels", {}),
                "generic_preview_ocr_untrusted": preview.get("ocr_text", ""),
                "recorded_trajectory_xyz_private": route.get(
                    "recorded_trajectory_xyz_private", []
                ),
                "proposal_count": len(proposals),
                "proposals": proposals,
            }
        )

    proposals = [proposal for scene in scenes for proposal in scene["proposals"]]
    review_index = output_dir / "index.private.html"
    review_index_error = None
    review_index_survivor_count = 0
    try:
        review_index_survivor_count = _render_review_index(
            scenes, review_index, project_root
        )
    except (OSError, TypeError, ValueError) as error:
        review_index_error = "review_index_render_failed:%s" % type(error).__name__
    return {
        "schema_version": "l2d.realsim.curation_review.private.v0.1",
        "evidence_scope": "recorded_route_topology_plus_real_preview_frames",
        "qualification_warning": (
            "These packets are curator aids only. Unset review fields, preview frames, "
            "and topology hypotheses never establish PCU eligibility."
        ),
        "model_input_resolution": MODEL_INPUT_RESOLUTION,
        "model_input_dimensions": MODEL_INPUT_DIMENSIONS,
        "evidence_rendering_enabled": bool(render_evidence),
        "review_index": _portable_path(review_index, project_root),
        "review_index_error": review_index_error,
        "review_index_survivor_count": review_index_survivor_count,
        "source_scene_count": len(topology.get("scenes", [])),
        "review_scene_count": len(scenes),
        "proposal_count": len(proposals),
        "rendered_evidence_count": sum(
            proposal["evidence_rendered"] for proposal in proposals
        ),
        "evidence_error_count": sum(
            proposal["evidence_error"] is not None for proposal in proposals
        ),
        "rendered_map_count": sum(
            proposal["map_evidence_rendered"] for proposal in proposals
        ),
        "map_error_count": sum(
            proposal["map_evidence_error"] is not None for proposal in proposals
        ),
        "timing_prefilter_pass_count": sum(
            proposal["timing_review"]["automatic_timing_prefilter_pass"]
            for proposal in proposals
        ),
        "timing_prefilter_reject_count": sum(
            not proposal["timing_review"]["automatic_timing_prefilter_pass"]
            for proposal in proposals
        ),
        "timing_prefilter_survivor_count": sum(
            proposal["timing_review"]["automatic_timing_prefilter_pass"]
            for proposal in proposals
        ),
        "pending_curator_review_count": len(proposals),
        "auto_qualified_count": 0,
        "scenes": scenes,
    }

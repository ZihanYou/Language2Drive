"""Explicit AlpaSim.md §13 protocols, not inferred reconstruction geometry."""

NATIVE_MODE = "alpasim_native_black_image_with_review"
POLYGON_MODE = "reviewed_support_polygon_and_native_black_image"


def native_render_protocol(annotations):
    """Opt-in only. Existing polygon-based annotations retain their semantics.

    The caller's source validation still requires legality/balance/render review.
    A native metric is an operational validity signal, not proof of a continuous
    spatial support region. Pilot references and this choice enter frozen source.
    """
    protocol = annotations.get("render_validity_protocol")
    if protocol is None:
        return False
    if not isinstance(protocol, dict) or protocol.get("mode") != NATIVE_MODE:
        raise ValueError("unsupported explicit render validity protocol")
    if annotations.get("reconstruction_support_polygon") is not None:
        raise ValueError("native-only protocol cannot silently override a support polygon")
    if (protocol.get("native_metric") != "img_is_black"
        or type(protocol.get("maximum_black_samples")) is not int
        or protocol["maximum_black_samples"] != 0
        or protocol.get("frozen_before_model_evaluation") is not True):
        raise ValueError("native validity requires frozen zero-black-sample policy")
    refs = protocol.get("pilot_evidence_refs")
    if (not isinstance(refs, list) or not refs
        or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
        or len(refs) != len(set(refs))):
        raise ValueError("native validity requires distinct pilot evidence references")
    return True

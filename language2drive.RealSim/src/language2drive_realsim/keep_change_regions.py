"""Opt-in, original-linked planning regions; never replace the native road map."""
import numpy as np
from shapely.geometry import Polygon, mapping

from .corridor_sections import linked_section_region

LINKED_VERSION = "exclusive_lane_dwell_linked_original_v3"


def original_linked_lane_region(vector_map, lane_ids):
    """Rebuild sections from the actual map, not caller-provided cached polygons.

    Only reciprocal original directed links may connect successive sections.
    The existing five-centimeter rail cap policy remains unchanged; rejected
    cap joins stay gaps. Native road and traffic checks are still mandatory.
    """
    if (not isinstance(lane_ids, list) or not lane_ids
        or any(not isinstance(v, str) or not v or v == "-1" for v in lane_ids)
        or len(set(lane_ids)) != len(lane_ids)):
        raise ValueError("distinct original lane identifiers required")
    lanes = [vector_map.get_road_lane(v) for v in lane_ids]
    sections, links = [], []
    for lane_id, lane in zip(lane_ids, lanes):
        if str(lane.id) != lane_id or lane.left_edge is None or lane.right_edge is None:
            raise ValueError("actual original lane identity and both rails required")
        polygon = Polygon(np.concatenate((lane.left_edge.xyz[:, :2], lane.right_edge.xyz[::-1, :2])))
        sections.append(dict(lane_id=lane_id, centerline_xyz=lane.center.xyz.tolist(), polygon=mapping(polygon)))
    for i in range(len(lanes)-1):
        left, right = lane_ids[i:i+2]
        if (right not in {str(v) for v in lanes[i].next_lanes}
            or left not in {str(v) for v in lanes[i+1].prev_lanes}):
            raise ValueError("reciprocal original directed section links required")
        links.append((left, right))
    region, audit = linked_section_region(sections, links)
    audit.update(original_sections=sections, original_directed_links=links,
                 native_original_map_modified=False, full_native_road_check_required=True,
                 source="current_actual_vector_map_original_rails_not_external_cached_geometry")
    return region, audit

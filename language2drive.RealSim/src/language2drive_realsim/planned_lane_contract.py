"""Original-lane first-commit checks on plans, never actual witness evidence."""
import json
import numpy as np

from .events import point_in_polygon
from .spatial_draft import lane_spatial_coverage
from .verifier import _lane_verifier

VERSION = "exclusive_lane_original_first_commit_v4"


def validate_predeclared_lane_geometry(predeclared, coverage, polygons, contracts=None, scope=None):
    """Reject post-run section/scope changes; this grants no actual approval.

    Contract scope names may be rebound by the existing draft/source pipeline,
    but both arms must still reference the identical pre-run geometry.
    """
    canonical=lambda value: json.loads(json.dumps(value))
    planned=predeclared['scope_selection']
    if (predeclared.get('all_own_pass_opposing_fail') is not True
        or predeclared.get('original_geometry_modified') is not False
        or coverage.get('unique_original_sections_both_arms') is not True
        or coverage.get('dwell_interval_s') != planned['dwell_interval_s']
        or canonical(coverage.get('shared_scope_proposal_geojson')) != predeclared['shared_scope']):
        raise ValueError('actual spatial scope must match the predeclared original lane contract')
    for arm in ('R0','R1'):
        selected=planned['arms'][arm]['covering_original_lane_ids']
        points=canonical(polygons.get(selected[0])) if len(selected) == 1 else None
        # Raw original rails use implicit closure; GeoJSON uses an explicit
        # repeated endpoint. Normalize representation only, not any vertex.
        if isinstance(points,list) and points and points[-1] != points[0]:
            points=points+[points[0]]
        if (coverage['arms'][arm]['covering_original_lane_ids'] != selected
            or len(selected) != 1
            or points != predeclared['original_lane_polygons'][selected[0]]):
            raise ValueError('actual lane selection or original polygon changed after preparation')
    if contracts is not None:
        if set(contracts) != {'R0','R1'} or not contracts['R0'].get('scope'):
            raise ValueError('two bound actual lane contracts required')
        for arm in ('R0','R1'):
            expected=dict(predeclared['contracts'][arm],scope=contracts['R0']['scope'])
            if contracts[arm] != expected:
                raise ValueError('actual semantic contract changed after preparation')
        if canonical(scope) != predeclared['shared_scope']:
            raise ValueError('actual contract scope geometry changed after preparation')


def audit_planned_lane_contract(plans, rig_to_aabb, sections_by_arm):
    """Fix the union of unique original final-dwell lanes, then check both arms.

    The original 1.7-second intervention, 7-second deadline, one-second stable
    commitment and opposing-first rule are unchanged. No scope search, boundary
    shift, cap join or safety/render flags enter this semantic component.
    """
    if set(plans) != {'R0','R1'} or set(sections_by_arm) != {'R0','R1'}:
        raise ValueError('both plans and original lane section groups required')
    transform=np.asarray(rig_to_aabb,dtype=float)
    if (transform.shape != (4,4) or not np.isfinite(transform).all()
        or not np.allclose(transform[3],[0,0,0,1],atol=1e-8,rtol=0)
        or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-6,rtol=0)
        or not np.isclose(np.linalg.det(transform[:3,:3]),1.,atol=1e-6,rtol=0)):
        raise ValueError('finite rigid original rig-to-AABB transform required')
    from shapely.geometry import shape
    polygons,ids={},{}
    for arm,sections in sections_by_arm.items():
        if not isinstance(sections,list) or not sections:
            raise ValueError('nonempty original section lists required')
        ids[arm]=[]
        for section in sections:
            key=section['lane_id']
            polygon=shape(section['polygon'])
            if (not isinstance(key,str) or not key or key in ids[arm]
                or polygon.geom_type != 'Polygon' or not polygon.is_valid or polygon.interiors
                or polygon.is_empty or polygon.area <= 0):
                raise ValueError('distinct valid original lane polygons required')
            coords=[list(p) for p in polygon.exterior.coords]
            if key in polygons and polygons[key] != coords:
                raise ValueError('one immutable polygon per original lane identity required')
            polygons[key]=coords
            ids[arm].append(key)
    traces,prefixes={},[]
    for arm,plan in plans.items():
        rows=np.asarray([[w[k] for k in ('time_s','x','y','z','yaw_rad')] for w in plan['waypoints']],dtype=float)
        if (rows.shape != (131,5) or not np.isfinite(rows).all()
            or not np.array_equal(rows[:,0],np.round(np.arange(131)/10.,6))):
            raise ValueError('exact finite 13-second plan grid required')
        prefixes.append(rows[rows[:,0]<=1.7].tolist())
        yaw=rows[:,4]; offset=transform[:3,3]
        xy=rows[:,1:3]+np.c_[np.cos(yaw)*offset[0]-np.sin(yaw)*offset[1],
                             np.sin(yaw)*offset[0]+np.cos(yaw)*offset[1]]
        traces[arm]=np.c_[rows[:,0],xy]
    if prefixes[0] != prefixes[1]:
        raise ValueError('identical preintervention planned prefix required')
    # Canonical JSON rings are the semantic reader's existing public shape;
    # Shapely mapping emits tuples, which must not bypass that strict reader.
    coverage=json.loads(json.dumps(lane_spatial_coverage(traces,polygons,ids,6.,7.)))
    coverage['scope']='planned_AABB_original_lane_selection_not_actual_evidence'
    report=dict(scope='planned_original_lane_first_commit_component_not_qualification',qualified=False,
        human_reviewed=False,position_reference='planned_yaw_only_rig_transformed_to_ego_AABB_center',
        intervention_s=1.7,deadline_s=7.,stable_duration_s=1.,original_geometry_modified=False,
        full_safety_or_render_verifier_executed=False,scope_selection=coverage,
        contracts=None,original_lane_polygons=None,shared_scope=None,arms={},all_own_pass_opposing_fail=False)
    if not coverage['unique_original_sections_both_arms']:
        report['rejection']='no_unique_original_final_dwell_sections'
        return json.loads(json.dumps(report))
    selected={a:coverage['arms'][a]['covering_original_lane_ids'][0] for a in ('R0','R1')}
    if selected['R0']==selected['R1']:
        raise ValueError('different original target lanes required')
    scope=coverage['shared_scope_proposal_geojson']
    lanes={k:polygons[k] for k in selected.values()}
    contracts={a:dict(kind='lane',target_lane=k,scope='predeclared-original-lane-scope',
                      stable_duration_s=1.,deadline_s=7.) for a,k in selected.items()}
    for arm,trace in traces.items():
        other='R1' if arm=='R0' else 'R0'
        rows=[dict(time_s=float(t),position=[float(x),float(y)],
            lane_ids=[k for k,p in lanes.items() if point_in_polygon([x,y],p)],
            scopes=['predeclared-original-lane-scope'] if point_in_polygon([x,y],scope) else [])
            for t,x,y in trace if t>=1.7]
        report['arms'][arm]=dict(own=_lane_verifier(contracts[arm],contracts[other],rows).to_dict(),
            opposing=_lane_verifier(contracts[other],contracts[arm],rows).to_dict())
    report.update(contracts=contracts,original_lane_polygons=lanes,shared_scope=scope,
        all_own_pass_opposing_fail=all(a['own']['passed'] and not a['opposing']['passed'] for a in report['arms'].values()))
    return json.loads(json.dumps(report))

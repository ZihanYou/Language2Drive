"""Fixed, loadable Bench2Drive scenario coverage for the FDQ workload."""

from __future__ import annotations


# One route per scenario type where the corresponding CARLA town is installed
# and passes the map-load audit.  The remaining XML scenario types only occur
# on unavailable Town07/Town12/Town13/Town15 maps in this installation.
# The first four stable IDs retain the original public FDQ names.
FDQ_ROUTE_BINDING_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("overtaking", "24240", "Town10HD", "Accident"),
    ("accident_two_ways", "25857", "Town05", "AccidentTwoWays"),
    ("blocked_intersection", "27494", "Town04", "BlockedIntersection"),
    ("construction_obstacle", "24785", "Town04", "ConstructionObstacle"),
    ("control_loss", "24340", "Town10HD", "ControlLoss"),
    ("emergency_brake", "24211", "Town01", "DynamicObjectCrossing"),
    ("enter_actor_flow", "27506", "Town05", "EnterActorFlow"),
    ("hard_break_route", "24330", "Town10HD", "HardBreakRoute"),
    ("hazard_at_side_lane", "25300", "Town04", "HazardAtSideLane"),
    ("hazard_at_side_lane_two_ways", "25854", "Town03", "HazardAtSideLaneTwoWays"),
    ("non_signalized_junction_left_turn", "26990", "Town03", "NonSignalizedJunctionLeftTurn"),
    ("non_signalized_junction_left_turn_enter_flow", "28087", "Town04", "NonSignalizedJunctionLeftTurnEnterFlow"),
    ("opposite_vehicle_running_red_light", "26944", "Town04", "OppositeVehicleRunningRedLight"),
    ("parked_obstacle", "25318", "Town05", "ParkedObstacle"),
    ("parked_obstacle_two_ways", "25896", "Town03", "ParkedObstacleTwoWays"),
    ("parking_crossing_pedestrian", "24206", "Town03", "ParkingCrossingPedestrian"),
    ("parking_cut_in", "24759", "Town05", "ParkingCutIn"),
    ("merging", "26393", "Town03", "ParkingExit"),
    ("pedestrian_crossing", "27515", "Town03", "PedestrianCrossing"),
    ("signalized_junction_left_turn_enter_flow", "28099", "Town04", "SignalizedJunctionLeftTurnEnterFlow"),
    ("signalized_junction_right_turn", "26956", "Town05", "SignalizedJunctionRightTurn"),
    ("static_cut_in", "26396", "Town05", "StaticCutIn"),
    ("t_junction", "28035", "Town01", "T_Junction"),
    ("give_way", "25378", "Town03", "YieldToEmergencyVehicle"),
)

FDQ_SEEDS = (1, 2, 3)


def canonical_fdq_route_bindings() -> dict[str, dict[str, str]]:
    """Return fresh dictionaries so plans cannot mutate module authority."""

    return {
        identifier: {"route_id": route_id, "town": town, "scenario": scenario}
        for identifier, route_id, town, scenario in FDQ_ROUTE_BINDING_ROWS
    }

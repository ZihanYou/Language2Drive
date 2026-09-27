SCHEMA_VERSION = "l2d.realsim.pcu.v0.1"
TRACE_SCHEMA_VERSION = "l2d.realsim.trace.v0.1"
RUN_MANIFEST_VERSION = "l2d.realsim.run.v0.1"
ALPASIM_COMMIT = "affc2eab209fa43bdfa2f26c0f8d437922d78a68"
NUREC_DATASET_REVISION = "26.01"
NUREC_VERSION_STRING = "26.1.112-59673f88"
ORACLE_CONTROLLER_ID = "privileged_oracle_path_v0.1"

ARMS = ("R0", "R1")
COUPLED_SEEDS = (1701, 1702, 1703)
LAYER_TARGETS = {"A": 20, "B": 18, "C": 12}
FAMILY_TARGETS = {
    "A1_intersection_left_right": 5,
    "A2_straight_turn": 4,
    "A3_fork_left_right": 3,
    "A4_lane_change_left_right": 4,
    "A5_lane_change_keep": 2,
    "A6_pull_over_continue": 2,
    "B1_segment_speed_band": 6,
    "B2_scoped_lane_preference": 4,
    "B3_triggered_speed_constraint": 4,
    "B4_object_relative_passing": 2,
    "B5_relative_stopping_region": 2,
    "C1_direction_sign_grounding": 4,
    "C2_road_exit_text_grounding": 3,
    "C3_landmark_adjacent_branch": 3,
    "C4_attribute_landmark_target": 2,
}

FAILURES = {
    "F0_PREINTERVENTION",
    "F1_NO_COMMIT",
    "F2_WRONG_SEMANTIC",
    "F3_UNSAFE_FOLLOW",
    "F4_WRONG_REFERENT_OUTCOME",
    "F5_QUANTITATIVE",
    "F6_ORDER_SCOPE",
    "F10_ANTI_PROGRESS_PATTERN",
    "INVALID_SIM",
}

# Identifiers and filenames in the model-visible bundle must not encode answers.
LEAKAGE_TOKENS = {
    "left",
    "right",
    "straight",
    "airport",
    "hospital",
    "fast",
    "slow",
    "correct",
    "positive",
    "r0",
    "r1",
}

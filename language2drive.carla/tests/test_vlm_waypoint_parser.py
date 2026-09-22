from model_zoo.language2drive_adapters.vlm_backends import parse_waypoints


def test_parse_waypoints_accepts_explicit_plus_and_trailing_comma():
    output = (
        "<answer>[[0.00, 0.00], [+0.08, 0.00], [+0.36, 0.00], "
        "[+0.93, -0.02], [+1.80, -0.03], [+2.95, -0.05], "
        "[+4.37, -0.08], [+5.99, -0.14,]]</answer>"
    )
    assert parse_waypoints(output) == [
        (0.0, 0.0),
        (0.08, 0.0),
        (0.36, 0.0),
        (0.93, -0.02),
        (1.8, -0.03),
        (2.95, -0.05),
        (4.37, -0.08),
        (5.99, -0.14),
    ]


def test_parse_waypoints_prefers_answer_section_over_reasoning_numbers():
    output = "reasoning at speed 3.5 <answer>[[+1, -2], [+3, -4]]</answer>"
    assert parse_waypoints(output) == [(1.0, -2.0), (3.0, -4.0)]


def test_parse_waypoints_does_not_let_one_final_triple_hide_pairs():
    output = (
        "<answer>[[0.00, 0.00], [0.02, 0.00], [0.16, 0.00], "
        "[0.50, 0.00], [1.05, 0.02], [1.83, 0.02], [2.86, 0.03], "
        "[4.09, 0.04, 0.02]]."
    )
    assert parse_waypoints(output) == [
        (0.0, 0.0),
        (0.02, 0.0),
        (0.16, 0.0),
        (0.5, 0.0),
        (1.05, 0.02),
        (1.83, 0.02),
        (2.86, 0.03),
    ]

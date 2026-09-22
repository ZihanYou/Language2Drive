"""Minimal executable fixed-policy controls for Language2Drive anti-gaming.

This agent deliberately ignores language and route identity.  It is a causal
control, not a competent driving baseline.  Failures, collisions, and timeouts
remain valid benchmark outcomes in the expected-run denominator.
"""

from __future__ import annotations

import json
from pathlib import Path

import carla
from leaderboard.autoagents import autonomous_agent


def get_entry_point():
    return "Language2DriveFixedPolicyAgent"


class Language2DriveFixedPolicyAgent(autonomous_agent.AutonomousAgent):
    def set_language2drive_context(self, public_context):
        # Keep only a receipt that the protocol call happened.  The instruction
        # value is intentionally never retained or inspected.
        self.context_received = isinstance(public_context, dict)
        return {"strategy": getattr(self, "strategy", "not_configured"), "instruction_ignored": True}

    def setup(self, path_to_conf_file, route_index=None):
        del route_index
        config_path = Path(str(path_to_conf_file).split("+", 1)[0]).expanduser()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.strategy = str(config["strategy"])
        if self.strategy not in {"always-left", "always-right", "fixed-branch"}:
            raise ValueError(f"unsupported fixed policy {self.strategy!r}")
        self.target_speed_mps = float(config.get("target_speed_mps", 3.0))
        self.fixed_steer = float(config["steer"])
        self.track = autonomous_agent.Track.SENSORS

    def sensors(self):
        return [{"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"}]

    def run_step(self, input_data, timestamp, sensors=None):
        del timestamp, sensors
        measurement = input_data.get("speed", (None, {"speed": 0.0}))[1]
        speed = float(measurement.get("speed", 0.0)) if isinstance(measurement, dict) else 0.0
        control = carla.VehicleControl()
        control.steer = max(-1.0, min(1.0, self.fixed_steer))
        error = self.target_speed_mps - speed
        control.throttle = max(0.0, min(0.45, 0.18 * error))
        control.brake = max(0.0, min(1.0, -0.35 * error))
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    def destroy(self):
        return None

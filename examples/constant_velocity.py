"""CPU interface demo, not a driving VLA or a paper baseline."""
class ConstantVelocity:
    def __init__(self, config):
        self.speed = float(config.get("speed_mps", 2.0))

    def predict(self, request):
        spec = request.output_spec
        return {"coordinate_frame": spec.coordinate_frame,
                "reference_point": spec.reference_point,
                "timestamps_s": list(spec.timestamps_s),
                "xy_m": [[self.speed * t, 0.0] for t in spec.timestamps_s]}


def create_adapter(config):
    return ConstantVelocity(config)

"""Replace the marked model calls once; all track exporters use this adapter."""
from l2d_benchmark.service import decode_image

class Adapter:
    def __init__(self, config):
        # Load your frozen weights here. Share immutable weights between session
        # wrappers if serving many concurrent RealSim sessions.
        self.config = config
        self.memory = None

    def predict(self, request):
        # Paths point to local public sensors. Live frames are PNG base64.
        cameras = [decode_image(c) for c in request.observation.get("cameras", [])]
        # Implement your model's documented preprocessing here. The observation
        # also contains ego_history / speed_mps when available and public native
        # metadata. Do not infer evaluator labels from filenames or instruction IDs.
        raise NotImplementedError("Call your model with cameras, observed history, request.instruction and request.seed; return finite XY metres at request.output_spec.timestamps_s in its declared frame")
        # return {"coordinate_frame": request.output_spec.coordinate_frame,
        #         "reference_point": request.output_spec.reference_point,
        #         "timestamps_s": list(request.output_spec.timestamps_s),
        #         "xy_m": xy_in_requested_frame}


def create_adapter(config):
    return Adapter(config)

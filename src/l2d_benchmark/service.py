"""Local JSON inference service; the model's dependencies stay in its own env."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import base64
import io
import hashlib
from pathlib import Path
import json
from urllib.request import Request as HTTPRequest, urlopen
from urllib.error import HTTPError
from .adapter import Request, load_adapter, predict


def image_payload(array):
    import numpy as np
    from PIL import Image
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError('Camera images must be HWC uint8 RGB')
    image = Image.fromarray(array)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return {"encoding": "png_base64", "data": base64.b64encode(out.getvalue()).decode("ascii")}


def decode_image(camera):
    """Return a PIL RGB image for either file-backed or live camera input."""
    from PIL import Image
    if camera.get("encoding") == "png_base64":
        return Image.open(io.BytesIO(base64.b64decode(camera["data"]))).convert("RGB")
    return Image.open(camera["path"]).convert("RGB")


class Client:
    def __init__(self, endpoint="http://127.0.0.1:8090", timeout=300):
        self.endpoint, self.timeout = endpoint.rstrip("/"), timeout

    def call(self, method, payload):
        data = json.dumps(payload, allow_nan=False).encode()
        req = HTTPRequest(self.endpoint + "/" + method, data=data,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            try:
                message = json.loads(error.read()).get('error', str(error))
            except (ValueError, AttributeError):
                message = str(error)
            raise ValueError(f'Model service {method}: {message}') from error

    def identity(self):
        with urlopen(self.endpoint + '/health', timeout=self.timeout) as response:
            return json.load(response)['model_identity']

    def reset(self, session):
        return self.call("reset", {"session": session})

    def predict(self, request, session):
        return self.call("predict", {"session": session, "request": request.to_dict()})


def serve(adapter_path, config, host="127.0.0.1", port=8090):
    # A session gets its own adapter state. This isolates recurrent model memory
    # across PCU arms and concurrently scheduled simulator sessions.
    adapters = {}
    identity = {"adapter_sha256": hashlib.sha256(Path(adapter_path).read_bytes()).hexdigest(), "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps({'status': 'ready', 'model_identity': identity}).encode())

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 128 * 1024 * 1024:
                    raise ValueError("Invalid payload size")
                payload = json.loads(self.rfile.read(length))
                session = payload["session"]
                if not isinstance(session, str) or not session:
                    raise ValueError("session must be a nonempty opaque string")
                if self.path == "/reset":
                    if session in adapters:
                        close = getattr(adapters.pop(session), "close", None)
                        if close: close()
                    # Reuse immutable model weights through the user's factory,
                    # while returning a separate session wrapper/state.
                    adapters[session] = load_adapter(adapter_path, config)
                    result = {"status": "reset"}
                elif self.path == "/predict":
                    if session not in adapters:
                        raise ValueError("Reset the session before prediction")
                    result = predict(adapters[session], Request.from_dict(payload["request"]))
                elif self.path == "/close":
                    adapter = adapters.pop(session, None)
                    close = getattr(adapter, "close", None)
                    if close: close()
                    result = {"status": "closed"}
                else:
                    self.send_error(404); return
                raw = json.dumps(result, allow_nan=False).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
            except Exception as error:
                self.send_response(422); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"error": str(error)}).encode())
        def log_message(self, *args):
            pass
    print(f"Model service: http://{host}:{port}", flush=True)
    HTTPServer((host, port), Handler).serve_forever()

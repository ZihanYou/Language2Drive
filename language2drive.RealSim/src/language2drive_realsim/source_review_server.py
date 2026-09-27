"""Isolated loopback source-review service. No writes to older review APIs."""
import ipaddress
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .curation_review_server import MAX_BODY_BYTES, ReviewRequestHandler
from .source_review import SourceReviewStore
from .source_review_ui import PAGE, SCRIPT


class SourceReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store, static_root):
        super().__init__(address, SourceReviewHandler)
        self.source_store = store
        self.static_root = Path(static_root).resolve()


class SourceReviewHandler(ReviewRequestHandler):
    def _valid_host(self):
        try:
            hostname = urlsplit('http://' + self.headers.get('Host', '')).hostname
            return hostname == 'localhost' or ipaddress.ip_address(hostname).is_loopback
        except (ValueError, TypeError):
            return False

    def _text(self, text, content_type):
        body = text.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._valid_host():
            self._json(403, {'error': '仅允许本机／SSH 转发 Host'})
            return
        path = unquote(urlsplit(self.path).path)
        try:
            if path in ('/', '/index.private.html'):
                self._text(PAGE, 'text/html')
            elif path == '/source-review.js':
                self._text(SCRIPT, 'text/javascript')
            elif path == '/api/health':
                self._json(200, {'ok': True, 'loopback_only': True, 'source_review_only': True})
            elif path == '/api/source-reviews':
                self._json(200, self.server.source_store.overview())
            elif path.startswith('/api/source-reviews/'):
                self._json(200, self.server.source_store.snapshot(path[len('/api/source-reviews/'):]))
            elif path.startswith('/api/'):
                self._json(404, {'error': '本端口不提供旧审核接口'})
            else:
                target = Path(self.translate_path(self.path)).resolve()
                if not target.is_relative_to(self.server.static_root):
                    self._json(403, {'error': '静态证据路径越界'})
                    return
                SimpleHTTPRequestHandler.do_GET(self)
        except (ValueError, OSError, KeyError) as exc:
            self._json(409, {'error': str(exc)})

    def do_HEAD(self):
        # Do not inherit an unchecked static-file HEAD path.
        self._json(405, {'error': '请使用 GET'})

    def do_POST(self):
        path = unquote(urlsplit(self.path).path)
        prefix = '/api/source-reviews/'
        if not path.startswith(prefix) or not path[len(prefix):]:
            self._json(405, {'error': '本端口只保存完整源稿终审'})
            return
        origin = self.headers.get('Origin')
        if (not self._valid_host() or not self._same_origin_or_no_origin()
            or (origin and urlsplit(origin).scheme != 'http')):
            self._json(403, {'error': 'origin 或 Host 不匹配'})
            return
        if self.headers.get_content_type() != 'application/json':
            self._json(415, {'error': '必须使用 application/json'})
            return
        try:
            size = int(self.headers.get('Content-Length', ''))
            if not 0 < size <= MAX_BODY_BYTES:
                self._json(413, {'error': '提交大小必须为 1–32768 字节'})
                return
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError('JSON 键重复')
                    result[key] = value
                return result
            payload = json.loads(self.rfile.read(size).decode('utf-8'), object_pairs_hook=unique,
                parse_constant=lambda v: (_ for _ in ()).throw(ValueError('非标准 JSON 常量')))
            result = self.server.source_store.submit(path[len(prefix):], payload)
            self._json(200, {'ok': True, **result})
        except (ValueError, UnicodeDecodeError, KeyError) as exc:
            self._json(400, {'error': str(exc)})
        except OSError as exc:
            self._json(500, {'error': str(exc)})


def serve_source_review(root, inventory, decisions, bind='127.0.0.1', port=8767):
    if bind != 'localhost' and not ipaddress.ip_address(bind).is_loopback:
        raise ValueError('只允许回环地址；请通过 SSH 访问')
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('端口范围必须为 0..65535')
    root = Path(root).resolve()
    store = SourceReviewStore(root, inventory, decisions)
    store.overview()
    with SourceReviewHTTPServer((bind, port), store, root / '.runtime/curation_review') as server:
        print(f'PCU source review: http://{bind}:{server.server_port}/; no prior reviews modified', flush=True)
        server.serve_forever()

#!/usr/bin/env python3
"""SSH-tunnel-only, authenticated per-PCU review service."""

from paths import PROJECT_ROOT, DATA_ROOT, ARTIFACT_ROOT, RELEASE_ROOT

import argparse
import base64
import csv
import hashlib
import hmac
import io
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


AUDIT = ARTIFACT_ROOT / 'audit'
FIELDS = ('observable', 'arm_0_feasible', 'arm_1_feasible',
          'mutually_exclusive', 'traffic_clearance', 'map_binding')
EXTRA = ('reviewer', 'submitted_at_utc', 'review_status') + FIELDS
MIME = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
        '.js': 'application/javascript; charset=utf-8', '.png': 'image/png',
        '.jpg': 'image/jpeg'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class ReviewStore:
    def __init__(self, audit, token=None):
        self.audit = Path(audit)
        self.audit.mkdir(parents=True, exist_ok=True)
        self.source = self.audit.parent / 'candidates/all_review_queue.jsonl'
        self.sheet = self.audit / 'review_sheet.csv'
        self.db_path = self.audit / 'reviews.sqlite3'
        self.export_path = self.audit / 'review_sheet_submitted.csv'
        self.source_sha256 = sha256(self.source)
        with self.sheet.open(newline='') as source:
            reader = csv.DictReader(source)
            self.columns = reader.fieldnames
            self.rows = list(reader)
        self.by_id = {row['pcu_id']: row for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError('Duplicate PCU ID in review sheet')
        self.lock = threading.RLock()
        self.token_path = self.audit / 'review_access_token'
        self.token = token or self._load_token()
        self._init_db()
        self.export()

    def _load_token(self):
        try:
            fd = os.open(str(self.token_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'w') as output:
                output.write(secrets.token_urlsafe(32) + '\n')
        os.chmod(str(self.token_path), 0o600)
        return self.token_path.read_text().strip()

    def _connect(self):
        return sqlite3.connect(str(self.db_path), timeout=10)

    def _init_db(self):
        with self._connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pcu_id TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                reviewer TEXT NOT NULL, decision TEXT NOT NULL,
                observable TEXT NOT NULL, arm_0_feasible TEXT NOT NULL,
                arm_1_feasible TEXT NOT NULL, mutually_exclusive TEXT NOT NULL,
                traffic_clearance TEXT NOT NULL, map_binding TEXT NOT NULL,
                notes TEXT NOT NULL, submitted_at_utc TEXT NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS submissions_pcu ON submissions(pcu_id, id)')
        os.chmod(str(self.db_path), 0o600)

    def _records(self, pcu_id=None):
        query = 'SELECT * FROM submissions WHERE source_sha256=?'
        args = [self.source_sha256]
        if pcu_id is not None:
            query += ' AND pcu_id=?'
            args.append(pcu_id)
        query += ' ORDER BY id DESC'
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(query, args)]

    def latest(self):
        result = {}
        for row in self._records():
            result.setdefault(row['pcu_id'], row)
        return result

    def submit(self, data):
        if sha256(self.source) != self.source_sha256:
            raise ValueError('候选源文件已变化，请重启审核服务并核对审核记录')
        if not isinstance(data, dict):
            raise ValueError('提交内容必须是 JSON 对象')
        pcu_id = data.get('pcu_id')
        if pcu_id not in self.by_id:
            raise ValueError('未知候选案例')
        row = self.by_id[pcu_id]
        reviewer = data.get('reviewer')
        notes = data.get('notes')
        decision = data.get('decision')
        if not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 80:
            raise ValueError('请填写审核人，最长 80 字')
        if decision not in ('approve', 'reject', 'needs_more_evidence'):
            raise ValueError('请选择审核结论')
        if not isinstance(notes, str) or not 10 <= len(notes.strip()) <= 2000:
            raise ValueError('请填写至少 10 字、最多 2000 字的审核依据')
        criteria = {}
        for field in FIELDS:
            value = data.get(field)
            if value not in ('yes', 'no', 'unknown'):
                raise ValueError('请逐项填写六项审核判断')
            criteria[field] = value
        if decision == 'approve':
            if row['auto_status'] != 'needs_human_review':
                raise ValueError('自动剔除的案例不能批准')
            if any(value != 'yes' for value in criteria.values()):
                raise ValueError('批准需要六项判断全部为“是”')
        submitted_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
        with self.lock:
            with self._connect() as db:
                db.execute('''INSERT INTO submissions
                    (pcu_id, source_sha256, reviewer, decision, observable,
                     arm_0_feasible, arm_1_feasible, mutually_exclusive,
                     traffic_clearance, map_binding, notes, submitted_at_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (pcu_id, self.source_sha256, reviewer.strip(), decision,
                     *(criteria[field] for field in FIELDS), notes.strip(), submitted_at))
            self.export()
            return self._records(pcu_id)[0]

    def export(self):
        with self.lock:
            if sha256(self.source) != self.source_sha256:
                raise ValueError('候选源文件已变化，请重启审核服务并核对审核记录')
            latest = self.latest()
            output = io.StringIO(newline='')
            writer = csv.DictWriter(output, fieldnames=self.columns + list(EXTRA), extrasaction='ignore')
            writer.writeheader()
            for original in self.rows:
                row = dict(original)
                review = latest.get(row['pcu_id'])
                row['human_decision'] = review['decision'] if review and review['decision'] in ('approve', 'reject') else ''
                row['human_notes'] = review['notes'] if review else ''
                row['reviewer'] = review['reviewer'] if review else ''
                row['submitted_at_utc'] = review['submitted_at_utc'] if review else ''
                row['review_status'] = review['decision'] if review else ''
                for field in FIELDS:
                    row[field] = review[field] if review else ''
                writer.writerow(row)
            payload = output.getvalue().encode('utf-8')
            temporary = self.export_path.with_suffix('.csv.tmp')
            fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'wb') as target:
                target.write(payload)
            os.chmod(str(temporary), 0o600)
            os.replace(str(temporary), str(self.export_path))
            return payload


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload, content_type='application/json; charset=utf-8', extra=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            if extra:
                for key, value in extra.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status, data):
            self._send(status, json.dumps(data, ensure_ascii=False).encode('utf-8'))

        def _authorized(self):
            header = self.headers.get('Authorization', '')
            try:
                scheme, encoded = header.split(' ', 1)
                username, password = base64.b64decode(encoded, validate=True).decode('utf-8').split(':', 1)
            except (ValueError, UnicodeError):
                username, password, scheme = '', '', ''
            if scheme.lower() == 'basic' and username == 'reviewer' and hmac.compare_digest(password, store.token):
                return True
            self._send(HTTPStatus.UNAUTHORIZED, b'Authentication required',
                       'text/plain; charset=utf-8',
                       {'WWW-Authenticate': 'Basic realm="nuScenes review"'})
            return False

        def do_GET(self):
            if not self._authorized():
                return
            url = urlsplit(self.path)
            path = url.path
            if path == '/api/config':
                self._json(200, {'cases': store.rows, 'latest': store.latest(),
                                 'source_sha256': store.source_sha256})
                return
            if path == '/api/history':
                pcu_id = parse_qs(url.query).get('pcu_id', [''])[0]
                if pcu_id not in store.by_id:
                    self._json(404, {'error': '未知案例'})
                else:
                    self._json(200, {'history': store._records(pcu_id)})
                return
            if path == '/api/export.csv':
                self._send(200, store.export(), 'text/csv; charset=utf-8',
                           {'Content-Disposition': 'attachment; filename="review_sheet_submitted.csv"'})
                return
            assets = {'/': store.audit / 'index.html',
                      '/index.html': store.audit / 'index.html',
                      '/review.js': Path(__file__).with_name('review.js'),
                      '/review.css': Path(__file__).with_name('review.css')}
            file = assets.get(path)
            if file is None:
                for directory in ('map_overlays', 'cameras'):
                    prefix = '/' + directory + '/'
                    if path.startswith(prefix):
                        name = path[len(prefix):]
                        if name and '/' not in name and name == Path(name).name and not name.startswith('.'):
                            file = store.audit / directory / name
                        break
            if file is None or file.is_symlink() or not file.is_file() or file.suffix not in MIME:
                self._json(404, {'error': 'Not found'})
                return
            payload = file.read_bytes()
            if path in ('/', '/index.html'):
                payload += (b'\n<link rel="stylesheet" href="/review.css">'
                            b'<script defer src="/review.js"></script>\n')
            self._send(200, payload, MIME[file.suffix])

        def do_POST(self):
            if not self._authorized():
                return
            if urlsplit(self.path).path != '/api/reviews':
                self._json(404, {'error': 'Not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('提交内容过大或为空')
                data = json.loads(self.rfile.read(length))
                review = store.submit(data)
            except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
                self._json(400, {'error': str(exc)})
                return
            self._json(201, {'review': review})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=AUDIT)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    store = ReviewStore(args.audit_dir)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(store))
    print('Review server listening on http://127.0.0.1:{} (SSH tunnel only)'.format(server.server_port), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()

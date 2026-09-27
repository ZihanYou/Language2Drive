"""Review server submission, authentication, revision history and CSV export."""

import base64
import csv
import io
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve_review import FIELDS, ReviewStore, make_handler  # noqa: E402


class ReviewServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        audit = Path(self.temp.name) / 'audit'
        candidates = audit.parent / 'candidates'
        audit.mkdir()
        candidates.mkdir()
        rows=[{'pcu_id':'synthetic-'+str(i),'auto_status':status,
               'family':'scoped_speed','location':'synthetic',
               'arm_0_instruction':'Slow down','arm_1_instruction':'Speed up',
               'human_decision':'','human_notes':''}
              for i,status in enumerate(('needs_human_review','auto_rejected'))]
        with (audit/'review_sheet.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        (candidates/'all_review_queue.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (audit / 'index.html').write_text('<h1>Review</h1>')
        self.store = ReviewStore(audit, token='test-secret')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.store))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = 'http://127.0.0.1:{}'.format(self.server.server_port)
        self.auth = 'Basic ' + base64.b64encode(b'reviewer:test-secret').decode()
        self.pcu = next(row['pcu_id'] for row in self.store.rows if row['auto_status'] == 'needs_human_review')

    def request(self, path, payload=None, auth=True):
        headers = {'Authorization': self.auth} if auth else {}
        data = None if payload is None else json.dumps(payload).encode()
        if data is not None:
            headers['Content-Type'] = 'application/json'
        return urlopen(Request(self.base + path, data=data, headers=headers), timeout=5)

    def payload(self, **overrides):
        result = {'pcu_id': self.pcu, 'reviewer': 'Test reviewer',
                  'decision': 'approve', 'notes': 'Both paths are visually supported by the cameras.'}
        result.update({field: 'yes' for field in FIELDS})
        result.update(overrides)
        return result

    def test_auth_submission_history_and_csv_export(self):
        with self.assertRaises(HTTPError) as error:
            self.request('/', auth=False)
        self.assertEqual(error.exception.code, 401)
        with self.assertRaises(HTTPError) as error:
            self.request('/reviews.sqlite3')
        self.assertEqual(error.exception.code, 404)
        with self.assertRaises(HTTPError) as error:
            self.request('/api/reviews', self.payload(observable='unknown'))
        self.assertEqual(error.exception.code, 400)
        first = json.load(self.request('/api/reviews', self.payload()))['review']
        self.assertEqual(first['decision'], 'approve')
        second = json.load(self.request('/api/reviews', self.payload(
            decision='needs_more_evidence', notes='Need clearer rear camera evidence.',
            observable='unknown')))['review']
        self.assertGreater(second['id'], first['id'])
        history = json.load(self.request('/api/history?pcu_id=' + self.pcu))['history']
        self.assertEqual([row['id'] for row in history], [second['id'], first['id']])
        exported = list(csv.DictReader(io.StringIO(self.request('/api/export.csv').read().decode())))
        row = next(row for row in exported if row['pcu_id'] == self.pcu)
        self.assertEqual(row['human_decision'], '')
        self.assertEqual(row['review_status'], 'needs_more_evidence')
        self.assertEqual(row['reviewer'], 'Test reviewer')
        self.assertEqual(len(exported), len(self.store.rows))
        self.request('/api/reviews', self.payload())
        restarted = ReviewStore(self.store.audit, token='test-secret')
        self.assertEqual(restarted.latest()[self.pcu]['decision'], 'approve')

    def test_auto_rejected_cannot_be_approved(self):
        rejected = next(row['pcu_id'] for row in self.store.rows if row['auto_status'] != 'needs_human_review')
        with self.assertRaises(HTTPError) as error:
            self.request('/api/reviews', self.payload(pcu_id=rejected))
        self.assertEqual(error.exception.code, 400)


if __name__ == '__main__':
    unittest.main()

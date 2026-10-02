#!/usr/bin/env python3
"""Serve the local project preview with byte ranges for video seeking.

ssh -N -L 18765:127.0.0.1:18765 USER@HOST
Then open http://localhost:18765. No public deployment or submission backend.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class PreviewHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range=None
        path=Path(self.translate_path(self.path)).resolve()
        root=Path(self.directory).resolve()
        if not path.is_relative_to(root):
            self.send_error(403);return None
        if path.is_dir():
            path=path/'index.html'
        if not path.is_file():
            self.send_error(404);return None
        size=path.stat().st_size
        start,end=0,size-1
        header=self.headers.get('Range')
        if header:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',header.strip())
            try:
                if not match or not any(match.groups()):raise ValueError
                a,b=match.groups()
                if not a:
                    count=int(b)
                    if count<=0:raise ValueError
                    start=max(0,size-count)
                else:
                    start=int(a);end=min(int(b),size-1) if b else size-1
                if start>end or start>=size:raise ValueError
            except ValueError:
                self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.send_header('Content-Length','0');self.end_headers();return None
            self.byte_range=(start,end)
        stream=path.open('rb')
        self.send_response(206 if header else 200)
        self.send_header('Content-Type',self.guess_type(str(path)))
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('Last-Modified',self.date_time_string(path.stat().st_mtime))
        self.send_header('Cache-Control','no-cache')
        if header:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers();stream.seek(start);return stream

    def copyfile(self,source,output):
        try:
            if self.byte_range is None:return super().copyfile(source,output)
            remaining=self.byte_range[1]-self.byte_range[0]+1
            while remaining:
                data=source.read(min(256*1024,remaining))
                if not data:break
                output.write(data);remaining-=len(data)
        except (BrokenPipeError,ConnectionResetError):
            pass  # A viewer selected another clip or sought to a different time.


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=18765);parser.add_argument('--bind',default='127.0.0.1');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]/'docs'
    server=ThreadingHTTPServer((args.bind,args.port),partial(PreviewHandler,directory=str(root)))
    print(f'Language2Drive preview: http://{args.bind}:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()

#!/usr/bin/env python3
"""Install the pinned replay assets, with archive and per-file SHA-256 checks."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.request import Request, ProxyHandler, HTTPRedirectHandler, build_opener
from urllib.parse import quote, urlsplit
import zipfile

ROOT=Path(__file__).resolve().parents[1]
MIRROR='https://hf-mirror.com'
OFFICIAL='https://huggingface.co'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe_relative(name):
    path=PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or str(path)!=name:
        raise ValueError('Unsafe asset path: '+name)
    return Path(*path.parts)


def target_path(destination,name):
    relative=safe_relative(name)
    target=destination/relative
    for part in (target,)+tuple(target.parents):
        if part==destination.parent:break
        if part.is_symlink():raise ValueError('Symlink in asset destination: '+name)
    if destination.resolve() not in target.resolve().parents:
        raise ValueError('Asset path escapes destination')
    return target


def verify_installed(manifest,destination):
    missing=[]
    for item in manifest['files']:
        path=target_path(destination,item['path'])
        if not path.is_file() or path.stat().st_size!=item['size_bytes'] or sha256(path)!=item['sha256']:
            missing.append(item['path'])
    return missing


def install_archive(archive,manifest,destination):
    archive=Path(archive);destination=Path(destination).resolve()
    if archive.stat().st_size!=manifest['size_bytes'] or sha256(archive)!=manifest['sha256']:
        raise ValueError('Archive size/SHA-256 mismatch')
    expected={item['path']:item for item in manifest['files']}
    if len(expected)!=len(manifest['files']):raise ValueError('Duplicate manifest path')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(expected) or set(z.namelist())!=set(expected):
            raise ValueError('Archive inventory does not match manifest')
        for item in z.infolist():
            target_path(destination,item.filename)
            mode=item.external_attr>>16
            if stat.S_ISLNK(mode) or item.is_dir():raise ValueError('Non-file archive entry')
            if item.file_size!=expected[item.filename]['size_bytes']:raise ValueError('Entry size mismatch')
        destination.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.l2d-assets-',dir=str(destination)) as temp:
            staged=Path(temp)
            for name,item in expected.items():
                path=staged/safe_relative(name);path.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as source,path.open('wb') as out:shutil.copyfileobj(source,out)
                if sha256(path)!=item['sha256']:raise ValueError('Entry SHA-256 mismatch: '+name)
                target=target_path(destination,name)
                if target.exists() and (not target.is_file() or sha256(target)!=item['sha256']):
                    raise ValueError('Existing different file; use a fresh destination: '+name)
            for name in expected:
                target=target_path(destination,name)
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    os.replace(str(staged/safe_relative(name)),str(target))
    if verify_installed(manifest,destination):raise ValueError('Installed asset verification failed')


class AssetRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme!='https':
            raise ValueError('Asset redirect must use HTTPS')
        redirected=super().redirect_request(req,fp,code,msg,headers,newurl)
        if redirected and urlsplit(req.full_url).hostname!=urlsplit(newurl).hostname:
            redirected.remove_header('Authorization')
        return redirected


def download(manifest,cache,endpoint,keep_proxy=False,token=None):
    if endpoint not in (MIRROR,OFFICIAL):raise ValueError('Endpoint must be hf-mirror.com or huggingface.co')
    cache.mkdir(parents=True,exist_ok=True)
    archive=cache/(manifest['sha256'][:16]+'-'+Path(manifest['archive']).name)
    if archive.is_file() and sha256(archive)==manifest['sha256']:return archive
    url='{}/datasets/{}/resolve/{}/{}'.format(endpoint,manifest['repo_id'],
        quote(manifest['revision'],safe=''),quote(manifest['archive'],safe='/'))
    headers={'User-Agent':'Language2Drive-nuScenes/1.0'}
    token=token or os.environ.get('HF_TOKEN')
    if manifest.get('private') and not token:
        raise ValueError('Private assets require HF_TOKEN or --prompt-token with dataset read access')
    if token:headers['Authorization']='Bearer '+token
    handlers=[AssetRedirectHandler()]
    if not keep_proxy:handlers.append(ProxyHandler({}))
    opener=build_opener(*handlers)
    temporary=archive.with_suffix('.part')
    try:
        with opener.open(Request(url,headers=headers),timeout=60) as source,temporary.open('wb') as out:
            count=0
            while True:
                chunk=source.read(1024*1024)
                if not chunk:break
                count+=len(chunk)
                if count>manifest['size_bytes']:raise ValueError('Download exceeds declared size')
                out.write(chunk)
        if count!=manifest['size_bytes'] or sha256(temporary)!=manifest['sha256']:
            raise ValueError('Downloaded asset hash mismatch')
        os.replace(str(temporary),str(archive))
    finally:
        if temporary.exists():temporary.unlink()
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'config/assets.json')
    p.add_argument('--destination',type=Path,default=ROOT)
    p.add_argument('--cache',type=Path,default=ROOT/'.cache/assets')
    p.add_argument('--archive',type=Path,help='Install an already downloaded archive')
    p.add_argument('--endpoint',default=MIRROR,choices=(MIRROR,OFFICIAL))
    p.add_argument('--keep-proxy',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--prompt-token',action='store_true',help='Read HF token without echo or persistence')
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    missing=verify_installed(manifest,args.destination)
    if not missing:
        print('All {} replay assets verified.'.format(len(manifest['files'])))
        return
    if args.verify_only:p.error('{} assets missing or changed'.format(len(missing)))
    try:
        token=getpass.getpass('HF read token (not stored): ') if args.prompt_token and not args.archive else None
        archive=args.archive or download(manifest,args.cache,args.endpoint,args.keep_proxy,token)
        install_archive(archive,manifest,args.destination)
    except Exception as error:
        p.exit(1,'Asset installation failed: {}\n'.format(str(error)))
    print('Installed and verified {} replay assets.'.format(len(manifest['files'])))


if __name__=='__main__':main()

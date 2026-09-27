"""Download or publish private benchmark assets, explicitly bypassing proxies."""
from __future__ import annotations
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shutil

PREFIX = 'langauge2drive.RealSim'  # Repository owner's requested spelling.
OFFICIAL_ENDPOINT = 'https://huggingface.co'
MIRROR_ENDPOINT = 'https://hf-mirror.com'


def validate_endpoint(endpoint, token):
    """Public mirror downloads are allowed; credentials stay on the official host."""
    if endpoint not in (OFFICIAL_ENDPOINT, MIRROR_ENDPOINT):
        raise ValueError('Unsupported endpoint; use the official host or hf-mirror.com')
    if token and endpoint == MIRROR_ENDPOINT:
        raise ValueError('Credentials must use the official endpoint; use --anonymous for public mirror downloads')


def direct_network(keep_proxy=False):
    if not keep_proxy:
        for key in list(os.environ):
            if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
                os.environ.pop(key)
        os.environ['NO_PROXY'] = os.environ['no_proxy'] = '*'
    os.environ.pop('HF_HUB_OFFLINE', None)


def configure_transport(keep_proxy=False):
    """Bound network waits for both supported generations of huggingface_hub."""
    import huggingface_hub as hub
    if hasattr(hub, 'configure_http_backend'):
        import requests

        class DirectSession(requests.Session):
            def __init__(self):
                super().__init__()
                self.trust_env = keep_proxy

            def request(self, method, url, **kwargs):
                if kwargs.get('timeout') is None:
                    kwargs['timeout'] = (15, 300)
                return super().request(method, url, **kwargs)

        hub.configure_http_backend(backend_factory=DirectSession)
    else:
        import httpx
        hub.set_client_factory(lambda: httpx.Client(
            trust_env=keep_proxy, follow_redirects=True,
            timeout=httpx.Timeout(300, connect=15)))


def ensure_private_repo(api, requested_repo=None):
    owner = api.whoami()['name']
    repo = requested_repo or f'{owner}/Language2Drive'
    if repo != f'{owner}/Language2Drive':
        raise ValueError('Uploads are restricted to the authenticated owner/Language2Drive dataset')
    api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
    api.update_repo_settings(repo, repo_type='dataset', private=True)
    if api.repo_info(repo, repo_type='dataset').private is not True:
        raise RuntimeError('Repository privacy was not verified; refusing upload')
    return repo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('whoami', 'init', 'upload', 'download', 'scenes', 'model'))
    p.add_argument('--repo', help='owner/Language2Drive; dataset repository')
    p.add_argument('--directory', type=Path, default=Path('assets'))
    p.add_argument('--bundle', type=Path)
    p.add_argument('--model', choices=('alpamayo2', 'alpamayo15'), default='alpamayo2')
    p.add_argument('--endpoint', default='https://huggingface.co')
    p.add_argument('--keep-proxy', action='store_true',
                   help='Explicitly use configured proxies; default is direct connections')
    p.add_argument('--revision', default='main')
    auth = p.add_mutually_exclusive_group()
    auth.add_argument('--token-file', type=Path)
    auth.add_argument('--prompt-token', action='store_true', help='Read token without echo; do not persist it')
    auth.add_argument('--anonymous', action='store_true', help='Do not use any token, including cached credentials')
    a = p.parse_args()
    direct_network(a.keep_proxy)
    # Import after network configuration: the hub caches environment defaults.
    from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download
    configure_transport(a.keep_proxy)
    token = (None if a.anonymous else getpass.getpass('HF token (hidden): ').strip() if a.prompt_token else
             a.token_file.read_text().strip() if a.token_file else get_token())
    try:
        validate_endpoint(a.endpoint, token)
    except ValueError as error:
        p.error(str(error))
    api = HfApi(endpoint=a.endpoint, token=token or False)
    if a.action == 'whoami':
        if not token:
            p.error('No HF_TOKEN/login found; use hf auth login or --token-file')
        print(json.dumps({'account': api.whoami()['name']}))
        return
    if a.action == 'init':
        if not token:
            p.error('A write-scoped token is required')
        repo = ensure_private_repo(api, a.repo)
        print(json.dumps(dict(repo=repo, private=True, prefix=PREFIX, files_uploaded=False)))
        return
    if a.action == 'upload':
        if not token:
            p.error('A write-scoped HF_TOKEN or --token-file is required')
        if not (a.directory / 'manifest.json').is_file():
            p.error('Upload directory must be a prepared release payload with manifest.json')
        # Small-file hygiene check; no large asset/checkpoint hashing pass.
        for path in a.directory.rglob('*'):
            if path.is_symlink():
                p.error(f'Symlink in upload payload: {path.name}')
            if path.is_file() and path.suffix in ('.json', '.yaml', '.md', '.txt', '.py', '.sh'):
                if re.search(r'hf_[A-Za-z0-9]{20,}|/(?:NEW_EDS|home|root)/|10\.0\.0\.', path.read_text()):
                    p.error(f'Private machine reference or credential in {path.name}')
        repo = ensure_private_repo(api, a.repo)
        result = api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=a.directory,
            path_in_repo=PREFIX, commit_message='Add portable RealSim benchmark and runtime artifacts')
        print(json.dumps(dict(repo=repo, private=True, prefix=PREFIX, commit=result.oid)))
    elif a.action == 'download':
        if not a.repo:
            p.error('--repo owner/Language2Drive is required')
        snapshot_download(repo_id=a.repo, repo_type='dataset', revision=a.revision,
            allow_patterns=[PREFIX + '/*'], local_dir=a.directory, endpoint=a.endpoint, token=token or False)
    else:
        if not a.bundle:
            p.error('--bundle is required')
        manifest = json.loads((a.bundle / 'assets.json').read_text())
        if a.action == 'model':
            snapshot_download(repo_id=manifest['models'][a.model], revision=a.revision,
                local_dir=a.directory, endpoint=a.endpoint, token=token or False)
        else:
            d = manifest['dataset']
            for name in d['files']:
                downloaded = hf_hub_download(repo_id=d['repo_id'], repo_type=d['repo_type'],
                    revision=d['revision'], filename=name, local_dir=a.directory / 'upstream',
                    endpoint=a.endpoint, token=token or False)
                target = a.directory / Path(name).name
                if target.exists():
                    if target.stat().st_size != Path(downloaded).stat().st_size:
                        p.error(f'Existing asset size differs: {target.name}')
                    continue
                # Hardlink if same filesystem; otherwise copy. No duplicate download.
                try:
                    target.hardlink_to(downloaded)
                except OSError:
                    shutil.copyfile(downloaded, target)


if __name__ == '__main__':
    main()

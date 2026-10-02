"""Create the pinned GPU-runtime Python environment without uploading a venv."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PIN = json.loads((ROOT / 'config/alpasim_realsim.json').read_text())['alpasim_commit']
UPSTREAM = 'https://github.com/NVlabs/alpasim.git'
UV_VERSION = '0.11.13'
RUST_VERSION = '1.90.0'
MARKER = '.l2d-managed-environment.json'


def make_environment(work, keep_proxy=False):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('PIP_', 'UV_')) or key in ('PYTHONPATH', 'VIRTUAL_ENV',
                'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN'):
            env.pop(key)
        if not keep_proxy and key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(key)
    if not keep_proxy:
        env['no_proxy'] = env['NO_PROXY'] = '*'
    env.update(PIP_CONFIG_FILE=os.devnull, PYTHONDONTWRITEBYTECODE='1',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', UV_PROJECT_ENVIRONMENT=str(work / 'venv'),
               UV_PYTHON_INSTALL_DIR=str(work / 'python'), UV_LINK_MODE='copy',
               UV_HTTP_TIMEOUT='120', HF_HUB_OFFLINE='1', GIT_LFS_SKIP_SMUDGE='1',
               MPLCONFIGDIR=str(work / 'matplotlib-cache'))
    return env


def validate_workdir(work):
    if work in (ROOT, ROOT.parent, Path('/'), Path.home()):
        raise ValueError('Refusing to use a repository root, home, or filesystem root as --work-dir')
    if work.exists():
        marker = work / MARKER
        if not marker.is_file():
            raise ValueError('Existing --work-dir has no ownership marker; choose a new empty path')
        if json.loads(marker.read_text()).get('alpasim_commit') != PIN:
            raise ValueError('Existing environment uses another AlpaSim revision; choose a new path')


def plan(args, work, uv):
    checkout = work / 'alpasim'
    python = work / 'venv/bin/python'
    # Frozen means neither resolving versions again nor rewriting upstream uv.lock.
    sync = [str(uv), 'sync', '--frozen', '--extra', 'all', '--python', args.python,
            '--default-index', args.index_url]
    if args.offline:
        sync.append('--offline')
    install = [str(uv), 'pip', 'install', '--python', str(python)]
    if args.offline:
        install += ['--offline']
    else:
        install += ['--index-url', args.index_url]
    steps = []
    if not args.offline and args.index_url.rstrip('/') != 'https://pypi.org/simple':
        exported = work / 'frozen-requirements.txt'
        registry = work / 'frozen-registry-requirements.txt'
        steps += [
            ('create-runtime-venv', [str(uv), 'venv', '--allow-existing', '--python', args.python, str(work / 'venv')], checkout),
            ('export-frozen-pins', [str(uv), 'export', '--frozen', '--all-extras', '--no-emit-workspace', '--no-header', '--no-annotate',
                                    '--format', 'requirements-txt', '--output-file', str(exported)], checkout),
            ('select-registry-pins', [sys.executable, str(ROOT / 'scripts/prepare_runtime_mirror.py'),
                                      '--source', str(exported), '--output', str(registry)], checkout),
            ('download-hash-bound-mirror', [str(uv), '--no-config', *install[1:], '--require-hashes', '--no-deps', '-r', str(registry)], checkout),
        ]
    return steps + [
        ('sync-frozen-upstream', sync, checkout),
        ('compile-protobuf', [str(python), '-c',
          'from scripts.compile_protos import compile_protos; compile_protos()'], checkout / 'src/grpc'),
        ('install-realsim-overlay', install + ['-e', str(ROOT) + '[adapter,test]',
          '-r', str(ROOT / 'config/runtime-compatibility.txt')], ROOT),
        ('runtime-python-check', [str(python), str(ROOT / 'scripts/doctor_runtime.py'),
          '--upstream', str(checkout)], ROOT),
        ('cpu-regressions', [str(python), '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
          str(ROOT / 'tests')], ROOT),
    ]


def install_rust(work, env, run, offline):
    if offline:
        raise ValueError('Cannot download Rust in offline mode; install cargo first')
    # Isolated toolchain: do not edit the login profile or a global Rust install.
    env.update(CARGO_HOME=str(work / 'cargo'), RUSTUP_HOME=str(work / 'rustup'),
               RUSTUP_DIST_SERVER='https://rsproxy.cn', RUSTUP_UPDATE_ROOT='https://rsproxy.cn/rustup')
    binary = work / 'rustup-init'
    url = 'https://rsproxy.cn/rustup/dist/x86_64-unknown-linux-gnu/rustup-init'
    # Build an opener with the already selected proxy policy.
    proxies = {scheme: env.get(scheme + '_proxy', env.get(scheme.upper() + '_PROXY', ''))
               for scheme in ('http', 'https')}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({k:v for k,v in proxies.items() if v}))
    print('Downloading isolated Rust bootstrap from rsproxy.cn', flush=True)
    with opener.open(url, timeout=120) as source, binary.open('wb') as target:
        shutil.copyfileobj(source, target)
    binary.chmod(0o700)
    run('install-rust', [str(binary), '-y', '--no-modify-path', '--profile', 'minimal',
                        '--default-toolchain', RUST_VERSION], work)
    env['PATH'] = str(work / 'cargo/bin') + os.pathsep + env.get('PATH', '')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', type=Path, default=ROOT / '.runtime-env')
    p.add_argument('--upstream-source', type=Path, help='Reuse a local upstream Git clone, without modifying it')
    p.add_argument('--python', default='3.12', help='Python 3.12 interpreter or uv-managed version request')
    p.add_argument('--uv', type=Path, help='Reuse uv 0.11.13; otherwise install it in an isolated bootstrap venv')
    p.add_argument('--index-url', default='https://pypi.tuna.tsinghua.edu.cn/simple')
    p.add_argument('--install-rust', action='store_true', help='Install isolated Rust via rsproxy.cn if cargo is missing')
    p.add_argument('--offline', action='store_true', help='Require cached dependencies and an existing/local upstream checkout')
    p.add_argument('--keep-proxy', action='store_true', help='Explicitly opt into configured proxies; disabled by default')
    p.add_argument('--plan', action='store_true', help='Print commands without creating files or accessing the network')
    p.add_argument('--skip-tests', action='store_true')
    args = p.parse_args(argv)
    if sys.version_info < (3, 10) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
        p.error('Bootstrap requires Python 3.10+ on Linux x86_64; runtime will use Python 3.12')
    work = args.work_dir.resolve()
    validate_workdir(work)
    uv = args.uv.resolve() if args.uv else work / 'bootstrap/bin/uv'
    steps = plan(args, work, uv)
    if args.skip_tests:
        steps = [x for x in steps if x[0] != 'cpu-regressions']
    if args.plan:
        print(json.dumps(dict(alpasim_commit=PIN, upstream=UPSTREAM, work_dir=str(work),
            uv_version=UV_VERSION, rust_version=RUST_VERSION, proxy_enabled=args.keep_proxy,
            environment_archive_uploaded=False, steps=[dict(name=n, command=c, cwd=str(d)) for n,c,d in steps]), indent=2))
        return 0
    for executable in ('git', 'cc'):
        if not shutil.which(executable):
            p.error(f'Missing system prerequisite: {executable}; see docs/ENVIRONMENT_SETUP_ZH.md')
    env = make_environment(work, args.keep_proxy)
    # Include a conventional Rust install if present, without changing login files.
    for cargo_bin in (work / 'cargo/bin', Path.home() / '.cargo/bin'):
        if (cargo_bin / 'cargo').is_file():
            env['PATH'] = str(cargo_bin) + os.pathsep + env.get('PATH', '')
            if cargo_bin == work / 'cargo/bin':
                env.update(CARGO_HOME=str(work / 'cargo'), RUSTUP_HOME=str(work / 'rustup'))
            break
    if not shutil.which('cargo', path=env['PATH']) and not args.install_rust:
        p.error('cargo is missing; rerun with --install-rust or install Rust first')
    work.mkdir(parents=True, exist_ok=True)
    if not (work / MARKER).exists():
        (work / MARKER).write_text(json.dumps(dict(alpasim_commit=PIN, schema_version=1)))
    (work / 'setup-result.json').write_text(json.dumps(dict(status='setup_in_progress',
        alpasim_commit=PIN, gpu_closed_loop_verified=False)) + '\n')
    logfile = work / f'setup-{time.time_ns()}.log'

    def run(name, command, cwd):
        print(f'[{name}] {shlex.join(command)}', flush=True)
        with logfile.open('a') as stream:
            stream.write(f'\n[{name}]\n')
            stream.flush()
            process = subprocess.Popen(command, cwd=cwd, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        if code:
            (work / 'setup-result.json').write_text(json.dumps(dict(status='setup_failed',
                failed_step=name, exit_code=code, gpu_closed_loop_verified=False)) + '\n')
            raise RuntimeError(f'Step {name} failed ({code}); log: {logfile}. '
                               'Fix the cause and rerun the same command; no success is recorded.')

    if not shutil.which('cargo', path=env['PATH']):
        install_rust(work, env, run, args.offline)
    if args.uv:
        version = subprocess.check_output([str(uv), '--version'], env=env, text=True).split()[1]
        if version != UV_VERSION:
            raise ValueError(f'Expected uv {UV_VERSION}, got {version}; omit --uv to bootstrap the pinned version')
    else:
        if not (work / 'bootstrap/pyvenv.cfg').exists():
            run('bootstrap-venv', [sys.executable, '-m', 'venv', str(work / 'bootstrap')], work)
        install = [str(work / 'bootstrap/bin/python'), '-m', 'pip', 'install', '--disable-pip-version-check']
        install += ['--no-index'] if args.offline else ['--index-url', args.index_url]
        run('bootstrap-uv', install + [f'uv=={UV_VERSION}'], work)
    checkout = work / 'alpasim'
    if not checkout.exists():
        if args.offline and not args.upstream_source:
            raise ValueError('Offline first install requires --upstream-source pointing to a local Git clone')
        origin = str(args.upstream_source.resolve()) if args.upstream_source else UPSTREAM
        command = ['git', 'clone', '--no-checkout', '--no-hardlinks']
        if not args.upstream_source:
            command.append('--filter=blob:none')
        run('clone-upstream', command + [origin, str(checkout)], work)
        run('checkout-upstream-pin', ['git', 'checkout', '--detach', PIN], checkout)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, env=env, text=True).strip()
    if head != PIN or subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=checkout, env=env).returncode:
        raise ValueError('Upstream checkout is modified or at the wrong pin; preserve it and use another --work-dir')
    # Upstream ignores uv.lock; ship the tested resolution alongside this release.
    lock_source = ROOT / 'config/alpasim-runtime.uv.lock'
    lock_target = checkout / 'uv.lock'
    if lock_target.exists() and lock_target.read_bytes() != lock_source.read_bytes():
        raise ValueError('Existing uv.lock differs from the release lock; use another --work-dir')
    if not lock_target.exists():
        shutil.copyfile(lock_source, lock_target)
    for name, command, cwd in steps:
        run(name, command, cwd)
    receipt = dict(status='runtime_python_ready', alpasim_commit=PIN, python=str(work / 'venv/bin/python'),
                   cpu_tests_run=not args.skip_tests, gpu_closed_loop_verified=False,
                   environment_archive_uploaded=False)
    (work / 'setup-result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'Environment setup stopped: {error}', file=sys.stderr)
        raise SystemExit(1)

"""Public language2drive command. Native track entry points remain unchanged."""
import argparse
import json
from pathlib import Path
import sys
from . import __version__
from .workspace import TRACKS, discover, initialize


def selection(value):
    names = list(TRACKS) if value == 'all' else value.split(',')
    if not names or len(set(names)) != len(names) or any(n not in TRACKS for n in names):
        raise argparse.ArgumentTypeError('Use all or comma-separated carla,realsim,nuscenes,navsim,wod')
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(prog='language2drive', description='One model adapter. Five instruction-following evaluation tracks.')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init', help='Extract verified backends from the wheel into a workspace')
    init.add_argument('--workspace', required=True)
    setup = sub.add_parser('setup', help='Install selected isolated environments and optional resources')
    setup.add_argument('--workspace')
    setup.add_argument('--tracks', type=selection, default=selection('all'))
    setup.add_argument('--network', choices=('auto', 'official', 'mirror'), default='auto')
    setup.add_argument('--runtime', action='store_true', help='Also provision CARLA/AlpaSim runtime dependencies')
    setup.add_argument('--download', action='store_true')
    setup.add_argument('--source-root')
    setup.add_argument('--package', help='Use this local wheel or checkout when installing evaluator environments')
    setup.add_argument('--dry-run', action='store_true')
    serve = sub.add_parser('serve', help='Run your adapter in its existing model environment')
    serve.add_argument('--adapter', required=True)
    serve.add_argument('--config', help='Model configuration JSON')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8090)
    run = sub.add_parser('evaluate', help='Infer, export and score selected tracks; save a coverage report')
    run.add_argument('--config', required=True)
    run.add_argument('--tracks', type=selection, default=selection('all'))
    run.add_argument('--output', default='outputs/my-model')
    run.add_argument('--dry-run', action='store_true')
    worker = sub.add_parser('_worker', help=argparse.SUPPRESS)
    worker.add_argument('--config', required=True)
    worker.add_argument('--track', required=True, choices=TRACKS)
    worker.add_argument('--output', required=True)
    demo = sub.add_parser('adapter-check', help='Validate one adapter against all five output contracts, without datasets')
    demo.add_argument('--adapter', required=True)
    demo.add_argument('--config')
    doctor = sub.add_parser('doctor', help='Check workspace and per-track interpreter availability')
    doctor.add_argument('--workspace')
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            result = {'workspace': str(initialize(args.workspace))}
        elif args.command == 'setup':
            from .setup import bootstrap
            result = bootstrap(args.workspace, args.tracks, args.network, args.runtime, args.download, args.source_root, args.dry_run, args.package)
        elif args.command == 'serve':
            from .service import serve
            serve(args.adapter, json.loads(Path(args.config).read_text()) if args.config else {}, args.host, args.port)
            return 0
        elif args.command in ('evaluate', '_worker'):
            from .pipeline import run, worker
            result = run(args.config, args.tracks, args.output, args.dry_run) if args.command == 'evaluate' else worker(args.config, args.track, args.output)
        elif args.command == 'adapter-check':
            from .adapter import Request, SPECS, load_adapter, predict
            config = json.loads(Path(args.config).read_text()) if args.config else {}
            result = {'mode': 'synthetic_interface_check', 'tracks': {}}
            for track, spec in SPECS.items():
                adapter = load_adapter(args.adapter, config)
                predict(adapter, Request(track, 'Maintain your current speed.', {'cameras': {}, 'speed_m_s': 2.0}, spec))
                result['tracks'][track] = 'valid'
                close = getattr(adapter, 'close', None)
                if close: close()
        else:
            root = discover(args.workspace)
            result = {'workspace': str(root), 'interpreters': {t: (root / '.l2d/envs' / t / 'bin/python').is_file() for t in TRACKS}, 'data_access': 'See docs/INSTALLATION.md; doctor does not assert dataset/runtime readiness'}
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == 'evaluate' and any(v.get('status') == 'failed' for v in result['tracks'].values()):
            return 1
        return 0
    except Exception as error:
        parser.exit(2, f'language2drive: {error}\n')


if __name__ == '__main__':
    sys.exit(main())

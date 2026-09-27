"""Explicit local post-training identity; never an official-release identity."""
import json
from pathlib import Path
from .io import file_hash, stable_hash

MODEL_ID = 'Language2Drive/SimLingo-Posttrained-epoch004'
MODEL_NAME = 'SimLingo-L2D-Posttrained'


def checkpoint_identity(path):
    path = Path(path).resolve()
    root = path.parents[2]
    receipt_path = root / 'POSTTRAINED_SOURCE.json'
    config = root / '.hydra/config.yaml'
    result = dict(model_id=MODEL_ID, path=str(path), complete=False,
                  official_release=False, snapshot_revision=None,
                  large_weight_hash_computed=False)
    try:
        source = json.loads(receipt_path.read_text())
        st = path.stat()
        actual = dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns)
        result.update(snapshot_revision=source['revision'],
            complete=(source['model_id'] == MODEL_ID and source['official_release'] is False
                      and source['epoch'] == 4 and source['global_step'] == 21660
                      and source['checkpoint'] == actual and st.st_size > 0
                      and source['large_weight_hash_computed'] is False
                      and (root / 'code/simlingo_training/models/driving.py').is_file()),
            checkpoint_metadata=actual, config_sha256=file_hash(config),
            source_receipt_sha256=file_hash(receipt_path),
            parent_model_id=source['parent_model_id'], epoch=source['epoch'],
            global_step=source['global_step'])
    except (OSError, ValueError, KeyError, TypeError):
        result['complete'] = False
    result['identity_binding'] = stable_hash(result)
    return result

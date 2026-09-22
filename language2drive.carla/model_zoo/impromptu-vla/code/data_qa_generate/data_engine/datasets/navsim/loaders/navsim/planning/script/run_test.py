from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from data_engine.datasets.navsim.loaders.navsim.agents.abstract_agent import AbstractAgent
from data_engine.datasets.navsim.loaders.navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from data_engine.datasets.navsim.loaders.navsim.common.dataclasses import SensorConfig
from data_engine.datasets.navsim.loaders.navsim.evaluate.pdm_score import pdm_score
from data_engine.datasets.navsim.loaders.navsim.planning.script.builders.worker_pool_builder import build_worker
from data_engine.datasets.navsim.loaders.navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from data_engine.datasets.navsim.loaders.navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from data_engine.datasets.navsim.loaders.navsim.planning.metric_caching.metric_cache import MetricCache
from data_engine.datasets.navsim.loaders.navsim.agents.diffusiondrive_dpo.transfuser_agent import DiffusionDPOAgent
from data_engine.datasets.navsim.loaders.navsim.dpo.PreferenceDataset import DPOCacheOnlyDataset, Dataset
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training_dpo"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    logger.info("Building Agent")
    agent: DiffusionDPOAgent = instantiate(cfg.agent)
    train_data = DPOCacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            preference_path=cfg.preference_data_path,
            log_names=cfg.train_logs[:500],
        )
    
    import numpy as np
    idx_list = np.random.choice(len(train_data), 200, replace=False).tolist()
    batch_list = []
    for idx in idx_list:
        feature, target = train_data[idx]
        for k, v in feature.items():
            feature[k] = v.unsqueeze(0)
        for k, v in target.items():
            target[k] = v.unsqueeze(0)
        batch_list.append((feature, target))
    
    agent.initialize()
    cnt_ref = np.zeros((20, ))
    cnt_post = np.zeros((20, ))
    from tqdm import tqdm
    for batch_idx, batch in tqdm(enumerate(batch_list)):
        feature, target = batch
        forward_result = agent.forward(feature)
        post_traj = forward_result["post_result"]["trajectory"]
        ref_traj = forward_result["ref_result"]["trajectory"]
        post_score = forward_result["post_result"]["score"]
        ref_score = forward_result["ref_result"]["score"]
        
        all_mse = ((post_traj - ref_traj) ** 2).mean().item()
        post_traj = post_traj.cpu().detach().numpy()[..., :2]
        ref_traj = ref_traj.cpu().detach().numpy()[..., :2]
        import torch
        mode_idx_post = torch.argmax(post_score[0, :]).item()
        mode_idx_ref = torch.argmax(ref_score[0, :]).item()
        # text the mse
        traj_mse = ((post_traj[0, mode_idx_post, ...] - ref_traj[0, mode_idx_ref, ...]) ** 2).mean().item()
        import json
        with open("debug/scores.txt", "a") as f:
            metadata = {
                "post_score": post_score[0, :].cpu().detach().numpy().tolist(),
                "ref_score": ref_score[0, :].cpu().detach().numpy().tolist(),
                "traj_mse": traj_mse,
                "mse": all_mse,
            }
            f.write(json.dumps(metadata) + "\n")
        cnt_ref[torch.argmax(ref_score[0, :]).item()] += 1
        cnt_post[torch.argmax(post_score[0, :]).item()] += 1
    
    with open("debug/scores.txt", "a") as f:
        f.write(f"ref: {cnt_ref/cnt_ref.sum()}\n")
        f.write(f"post: {cnt_post/cnt_post.sum()}\n")
    
if __name__ == "__main__":
    main()

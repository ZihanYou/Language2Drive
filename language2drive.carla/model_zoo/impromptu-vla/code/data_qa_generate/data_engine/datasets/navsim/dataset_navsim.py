from data_engine.datasets.navsim.loaders.pipelines import *
from data_engine.datasets.navsim.loaders.navsim.planning.training.agent_lightning_module import AgentLightningModule
from data_engine.datasets.navsim.loaders.navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from data_engine.datasets.navsim.loaders.navsim.common.dataloader import SceneLoader
from data_engine.datasets.navsim.loaders.navsim.common.dataclasses import Scene, SceneFilter, SceneMetadata, SensorConfig
from data_engine.datasets.navsim.loaders.navsim.agents.abstract_agent import AbstractAgent
from data_engine.common_misc.external_helpers.openai_query import construct_external_query
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import instantiate
from PIL import Image
import hydra
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Tuple
import os
import ray
import json
import math

from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent.parent.parent

# append current directory to path
import sys

import torch
sys.path.append(os.path.dirname(__file__))


logger = logging.getLogger(__name__)

PIPELINES = {
    "metadata": PromptNavsimMetadata,
    "ego_status": PromptNavsimEgoStatus,
    "planning": PromptNavsimPlanning,
    'meta_planning': PromptNavsimMetaPlanning,
    "road_agent_analysis": PromptNavsimRoadAgentAnalysis,
    "scene_description": PromptNavsimSceneDescription,
}

v0_pipelines = [
    {"type": "metadata", "use_image": "3v"},
    {"type": "ego_status", "mode": "x-y"},
    {"type": "planning", "mode": "x-y"},
    # ! the constructed queries of this stage are dependent on the mode of previous planning pipeline
    {"type": "meta_planning"},
    {"type": "road_agent_analysis"},
    {"type": "scene_description"},
]
v0_container_out_key_comb = ["scene_description",
                             'road_agent_analysis', 'meta_planning', 'planning']
# v0_container_out_key_comb = ['meta_planning', 'planning']

va_pipelines = [
    {"type": "metadata", "use_image": "3v"},
    {"type": "ego_status", "mode": "dist-dtheta"},
    # {"type": "scene_description"},
    # {"type": "road_agent_analysis_text"},
    {"type": "planning", "mode": "dist-dtheta"},
    # {"type": "meta_planning"},  #! the constructed queries of this stage are dependent on the mode of previous planning pipeline
]
va_container_out_key_comb = ['planning']


class VLMNavsim(torch.utils.data.Dataset):
    # Directly mapping the input images to the output images
    def __init__(self, mode="train", length=None, pipelines=[], container_out_key_comb=[]):
        cfg = OmegaConf.load(
            "data_qa_generate/data_engine/datasets/navsim/loaders/navsim/planning/script/config/training/default_training.yaml")
        navsim = self.build_datasets(cfg, mode)

        self.navsim = navsim

        self.length = len(self.navsim)
        if length is not None:
            self.length = min(self.length, length)

        self.pipelines = []
        self.external_helper = construct_external_query(
            "Qwen/Qwen2.5-VL-72B-Instruct")
        for pipeline in pipelines:
            if pipeline["type"] in PIPELINES:
                self.pipelines.append(PIPELINES[pipeline["type"]](
                    navsim=self.navsim, **pipeline, external_helper=self.external_helper))
            else:
                raise ValueError("Pipeline {} not found".format(pipeline))
        self.container_out_key_comb = container_out_key_comb

    def __len__(self):
        return self.length

    def build_datasets(self, cfg: DictConfig, mode="train") -> Dataset:
        """
        Builds training and validation datasets from omega config
        :param cfg: omegaconf dictionary
        :param agent: interface of agents in NAVSIM
        :return: tuple for training and validation dataset
        """
        sensor_config = SensorConfig.build_all_sensors(include=[3])

        navtrain_filter_cfg = OmegaConf.load(
            "data_qa_generate/data_engine/datasets/navsim/loaders/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml")
        navtest_cfg = OmegaConf.load(
            "data_qa_generate/data_engine/datasets/navsim/loaders/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")

        split_logs = OmegaConf.load(
            "data_qa_generate/data_engine/datasets/navsim/loaders/navsim/planning/script/config/training/default_train_val_test_log_split.yaml")
        cfg.train_logs, cfg.val_logs, cfg.test_logs = split_logs.train_logs, split_logs.val_logs, split_logs.test_logs

        if mode == "train":

            trainval_scene_filter: SceneFilter = instantiate(
                navtrain_filter_cfg)
            if trainval_scene_filter.log_names is not None:
                trainval_scene_filter.log_names = [
                    log_name for log_name in trainval_scene_filter.log_names if log_name in cfg.train_logs or log_name in cfg.val_logs
                ]
            else:
                trainval_scene_filter.log_names = cfg.train_logs + cfg.val_logs

            data_path_trainval = Path(
                f"{project_root}/data_qa_generate/data_engine/data_storage/external_datasets/navsim/navsim_logs/trainval")
            sensor_blobs_path_trainval = Path(
                f"{project_root}/data_qa_generate/data_engine/data_storage/external_datasets/navsim/sensor_blobs/trainval")

            trainval_scene_loader = SceneLoader(
                sensor_blobs_path=sensor_blobs_path_trainval,
                data_path=data_path_trainval,
                scene_filter=trainval_scene_filter,
                sensor_config=sensor_config,
            )

            trainval_data = Dataset(
                scene_loader=trainval_scene_loader,
                feature_builders=[],
                target_builders=[],
                cache_path=None,
                force_cache_computation=False,
            )

            return trainval_data

        elif mode == "test":

            test_scene_filter: SceneFilter = instantiate(navtest_cfg)
            if test_scene_filter.log_names is not None:
                test_scene_filter.log_names = [
                    log_name for log_name in test_scene_filter.log_names if log_name in cfg.test_logs]
            else:
                test_scene_filter.log_names = cfg.test_logs

            data_path_test = Path(
                f"{project_root}/data_qa_generate/data_engine/data_storage/external_datasets/navsim/navsim_logs/test")
            sensor_blobs_path_test = Path(
                f"{project_root}/data_qa_generate/data_engine/data_storage/external_datasets/navsim/sensor_blobs/test")

            test_scene_loader = SceneLoader(
                sensor_blobs_path=sensor_blobs_path_test,
                data_path=data_path_test,
                scene_filter=test_scene_filter,
                sensor_config=sensor_config
            )

            test_data = Dataset(
                scene_loader=test_scene_loader,
                feature_builders=[],
                target_builders=[],
                cache_path=None,
                force_cache_computation=False,
            )

            return test_data

        else:
            raise ValueError("Mode {} not found".format(mode))

    def cache_data(self, cache_filename):
        assert cache_filename.endswith(".json")
        # create cache file if not exists
        cache_filename = os.path.join("data", "navsim", cache_filename)

        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)

        all_iters = list(range(len(self)))
        # Parallel processing even slower than serial processing. do not know why. by c7w
        # num_cpus = min(math.ceil(os.cpu_count() * 0.8), 16)
        # print("Using {} CPUs for caching".format(num_cpus))
        # all_jsons = p_map(self.__getitem__, all_iters, num_cpus=num_cpus, desc="Caching data")
        all_jsons = []
        for i in tqdm(all_iters, desc="Caching data"):
            all_jsons.append(self.__getitem__(i))

        # DUMP a json file!
        with open(cache_filename, "w") as f:
            f.write("[\n")
            all_len = len(all_jsons)
            for idx, json_obj in enumerate(all_jsons):
                f.write(json.dumps(json_obj))
                if idx != all_len - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("]")

        # cleanup all pipelines
        for pipeline in self.pipelines:
            if hasattr(pipeline, "cleanup"):
                pipeline.cleanup()

    def evaluate(self, jsonl_file):
        # load jsonl file to a list of dict
        predicted_data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                predicted_data.append(json.loads(line))

        assert len(predicted_data) == len(
            self), "Length of predictions and dataset do not match: {} vs {}".format(len(predicted_data), len(self))
        # evaluate every single prediction
        # import pdb;pdb.set_trace()
        for idx, pred in tqdm(enumerate(predicted_data)):
            # import pdb;pdb.set_trace()

            container_in = self.get_container_in(idx)
            container_in["idx"] = idx
            container_out = self.__getitem__(idx)
            import pdb
            pdb.set_trace()

            # evaluate the prediction
            for pipeline in self.pipelines:
                pipeline.evaluation_update(pred, container_out, container_in)
        # import pdb;pdb.set_trace()

        results = {}
        for pipeline in self.pipelines:
            pipeline.evaluation_compute(results)
        print(results)
        return results

    def viz_all_results(self, jsonl_file, interval):
        # load jsonl file to a list of dict
        predicted_data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                predicted_data.append(json.loads(line))

        assert len(predicted_data) == len(
            self), "Length of predictions and dataset do not match: {} vs {}".format(len(predicted_data), len(self))
        # evaluate every single prediction
        # import pdb;pdb.set_trace()

        viz_path = jsonl_file.replace(".json", "")
        os.makedirs(viz_path, exist_ok=True)
        print(f"Saving visualization to {viz_path}")

        for idx, pred in tqdm(enumerate(predicted_data)):
            # import pdb;pdb.set_trace()
            if idx % interval != 0:
                continue

            container_in = self.get_container_in(idx)
            container_in["idx"] = idx
            container_out = self.__getitem__(idx)
            # import pdb;pdb.set_trace()
            save_name = f"{idx:06d}.html"
            id = container_out["id"]
            images = container_out['images']
            query = container_out['messages'][0]['content']
            gt = container_out['messages'][1]['content']
            prediction = pred['predict']

            def escape_html(text):
                return text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            from PIL import Image
            image_paths = images
            target_size = (576, 384)
            output_dir_imgs = os.path.join(viz_path, f"images_{idx}")
            output_img_names = []

            os.makedirs(output_dir_imgs, exist_ok=True)

            # Process each image
            for image_path in image_paths:
                # Extract the camera status from the path
                camera_status = image_path.split('/')[-2]

                # Open the image
                with Image.open(image_path) as img:
                    # Resize the image
                    img_resized = img.resize(
                        target_size, Image.Resampling.NEAREST)

                    # Create the output file name
                    output_file_name = f"{camera_status}.jpg"
                    output_file_path = os.path.join(
                        output_dir_imgs, output_file_name)
                    output_img_names.append(output_file_name)

                    # Save the resized image
                    img_resized.save(output_file_path, "JPEG")
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>All In One</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                    }}
                    h2 {{
                        color: #333;
                    }}
                    p {{
                        font-family: monospace;
                    }}
                    .section {{
                        margin-bottom: 20px;
                    }}
                    .images {{
                        display: flex;
                        flex-wrap: wrap;
                    }}
                    .images img {{
                        margin: 5px;
                        max-width: 200px;
                    }}
                </style>
            </head>
            <body>
                <div class="section">
                    <h2>Query</h2>
                    <p>{escape_html(query)}</p>
                </div>
                <div class="section">
                    <h2>Ground Truth</h2>
                    <p>{escape_html(gt)}</p>
                </div>
                <div class="section">
                    <h2>Prediction</h2>
                    <p>{escape_html(prediction)}</p>
                </div>
                <div class="section">
                    <h2>Images</h2>
                    <div class="images">
            """
            # Add the images to the HTML content
            for image_file in output_img_names:
                image_path = os.path.join(output_dir_imgs, image_file)
                html_content += f'            <img src="images_{idx}/{image_file}" alt="{image_file}">\n'

            # Close the HTML tags
            html_content += """</div>
                </div>
            </body>
            </html>"""

        # import pdb;pdb.set_trace()
            with open(os.path.join(viz_path, save_name), "w") as f:
                f.write(html_content)

    def get_container_in(self, idx):
        container_in = {}  # first, aggregate container_in
        this_token = self.navsim._scene_loader.tokens[idx]
        scene_dict_list = self.navsim._scene_loader.scene_frames_dicts[this_token]
        sensor_blobs_path = self.navsim._scene_loader._sensor_blobs_path
        num_history_frames, num_future_frames = self.navsim._scene_loader._scene_filter.num_history_frames, self.navsim._scene_loader._scene_filter.num_future_frames
        sensor_config = self.navsim._scene_loader._sensor_config

        this_scene_metadata = SceneMetadata(
            log_name=scene_dict_list[num_history_frames - 1]["log_name"],
            scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
            map_name=scene_dict_list[num_history_frames - 1]["map_location"],
            initial_token=scene_dict_list[num_history_frames - 1]["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )

        container_in["token"] = this_token
        container_in["scene_metadata"] = this_scene_metadata

        frame_data_dict = []
        for frame_idx in range(len(scene_dict_list)):
            global_ego_status = Scene._build_ego_status(
                scene_dict_list[frame_idx])
            annotations = Scene._build_annotations(scene_dict_list[frame_idx])

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            if len(sensor_names) > 0:
                sensor_names = sensor_names[:-1]  # DROP lidar_pc

            this_frame_cameras = {}
            camera_dict = scene_dict_list[frame_idx]["cams"]
            data_dict = {}
            for camera_name in camera_dict.keys():
                camera_identifier = camera_name.lower()
                if camera_identifier in sensor_names:
                    image_path = sensor_blobs_path / \
                        camera_dict[camera_name]["data_path"]
                    data_dict[camera_identifier] = {
                        "image_path": str(image_path),
                        "sensor2lidar_rotation": camera_dict[camera_name]["sensor2lidar_rotation"],
                        "sensor2lidar_translation": camera_dict[camera_name]["sensor2lidar_translation"],
                        "intrinsics": camera_dict[camera_name]["cam_intrinsic"],
                        "distortion": camera_dict[camera_name]["distortion"],
                    }

                else:
                    data_dict[camera_identifier] = {}  # empty camera
            this_frame_cameras = data_dict  # rename it

            frame_data_dict.append({
                "token": scene_dict_list[frame_idx]["token"],
                "timestamp": scene_dict_list[frame_idx]["timestamp"],
                "ego_status": global_ego_status,
                "annotations": annotations,
                "cameras": this_frame_cameras
            })
        container_in["frame_data"] = frame_data_dict

        container_in["ego_status"] = []
        for frame in frame_data_dict:
            container_in["ego_status"].append(frame["ego_status"])
        return container_in

    def cache_queries(self, query_filename, pipeline, max_len=8000):
        assert query_filename.endswith(".json")
        # cache all queries
        assert pipeline in self.pipelines, "Pipeline not in the list of pipelines"

        all_queries = []
        for idx in tqdm(range(len(self)), desc="Caching queries"):
            batch = self.get_container_in(idx)
            container_out = {}
            # container_out = self.prompt_metadata(container_out, batch)
            for self_pipeline in self.pipelines:
                if self_pipeline != pipeline:
                    container_out = self_pipeline(container_out, batch)
                else:
                    break  # for caching

                if self_pipeline.cache_response_filename is not None and 'navsim_meta_planning.json' in self_pipeline.cache_response_filename:
                    break  # stop it!

            query = pipeline.cache_construct_query(container_out, batch)
            all_queries.append(query)

        # divide the queries into chunks, and save them
        if len(all_queries) > max_len:
            num_chunks = math.ceil(len(all_queries) / max_len)
            chunk_size = math.ceil(len(all_queries) / num_chunks)
            for i in range(num_chunks):
                chunk_queries = all_queries[i *
                                            chunk_size: (i + 1) * chunk_size]
                chunk_filename = query_filename.replace(
                    ".json", "_{}.json".format(i))
                with open(chunk_filename, "w") as f:
                    json.dump(chunk_queries, f)
        else:
            with open(query_filename, "w") as f:
                json.dump(all_queries, f)

    def cache_responses(self, response_filename, pipeline, max_len=8000):
        assert pipeline in self.pipelines, "Pipeline not in the list of pipelines"
        assert response_filename.endswith(".jsonl")
        all_responses = []

        if len(self) > max_len:
            num_chunks = math.ceil(len(self) / max_len)
            for i in range(num_chunks):
                chunk_filename = response_filename.replace(
                    ".jsonl", "_{}.jsonl".format(i))
                with open(chunk_filename, "r") as f:
                    for line in f:
                        all_responses.append(json.loads(line))
        else:
            with open(response_filename, "r") as f:
                for line in f:
                    all_responses.append(json.loads(line))
        assert len(all_responses) == len(
            self), "Length of responses and dataset do not match: {} vs {}".format(len(all_responses), len(self))
        for idx, response in enumerate(tqdm(all_responses)):
            batch = self.get_container_in(idx)
            container_out = {}
            for self_pipeline in self.pipelines:
                if self_pipeline != pipeline:
                    container_out = self_pipeline(container_out, batch)
                else:
                    break
            pipeline.cache_from_response(response, container_out, batch)
        pipeline.cleanup()
        return len(all_responses)

    def __getitem__(self, idx):
        batch = self.get_container_in(idx)
        batch["idx"] = idx
        # Let's forward the pipelines with container_in!

        # pipeline
        container_out = {}
        for pipeline in self.pipelines:
            container_out = pipeline(container_out, batch)
        # pipeline
        for key in self.container_out_key_comb:
            assert key in container_out["buffer_container"], "Key {} not found in container_out.buffer_container".format(
                key)
            container_out["messages"][1]["content"] += container_out["buffer_container"][key]

        container_out.pop("buffer_container")

        return container_out


def generate_batch_dataset():
    experiments = [
        {"id": "camera_ego_planning", "pipelines": [
            {"type": "metadata", "use_image": "3v"},
            {"type": "ego_status", "mode": "x-y"},
            {"type": "planning", "mode": "x-y"}
        ], "container_out_key_comb": ["planning"]},

        {"id": "camera_ego_metaplanning_planning", "pipelines": [
            {"type": "metadata", "use_image": "3v"},
            {"type": "ego_status", "mode": "x-y"},
            {"type": "meta_planning"},
            {"type": "planning", "mode": "x-y"}
        ], "container_out_key_comb": ["meta_planning", "planning"]},

        {"id": "camera_ego_road_metaplanning_planning", "pipelines": [
            {"type": "metadata", "use_image": "3v"},
            {"type": "ego_status", "mode": "x-y"},
            {"type": "road_agent_analysis"},
            {"type": "meta_planning"},
            {"type": "planning", "mode": "x-y"}
        ], "container_out_key_comb": ["road_agent_analysis", "meta_planning", "planning"]},

        {"id": "camera_ego_scene_road_metaplanning_planning", "pipelines": [
            {"type": "metadata", "use_image": "3v"},
            {"type": "ego_status", "mode": "x-y"},
            {"type": "scene_description"},
            {"type": "road_agent_analysis"},
            {"type": "meta_planning"},
            {"type": "planning", "mode": "x-y"}
        ], "container_out_key_comb": ["scene_description", "road_agent_analysis", "meta_planning", "planning"]}
    ]

    for experiment in experiments:
        v0_pipelines = experiment["pipelines"]
        v0_container_out_key_comb = experiment["container_out_key_comb"]

        dataset = VLMNavsim(mode="test", pipelines=v0_pipelines,
                            container_out_key_comb=v0_container_out_key_comb)
        dataset.cache_data(f"navsim_test_{experiment['id']}.json")

        dataset = VLMNavsim(mode="train", pipelines=v0_pipelines,
                            container_out_key_comb=v0_container_out_key_comb)
        dataset.cache_data(f"navsim_train_{experiment['id']}.json")


if __name__ == "__main__":
    generate_batch_dataset()

    # Load the config
    # dataset = VLMNavsim(mode="test", pipelines=v0_pipelines, container_out_key_comb=v0_container_out_key_comb)
    # batch = dataset[240]

    # dataset.cache_data("navsim_test_v00.json")
    # dataset.cache_queries("data/navsim/navsim_test_mp_queries.json", dataset.pipelines[3])
    # dataset.cache_responses("saves_20250215/Qwen2_5-VL-72B-Instruct/freeze/inference/navsim_test_mp_queries.jsonl", dataset.pipelines[3])
    # dataset.cache_queries("data/navsim/navsim_test_ra_queries.json", dataset.pipelines[-2])
    # dataset.cache_responses("saves_20250215/Qwen2_5-VL-72B-Instruct/freeze/inference/navsim_test_ra_queries.jsonl", dataset.pipelines[-2])
    # dataset.cache_queries("data/navsim/navsim_test_sd_queries.json", dataset.pipelines[-1])
    # dataset.cache_responses("saves_20250215/Qwen2_5-VL-72B-Instruct/freeze/inference/navsim_test_sd_queries.jsonl", dataset.pipelines[-1])
    # dataset.cache_data("navsim_test_full.json")
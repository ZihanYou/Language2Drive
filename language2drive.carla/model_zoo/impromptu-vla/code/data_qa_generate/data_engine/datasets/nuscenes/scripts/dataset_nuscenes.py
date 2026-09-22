import os
import time
import json
import torch
import math
import pickle
from tqdm import tqdm
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from p_tqdm import p_map
from torch.utils.data import Dataset
from data_engine.datasets.nuscenes.loaders.mmdet3d_plugin.nuscenes_3d import build_nuscenes_3d
from data_engine.common_misc.external_helpers.openai_query import construct_external_query
from data_engine.datasets.nuscenes.loaders.pipelines import *

PIPELINES = {
    "metadata": PromptNuScenesMetadata,
    "ego_status": PromptNuScenesEgoStatus,
    "scene_description": PromptNuScenesSceneDescription,  # scene_desc
    "road_agent_analysis": PromptNuScenesRoadAgentAnalysis,  # road_agent_analysis
    "meta_planning": PromptNuScenesMetaPlanning,  # meta_planning
    "planning": PromptNuScenesPlanning  # planning
}

v0_pipelines = [  # keep the pipeline order as is. can comment out some of them
    # assert self.use_image in ["6v", "1v", "t-6v", "t-1v", "none"]
    {"type": "metadata", "use_image": "1v"},
    {"type": "ego_status", "mode": "x-y"},
    {"type": "planning", "mode": "x-y"},
    {"type": "meta_planning", "use_query_gt_status": True},
    {"type": "road_agent_analysis"},
    {"type": "scene_description"},
]
v0_container_out_key_comb = ["scene_description",
                             "road_agent_analysis", "meta_planning", "planning"]
# v0_container_out_key_comb = ["road_agent_analysis", "meta_planning", "planning"]
# v0_container_out_key_comb = ["meta_planning", "planning"]


class VLMNuScenes(Dataset):
    # Directly mapping the input images to the output images
    def __init__(self, mode="train", length=None, pipelines=[], container_out_key_comb=[], skip_nuscenes_build=False):

        # cache it
        tmp_nuscenes_path = f'../nuscenes_mmcv_{mode}.pkl'
        tmp_nuscenes_3d_path = f'../nuscenes_{mode}.pkl'

        if not os.path.exists(tmp_nuscenes_path):
            self.nuscenes = build_nuscenes_3d(mode=mode)

            if not skip_nuscenes_build:
                self.nuscenes_3d = NuScenes(
                    version=self.nuscenes.version, dataroot=self.nuscenes.data_root, verbose=True)
            else:
                self.nuscenes_3d = None

            with open(tmp_nuscenes_path, 'wb') as f:
                pickle.dump(self.nuscenes, f)
            with open(tmp_nuscenes_3d_path, 'wb') as f:
                pickle.dump(self.nuscenes_3d, f)

        else:
            print("Starting to load cached nuscenes dataset :>")
            s_time = time.time()
            # To load the object back from the file
            with open(tmp_nuscenes_path, 'rb') as f:
                self.nuscenes = pickle.load(f)

            # To load the object back from the file
            with open(tmp_nuscenes_3d_path, 'rb') as f:
                self.nuscenes_3d = pickle.load(f)
            e_time = time.time()
            print(
                f"Loaded cached nuscenes dataset in {(e_time - s_time):.2f} seconds :>")

        self.nusc_can_bus = NuScenesCanBus(dataroot=self.nuscenes_3d.dataroot)

        self.length = len(self.nuscenes)
        if length is not None:
            self.length = min(self.length, length)

        self.pipelines = []
        self.external_helper = construct_external_query(
            "Qwen/Qwen2.5-VL-72B-Instruct")
        for pipeline in pipelines:
            if pipeline["type"] in PIPELINES:
                self.pipelines.append(PIPELINES[pipeline["type"]](
                    nuscenes=self.nuscenes_3d, **pipeline, external_helper=self.external_helper, can_bus=self.nusc_can_bus))
            else:
                raise ValueError("Pipeline {} not found".format(pipeline))
        self.container_out_key_comb = container_out_key_comb
        # defines in this version

    def cache_data(self, cache_filename):
        assert cache_filename.endswith(".json")
        # create cache file if not exists
        cache_filename = os.path.join("data", "nuscenes", cache_filename)

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

    def __len__(self):
        return self.length

    def get_container_in(self, idx):
        return self.nuscenes[idx]

    def cache_queries(self, query_filename, pipeline, max_len=8000):
        assert query_filename.endswith(".json")
        # cache all queries
        assert pipeline in self.pipelines, "Pipeline not in the list of pipelines"
        # import pdb
        # pdb.set_trace()
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
            # import pdb
            # pdb.set_trace()
            # images = container_out['images']
            # q, a, l1, l2, nn = pipeline(container_out, batch)
            # if l1>=3 and l2>=10 and nn:
            #     all_queries.append({"images":images, "messages":[{"role":"user", "content": q}, {"role":"assistant", "content":a}]})
            query = pipeline.cache_construct_query(container_out, batch)
            all_queries.append(query)
            # query = pipeline.cache_construct_query_rest(container_out, batch)
            # if query is not None:
            #     all_queries.append(query)

        # divide the queries into chunks, and save them
        # print(len(all_queries))
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

    def __getitem__(self, original_idx):
        idx = original_idx  # Work with a copy
        while True:
            resample = False
            batch = self.nuscenes[idx]  # Always use current idx
            # pipeline
            container_out = {}
            for pipeline in self.pipelines:
                container_out = pipeline(container_out, batch)
            # pipeline
            for key in self.container_out_key_comb:
                assert key in container_out["buffer_container"], "Key {} not found in container_out.buffer_container".format(
                    key)
                container_out["messages"][1]["content"] += container_out["buffer_container"][key]

                if container_out["buffer_container"][key] == "%DROP%":
                    import random
                    idx = random.randint(0, len(self) - 1)
                    resample = True

            container_out['drop_flag'] = container_out["buffer_container"]['planning'] == "%DROP%"
            if resample:
                continue

            # pop buffer container
            container_out.pop("buffer_container")
            # Store original index before return
            container_out["original_idx"] = original_idx
            return container_out

    def evaluate(self, jsonl_file):
        # load jsonl file to a list of dict
        predicted_data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                predicted_data.append(json.loads(line))

        assert len(predicted_data) >= len(
            self), "Length of predictions and dataset do not match: {} vs {}".format(len(predicted_data), len(self))

        # evaluate every single prediction
        for idx in range(len(self)):
            pred = predicted_data[idx]

            container_in = self.nuscenes[idx]
            container_in["idx"] = idx
            container_out = self.__getitem__(idx)

            # evaluate the prediction
            for pipeline in self.pipelines:
                pipeline.evaluation_update(pred, container_out, container_in)

        results = {}
        for pipeline in self.pipelines:
            pipeline.evaluation_compute(results)
        print(results)
        return results


def generate_batch_dataset():
    experiments = [
        {"id": "exp", "pipelines": [
            {"type": "metadata", "use_image": "1v"},
            {"type": "ego_status", "mode": "x-y"},
            {"type": "planning", "mode": "x-y"},
        ], "container_out_key_comb": ["planning"]},
    ]

    for experiment in experiments:
        v0_pipelines = experiment["pipelines"]
        v0_container_out_key_comb = experiment["container_out_key_comb"]

        dataset = VLMNuScenes(mode="test", pipelines=v0_pipelines,
                              container_out_key_comb=v0_container_out_key_comb)
        dataset.cache_data(f"nuscenes_test.json")

        dataset = VLMNuScenes(mode="train", pipelines=v0_pipelines,
                              container_out_key_comb=v0_container_out_key_comb)
        dataset.cache_data(f"nuscenes_train_b2.json")


if __name__ == "__main__":
    generate_batch_dataset()

import re
import os
import json
import torch
import traceback
import numpy as np
from data_engine.datasets.nuscenes.loaders.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric
from data_engine.common_misc.external_helpers.openai_query import construct_external_query


class PromptNuScenesBlueprint:
    def __init__(self, nuscenes=None, cache_filename=None, container_out_key=None, external_helper=None, need_helper=False, **kwargs):
        self.nuscenes = nuscenes
        self.external_helper = external_helper
        self.container_out_key = container_out_key
        self.need_helper = need_helper
        if cache_filename is not None:
            self.cache_response_filename = os.path.join("data_engine", "data_storage", "cached_responses", cache_filename)
        else:
            self.cache_response_filename = None
        
        if self.cache_response_filename is not None:
            if os.path.exists(self.cache_response_filename):
                print(f"Loading cache from {self.cache_response_filename}")
                with open(self.cache_response_filename, "r") as f:
                    self.cache = json.load(f)
            else:
                print(f"Creating cache at {self.cache_response_filename}")
                self.cache = {}
        
    def cleanup(self, *args):
        if self.cache_response_filename is not None:
            with open(self.cache_response_filename, "w") as f:
                json.dump(self.cache, f)


    def get_query(self, container_out, container_in):
        raise NotImplementedError  # implement as you go
    
    def get_token(self, container_in):
        return container_in['img_metas'].data['token']
    
    def cache_construct_query(self, container_out, container_in):
        # can be overwritten to support multiple ids
        query_final, this_images = self.get_query(container_out, container_in)
        this_id = self.get_token(container_in)
        messages = [
            {"role": "user", "content": query_final},
            {"role": "assistant", "content": "[Please answer the question.]"}
        ]
        query = {"id": this_id, "images": this_images, "messages": messages}
        return query

    def cache_construct_query_rest(self, container_out, container_in):
        # can be overwritten to support multiple ids
        query_final, this_images = self.get_query(container_out, container_in)
        this_id = self.get_token(container_in)
        messages = [
            {"role": "user", "content": query_final},
            {"role": "assistant", "content": "[Please answer the question.]"}
        ]
        query = {"id": this_id, "images": this_images, "messages": messages}
        if this_id in self.cache:
            return None
        return query

    def cache_from_response(self, helper_res, container_out, container_in,):
        helper_ret = helper_res["predict"]
        try:
            helper_ret = self.parse_helper_ret(helper_ret, container_out, container_in)
            self.save_to_cache(helper_ret, container_out, container_in, clean=False)
        except:
            this_scene_token = self.get_token(container_in)
            print(f"Error saving to cache for scene {this_scene_token}, {helper_ret}")

    def save_to_cache(self, helper_ret: str, container_out, container_in, clean=True):
        # will modify the cache
        if helper_ret and self.cache_response_filename is not None:
            this_id = self.get_token(container_in)
            self.cache[this_id] = helper_ret
            if clean:
                self.cleanup()
        
    def load_from_cache(self, container_out, container_in):
        this_id = self.get_token(container_in)
        if this_id in self.cache:
            helper_ret = self.cache[this_id]
            return self.format_output(helper_ret, container_out, container_in)
        else:
            return None
    
    def parse_helper_ret(self, helper_ret, container_out, container_in):
        
        # for every line in helper_ret, if there are only two ` symbols, replace that two symbols with ```
        helper_ret_splitted = helper_ret.split("\n")
        helper_ret_splitted_new = []
        for i in range(len(helper_ret_splitted)):
            if helper_ret_splitted[i].count("`") == 2:
                helper_ret_splitted[i] = helper_ret_splitted[i].replace("``", "```")
            if helper_ret_splitted[i].count("`") == 1:
                helper_ret_splitted[i] = helper_ret_splitted[i].replace("`", "```")
            if helper_ret_splitted[i].strip().startswith("//"):
                continue
            helper_ret_splitted_new.append(helper_ret_splitted[i])
        helper_ret = "\n".join(helper_ret_splitted_new)
        
        # helper_ret = '```json\n{\n....}\n```' now remove the ```json and ```, and parse the json
        pattern = re.compile(r"```json(.*?)```", re.DOTALL)
        helper_ret_match = pattern.search(helper_ret)
        
        if helper_ret_match is None:
            released_pattern = re.compile(r"```(.*?)```", re.DOTALL)
            helper_ret_match = released_pattern.search(helper_ret)
            assert helper_ret_match is not None
        
        helper_ret = helper_ret_match.group(1)
        assert self.external_helper is not None
        helper_ret_fixed = self.external_helper.fix_helper_ret(helper_ret)
        helper_ret = json.loads(helper_ret_fixed)
        return helper_ret
    
    def load_from_external_helper(self, container_out, container_in):
        print("Loading from external helper")
        attempts = 0
        max_attempts = 3
        helper_ret = None
        
        while attempts < max_attempts:
            try:
                query_final, this_images = self.get_query(container_out, container_in)
                print(">>>>>>>>>")
                print(query_final)
                print("Images:", str(this_images))
                print(">>>>>>>>>")
                
                assert self.external_helper is not None, "External helper is not loaded."
                stream_helper_ret = self.external_helper.query_with_context(query_final, img=this_images)
                
                print("<<<<<<<<<<")
                streamed_data = ""
                for chunk in stream_helper_ret:
                    # if 'choices' in chunk and 'delta' in chunk['choices'][0]:
                    arrived_content = chunk.choices[0].delta.content
                    streamed_data += arrived_content
                    print(arrived_content, end='', flush=True)  # Optional: print each chunk as it arrives       
                helper_ret = self.parse_helper_ret(streamed_data, container_out, container_in)
                print("<<<<<<<<<<")
                
                self.save_to_cache(helper_ret, container_out, container_in)
                break  # Break the loop if successful
            except Exception as e:
                print(f"Error parsing the external model response on attempt {attempts + 1}:")
                print(e)
                traceback.print_exc()
                attempts += 1

        if helper_ret is None:
            helper_ret = {}
        this_out = self.format_output(helper_ret, container_out, container_in)
        return this_out

    def format_output(self, helper_ret, container_out, container_in) -> str:
        raise NotImplementedError  # implement as you go

    def __call__(self, container_out, container_in):
        # container_out, container_in -> helper_ret -> output (cache operates here) -> container_out
        this_out = None
        if self.cache_response_filename is not None:
            this_out = self.load_from_cache(container_out, container_in)
        
        if this_out is None:  # cache miss or not caching
            if self.need_helper:
                this_out = self.load_from_external_helper(container_out, container_in)
            else:
                this_out = self.format_output({}, container_out, container_in)

        container_out["buffer_container"][self.container_out_key] = this_out
        return container_out

    def evaluation_update(self, pred, container_out, container_in):
        pass

    def evaluation_compute(self, results):
        pass
    
    
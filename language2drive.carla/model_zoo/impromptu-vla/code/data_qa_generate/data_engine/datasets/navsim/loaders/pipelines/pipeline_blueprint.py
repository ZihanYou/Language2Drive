import re
import os
import json
import torch
import traceback
import numpy as np
from data_engine.common_misc.external_helpers.openai_query import construct_external_query
from data_engine.datasets.nuscenes.loaders.pipelines.pipeline_blueprint import PromptNuScenesBlueprint


class PromptNavsimBlueprint(PromptNuScenesBlueprint):
    def __init__(self, navsim=None, cache_filename=None, container_out_key=None, external_helper=None, need_helper=False, **kwargs):
        super().__init__(nuscenes=navsim, cache_filename=cache_filename, container_out_key=container_out_key, external_helper=external_helper, need_helper=need_helper)
        self.navsim = navsim

    def get_token(self, container_in):
        return container_in['token']

    def get_query(self, container_out, container_in):
        raise NotImplementedError  # implement as you go

    def format_output(self, helper_ret, container_out, container_in) -> str:
        raise NotImplementedError  # implement as you go


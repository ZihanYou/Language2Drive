import re
import os
import math
import torch
import numpy as np

from data_engine.datasets.navsim.loaders.pipelines.pipeline_blueprint import PromptNavsimBlueprint


class PromptNavsimSceneDescription(PromptNavsimBlueprint):

    def __init__(self, navsim, **kwargs):
        super().__init__(navsim=navsim, cache_filename="navsim_scene_description.json", container_out_key="scene_description", need_helper=True, **kwargs)
    
    def get_query(self, container_out, container_in):        
        
        
        self.query_prompt = """## Problem Formulation\nYou are an expert in driving scene analysis. Your task is to analyze a driving scene using information provided from 3 camera views:  
1. **Front view** (focus on this view with the most attention) <image>  
2. **Front-right view** <image>  
3. **Front-left view** <image>

You will analyze the driving scene based on the following chain of thought. Please answer EACH question proposed for a better understanding of the scene.

Q0. Descibe the image of current time (I1, I2, I3), but more focus on static scene information, i.e., background information instead of dynamic road agents.

Q1. Future movement understanding. In C1, an external expert planner gives the following meta plans for the ego vehicle. Try to guess why such meta plans are deduced based on the scene information (please focus only on the static scene information). It is okay if you cannot find anything related to the meta plans in the background image.

If you are analyzing the possible influence in the subsequent questions, but tell them in the tone as if you are the ego vehicle. That is to say, take the information from the image and try to understand the scene from the perspective of the ego vehicle. If you are using metaplans, try to fake that you come up with the metaplans based on the image content. For example, you might say, "I see the road is narrow, so I might need to slow down to avoid hitting the side of the road." Do not say something like "The expert planner says I need to slow down, and I think that is because the road is narrow."

Q2. Try to find the most matching category for the scene based on the image content.

Q2.1. What is the weather condition in the scene? How does it affect driving strategies?

Q2.2. What is the time of day in the scene? How does it affect driving strategies?

Q2.3. What is the type of road in the scene? What are the challenges associated with it for driving? Are there traffic signs or signals? If yes, what do they indicate? How many lanes are there? Which lane is the ego vehicle in? What are the possible maneuvers? Are the roads are straight or curved? If curved, in which direction?

Q2.4. How many lanes are there? Which lane is the ego vehicle in? What are the possible maneuvers? Are the roads are straight or curved? If curved, in which direction?

Q2.5. Based on the scene information, tag the category that matches the scene from the following classifications (you can choose multiple categories if applicable):

categories = [
    "Road Construction",
    "Close-range Cut-ins",
    "Roundabout",
    "Animals Crossing Road",
    "Traffic Police Officers",
    "Blocking Traffic Lights",
    "Cutting into Other Vehicle",
    "Ramp",
    "Debris on the Road",
    "Narrow Roads",
    "Pedestrians Popping Out",
    "People on Bus Posters",
    "Complex Intersections",
    "Near Multiple Vehicles",
    "On Pickup Dropoff",
    "Turning at Intersection",
    "Waiting for Traffic Lights",
    "Emergency Vehicles",
    "Parking Lot",
]

Please consider these tags one by one and try to find the matching category for the scene based on the image content. DO NOT THINK UP TAGS BY YOURSELF. Only choose from the above list.

Q3. Aggregate Results. Please aggregate all your *answer* (not the reasoning process) into a single JSON file.

Your output format should be a json object with the following structure:
```json
{{
    "E_static_desc": "$$static scene description$$",
    "E_weather": "$$weather condition$$",
    "E_time": "$$judged by the brightness of the image$$",
    "E_road": "$$road type, can be a list of multiple tags$$",
    "E_lane": "$$lane positioning and possible maneuvers, in less than 3 words$$",
    "Scenario Category": $$category most matching the scene based on the pic you see$$",
    "S_analysis": "(Write the above analysis of background of driving scene in the tone of ego car at current time in ONE PARAGRAPH. Keep answer from (Q0) to (Q2.5) in this paragraph, but do not include content from metaplans (C1). DO NOT REPEAT THE META PLANS AT THE END.)" // Just act as if you think up all the context information by yourself.
}}
```

Please include the '`' symbol at the beginning and end of the JSON format to ensure proper formatting. Please ensure the keys are kept as is to ensure proper evaluation.

Q5. After solving all above questions, self-check if you have summarized the keys in the JSON format. If not, please do so. If this meets, please end your answer here with "STOP" token. Do NOT output any code or results or `final answers` paragraph below this cell.

## Context

## C1. Meta Plans

An external expert planner gives the following meta plans for the ego vehicle.

%%c1%%

Try to guess why such meta plans are deduced from the background scene information. It is okay if you cannot find anything related to the meta plans in the background image.


## Answer

Please think step by step to solve this problem. Subtitle your solutions by stages, i.e., A0, A1, etc. Note that A3 shall include ```json and ```.\n"""

        c1_txt = container_out["buffer_container"]["meta_planning"]
        self.query_prompt = self.query_prompt.replace("%%c1%%", c1_txt)

        assert len(container_out["images"]) == 3
        return self.query_prompt, container_out["images"]


    def format_output(self, helper_ret, container_out, container_in):
        # raise NotImplementedError("This method should be implemented by the subclass.")
        scene_desc_string = ""
        scene_desc_string += "<Scene Description>\n"
        scene_desc_string += helper_ret['S_analysis'] + "\n"
        scene_desc_string += "</Scene Description>\n"
        return scene_desc_string

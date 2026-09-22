import re
import os
import torch
import numpy as np

from data_engine.datasets.nuscenes.loaders.pipelines.pipeline_blueprint import PromptNuScenesBlueprint

class PromptNuScenesRoadAgentAnalysisText(PromptNuScenesBlueprint):
    
    def __init__(self, nuscenes, **kwargs):
        super().__init__(nuscenes=nuscenes, cache_filename="nuscenes_road_agent_analysis_text.json", container_out_key="road_agent_analysis_text", need_helper=True, **kwargs)
    
    
    def get_query(self, container_out, container_in):        
        self.query_prompt = """You are an expert in driving scene analysis. Your task is to analyze a driving scene using information provided from six camera views:  
1. **Front view** (focus on this view with the most attention) <image>  
2. **Front-right view** <image>  
3. **Front-left view** <image>  
4. **Back view** <image>  
5. **Back-left view** <image>  
6. **Back-right view** <image>  

In this stage, your task is to identify **critical objects** in the scene and analyze them based on the following three perspectives:  

1. **Static Attributes**: Describe the inherent, unchanging properties of the object. These include visual characteristics (e.g., a roadside billboard, a stationary parked car, or oversized cargo on a truck).  
2. **Motion States**: Describe the object's dynamics, including its position, direction of movement, speed, and current actions (e.g., a pedestrian crossing the street, a vehicle merging into the lane, or a cyclist stopped at a red light).  
3. **Particular Behaviors**: Highlight any special or unusual actions or gestures that could influence the ego vehicle’s next decision (e.g., a pedestrian waving to signal crossing intentions, a car suddenly braking, or a cyclist swerving into the lane).  

After analyzing these traits, you must assess the **potential influence** of these objects on the ego vehicle, explaining how they might affect its driving decisions, such as slowing down, changing lanes, or stopping.  

Your output should be in JSON format with the following structure:  

```json
{{
    "Critical_Objects": [
        {{
            "class": "$category of the object, using nouns$",
            "characteristics": "$description of the object from Static Attributes, Motion States, and Particular Behaviors, written in a paragraph$",
            "influence": "$potential influence of the object on the ego vehicle, written in a paragraph$"
        }},
        {{
            "class": "$category of another object$",
            "characteristics": "$description of this object from Static Attributes, Motion States, and Particular Behaviors, written in a paragraph$",
            "influence": "$potential influence of this object on the ego vehicle, written in a paragraph$"
        }}
        // Add more objects as needed
    ]
}}
```

**Examples for Better Understanding:**

1. **Example 1**: A pedestrian crossing the road.  
   ```json
   {{
       "Critical_Objects": [
           {{
               "class": "pedestrian",
               "characteristics": "A pedestrian is wearing a red jacket and carrying a backpack. They are currently walking across a zebra crossing from left to right. The pedestrian has turned their head toward the ego vehicle, possibly checking for traffic.",
               "influence": "The ego vehicle should slow down and prepare to stop to allow the pedestrian to cross safely, as they are directly in the vehicle's path."
           }}
       ]
   }}
   ```

2. **Example 2**: A stopped vehicle with hazard lights on.  
   ```json
   {{
       "Critical_Objects": [
           {{
               "class": "stopped vehicle",
               "characteristics": "A white sedan with visible hazard lights on the right side of the road. The vehicle is stationary with no visible driver inside. The hazard lights suggest it may be broken down or parked temporarily.",
               "influence": "The ego vehicle should maintain a safe distance and consider changing lanes to avoid potential obstruction caused by the stopped vehicle."
           }}
       ]
   }}
   ```

3. **Example 3**: A cyclist swerving into the lane.  
   ```json
   {{
       "Critical_Objects": [
           {{
               "class": "cyclist",
               "characteristics": "A cyclist wearing a yellow helmet and blue jacket. The cyclist is moving at a moderate speed but is swerving slightly into the ego vehicle's lane. The swerving behavior suggests they may be avoiding an obstacle or losing control.",
               "influence": "The ego vehicle should reduce speed and maintain a larger buffer zone to avoid a potential collision if the cyclist continues swerving into the lane."
           }}
       ]
   }}
   ```

4. **Example 4**: A truck with oversized cargo.  
   ```json
   {{
       "Critical_Objects": [
           {{
               "class": "truck",
               "characteristics": "A large truck with oversized cargo extending beyond its rear. The truck is moving slowly in the right lane. The oversized cargo is poorly secured and appears to sway slightly with the vehicle's motion.",
               "influence": "The ego vehicle should avoid driving too close to the truck and consider overtaking it safely to minimize the risk of collision with the swaying cargo."
           }}
       ]
   }}
   ```

5. **Example 5**: A dog running across the road.  
   ```json
   {{
       "Critical_Objects": [
           {{
               "class": "dog",
               "characteristics": "A small brown dog is running diagonally across the road from the right sidewalk to the left. The dog is moving unpredictably and may stop or change direction suddenly.",
               "influence": "The ego vehicle should slow down immediately and prepare to stop to avoid hitting the dog, as its movements are erratic and it is crossing the vehicle's path."
           }}
       ]
   }}
   ```

By following this structure and using the examples as a guide, you can provide detailed and actionable analyses of driving scenes. Output JSON in at most 800 tokens. Now give your output in the above JSON format:\n"""

        assert len(container_out["images"]) == 6  # must be 6 images
        return self.query_prompt, container_out["images"]

    
    def format_output(self, helper_ret, container_out, container_in):
        road_agent_analysis_str = ""
        road_agent_analysis_str += "<Road Agent Analysis>\n"
        road_agent_analysis_str += "Then I'll analyze all critical agents in the scene based on their static attributes, motion states, and particular behaviors. I'll also assess their potential influence on the ego vehicle.\n\n"
        
        for agent in helper_ret["Critical_Objects"]:
            road_agent_analysis_str += f"[{agent['class']}] "
            road_agent_analysis_str += agent['characteristics'] + " "
            road_agent_analysis_str += agent['influence'] + "\n"
        
        road_agent_analysis_str += "</Road Agent Analysis>\n"
        return road_agent_analysis_str
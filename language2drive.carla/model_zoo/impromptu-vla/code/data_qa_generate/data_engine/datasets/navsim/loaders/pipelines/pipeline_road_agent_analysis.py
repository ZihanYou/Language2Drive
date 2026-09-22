import re
import os
import math
import torch
import numpy as np

from data_engine.datasets.navsim.loaders.pipelines.pipeline_blueprint import PromptNavsimBlueprint
from data_engine.datasets.navsim.loaders.navsim.common.enums import BoundingBoxIndex


class PromptNavsimRoadAgentAnalysis(PromptNavsimBlueprint):

    def __init__(self, navsim, **kwargs):
        super().__init__(navsim=navsim, cache_filename="navsim_road_agent_analysis.json", container_out_key="road_agent_analysis", need_helper=True, **kwargs)
    
    def get_query(self, container_out, container_in):        
        
        context_message = ""
        annotations = container_in['frame_data'][3]['annotations']
        
        from data_engine.datasets.navsim.loaders.navsim.visualization.camera import _transform_annotations_to_camera, _rotation_3d_in_axis, _transform_points_to_image
        
        cam_f0 = container_in['frame_data'][3]['cameras']['cam_f0']
        cam_r0 = container_in['frame_data'][3]['cameras']['cam_r0']
        cam_l0 = container_in['frame_data'][3]['cameras']['cam_l0']
        cams = {
            "FRONT": cam_f0,
            "FRONT_RIGHT": cam_r0,
            "FRONT_LEFT": cam_l0
        }
        
        for cam_name, camera in cams.items():
            
            cam_context = f"### View {cam_name}\n"
            
            context_boxes = []
            box_labels = annotations.names
            boxes = _transform_annotations_to_camera(
                annotations.boxes,
                camera['sensor2lidar_rotation'],
                camera['sensor2lidar_translation'],
            )
            box_positions, box_dimensions, box_heading = (
                boxes[:, BoundingBoxIndex.POSITION],
                boxes[:, BoundingBoxIndex.DIMENSION],
                boxes[:, BoundingBoxIndex.HEADING],
            )
            corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
            corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
            corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
            corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
            corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
            corners += box_positions.reshape(-1, 1, 3)

            # Then draw project corners to image.
            box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera['intrinsics'])
            box_corners = box_corners.reshape(-1, 8, 2)
            corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
            valid_corners = corners_pc_in_fov.any(-1)

            box_bevs, box_corners, box_labels, box_velocity_3d = annotations.boxes[valid_corners], box_corners[valid_corners], box_labels[valid_corners], annotations.velocity_3d[valid_corners]
        
        
            for name_value, box_value, velocity_3d, img_corner in zip(box_labels, box_bevs, box_velocity_3d, box_corners):
                
                x, y, heading = (
                    box_value[BoundingBoxIndex.X],
                    box_value[BoundingBoxIndex.Y],
                    box_value[BoundingBoxIndex.HEADING],
                )
                
                distance = np.sqrt(x ** 2 + y ** 2)
                if distance > 45:
                    continue
                
                if img_corner.mean(0)[0] < 0 or img_corner.mean(0)[0] > 1920 or img_corner.mean(0)[1] < 0 or img_corner.mean(0)[1] > 1080:
                    continue
                
                img_corner = img_corner.mean(0).tolist()
                img_corner[0] /= 1920
                img_corner[1] /= 1080
                
                velocity = velocity_3d[:2]
                
                context_boxes.append(
                    {"x": x, "y": y, "velocity": velocity, "img_corner": img_corner, "name": name_value}
                )
        
            # sort according to thetas (descending)
            if len(context_boxes) > 0:
                context_boxes = sorted(context_boxes, key=lambda x: x["x"])
        
            def format_context_box(box):
                x = int(box["x"])
                y = int(box["y"])
                velocity = np.round(box["velocity"], 2)
                vx, vy = velocity[0].item(), velocity[1].item()
                img_corner_x, img_corner_y = box["img_corner"]
                img_corner_x = round(img_corner_x, 2)
                img_corner_y = round(img_corner_y, 2)
                ret =  f"Name: {box['name']}, [{x}, {y}] m in BEV space ([{img_corner_x}, {img_corner_y}] on the camera view)"
                
                v = np.sqrt(vx ** 2 + vy ** 2)
                if v > 0.5:
                    ret += f", moving [{vx}, {vy}] m/s in BEV space.\n"
                else:
                    ret += " not moving.\n"
                return ret
            
            context_box_txt = []
            for box in context_boxes:
                context_box_txt.append(format_context_box(box))
            context_box_txt = ''.join(context_box_txt)
            cam_context += context_box_txt
            context_message += cam_context
        
        self.query_prompt = """## Problem Formulation\nYou are an expert in driving scene analysis. Your task is to analyze a driving scene using information provided from 3 camera views:  
1. **Front view** (focus on this view with the most attention) <image>  
2. **Front-right view** <image>  
3. **Front-left view** <image>

You will analyze the driving scene and identify **critical objects** in the scene and analyze them. Please *strictly* follow the chain-of-thought below, i.e., answer EACH question proposed:

Q0. Dense caption the image of current time (I1, I2, I3). You should focus on your analysis on both common road agents, including vehicles such as cars, trucks, buses, and motorcycles, as well as pedestrians, cyclists, and traffic control devices like traffic lights and stop signs, and uncommon agents, encompassing emergency vehicles, construction equipment, large wildlife, debris, roadblocks, or spills, and even special events like parades or marathons. 

Q1. Road Agent Analysis. 

Q1.1. **Static Attributes**: Describe the inherent, unchanging properties of the object. These include visual characteristics (e.g., a roadside billboard, a stationary parked car, or oversized cargo on a truck).

Q1.2. **Motion States**: Describe the object's dynamics, including its position, direction of movement, speed, and current actions (e.g., a pedestrian crossing the street, a vehicle merging into the lane, or a cyclist stopped at a red light).  

Q1.3. **Particular Behaviors**: Highlight any special or unusual actions or gestures that could influence the ego vehicle’s next decision (e.g., a pedestrian waving to signal crossing intentions, a car suddenly braking, or a cyclist swerving into the lane).  

Q1.4/ **Potential Influence** of these objects on the ego vehicle, explaining how they might affect its driving decisions, such as slowing down, changing lanes, or stopping.  

Q2. 3D Location Understanding in Numbers. In C1, we provide a list of all objects detected in the scene by an external detector. While the detector may not be perfectly accurate, it offers a reasonable understanding of the 3D distances and locations of these objects. Try to ground these objects within the image, and complement the description of the objects in (Q1.1, Q1.2) with the numerical data. Also, the objects are only provided with raw categorical names, so please provide their attributes in a more detailed manner.

Q3. Critical Object Selection and Influence Revision. In C2, an external expert planner gives the following meta plans for the ego vehicle. Try to guess why such meta plans are deduced to identify the critical objects in the scene. You can use these meta plans to analyze which objects are critical enough to affect the ego vehicle to have such meta plans. You should also revise the predicted behavior (Q1.3) of the road agents and their potential influence (Q1.4) based on the actual meta plans.

For example, if the meta plan is to "slow down and prepare to stop," you should identify the objects that might cause the ego vehicle to slow down and stop. If the meta plan is speeding up, maybe the road is clear and there are no critical objects jamming the road, and in such cases, you should identify the objects that might be taken into account to ensure a safe acceleration.

Q4. Aggregate Results. Please aggregate all your *revised* results (i.e., predictions of Q1 grounded on C1 and C2) into a single JSON file. You can try to add numbers for better details.

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
        // Add more objects as needed  <-- DO NOT OUTPUT THIS LINE!
    ],
    "R_analysis": "(Write the above analysis of critical objects in the tone of ego car at current time in ONE PARAGRAPH. Introduce the objects you see, their characteristics, and their future influence on the ego vehicle. Keep the quantitative analysis of the previous sections (Q2, Q3, C1), such as locations or velocities, but do not include content from meta plans (C2).) DO NOT REPEAT THE META PLANS AT THE END." // Keep the numbers within the previous keys. Follow the actual meta plans as the influence of the critical objects on the ego vehicle. Do NOT use phases like `To execute the planned actions` or `according to the external detector`. Just act as if you think up all the context information by yourself.
}}
```

Please include the '`' symbol at the beginning and end of the JSON format to ensure proper formatting. Please ensure the keys are kept as is to ensure proper evaluation.

Q5. After solving all above questions, self-check if you have summarized the keys in the JSON format. If not, please do so. If this meets, please end your answer here with "STOP" token. Do NOT output any code or results or `final answers` paragraph below this cell.

## Context

## C1. External Detector Detected Objects

In this section, I will provide a list of all objects detected in the scene by an external detector. While the detector may not be perfectly accurate, it offers a reasonable understanding of the 3D distances and locations of these objects. Try to ground these objects within the image. There may be additional objects not listed, so if you identify more, please include them as well.

(In BEV space, positive X is forward, positive Y is left. This applies for both the BEV location and the velocity of the objects.)

(On Camera View space, X and Y are normalized to [0, 1], where (0, 0) is the top-left corner and (1, 1) is the bottom-right corner.)

%%c1%%

## C2. Meta Plans

An external expert planner gives the following meta plans for the ego vehicle.

%%c2%%

Try to guess why such meta plans are deduced to identify the critical objects in the scene. In other words, you can use these meta plans to analyze the potential influence of the critical objects on the ego vehicle.

## Answer

Please think step by step to solve this problem. Subtitle your solutions by stages, i.e., A0, A1, etc. Note that A4 shall include ```json and ```.\n"""

        self.query_prompt = self.query_prompt.replace("%%c1%%", context_message)
        
        c2_txt = container_out["buffer_container"]["meta_planning"]
        self.query_prompt = self.query_prompt.replace("%%c2%%", c2_txt)


        assert len(container_out["images"]) == 3
        return self.query_prompt, container_out["images"]


    def format_output(self, helper_ret, container_out, container_in):
        # raise NotImplementedError("This method should be implemented by the subclass.")
        road_agent_analysis_str = ""
        road_agent_analysis_str += "<Road Agent Analysis>\n"
        road_agent_analysis_str += helper_ret['R_analysis'] + "\n"
        road_agent_analysis_str += "</Road Agent Analysis>\n"
        return road_agent_analysis_str
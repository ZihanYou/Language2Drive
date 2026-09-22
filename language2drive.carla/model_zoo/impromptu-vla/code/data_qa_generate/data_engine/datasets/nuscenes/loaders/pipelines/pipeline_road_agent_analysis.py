import re
import os
import torch
import json
import numpy as np

from data_engine.datasets.nuscenes.loaders.pipelines.pipeline_blueprint import PromptNuScenesBlueprint

class PromptNuScenesRoadAgentAnalysis(PromptNuScenesBlueprint):
    
    def __init__(self, nuscenes, **kwargs):
        super().__init__(nuscenes=nuscenes, cache_filename="nuscenes_road_agent_analysis.json", container_out_key="road_agent_analysis", need_helper=True, **kwargs)
    
        self.scene_frame_ann_2d_filename = "data_engine/data_storage/cached_responses/nuscenes_road_agent_ann_2d.json"
        # if file exists, load it
        if os.path.exists(self.scene_frame_ann_2d_filename):
            with open(self.scene_frame_ann_2d_filename, "r") as f:
                self.scene_frame_ann_2d = json.load(f)
        else:        
            self.scene_frame_ann_2d = {}

    def cleanup(self, *args):
        if self.cache_response_filename is not None:
            with open(self.cache_response_filename, "w") as f:
                json.dump(self.cache, f)
        
        if self.scene_frame_ann_2d_filename is not None:
            print("Saving scene_frame_ann_2d")
            with open(self.scene_frame_ann_2d_filename, "w") as f:
                json.dump(self.scene_frame_ann_2d, f)
    
    def get_query(self, container_out, container_in):        
        self.query_prompt = """## Problem Statement
You are an expert in driving scene agent analysis. You are shown front-view image of a driving scene:
I1. Current time <image>


You will analyze the driving scene and identify **critical objects** in the scene and analyze them. Please *strictly* follow the chain-of-thought below, i.e., answer EACH question proposed:

Q0. Dense caption the image of current time (I1). You should focus on your analysis on both common road agents, including vehicles such as cars, trucks, buses, and motorcycles, as well as pedestrians, cyclists, and traffic control devices like traffic lights and stop signs, and uncommon agents, encompassing emergency vehicles, construction equipment, large wildlife, debris, roadblocks, or spills, and even special events like parades or marathons. 

Q1. Road Agent Analysis. 

Q1.1. **Static Attributes**: Describe the inherent, unchanging properties of the object. These include visual characteristics (e.g., a roadside billboard, a stationary parked car, or oversized cargo on a truck).

Q1.2. **Motion States**: Describe the object's dynamics, including its position, direction of movement, speed, and current actions (e.g., a pedestrian crossing the street, a vehicle merging into the lane, or a cyclist stopped at a red light).  

Q1.3. **Particular Behaviors**: Highlight any special or unusual actions or gestures that could influence the ego vehicle’s next decision (e.g., a pedestrian waving to signal crossing intentions, a car suddenly braking, or a cyclist swerving into the lane).  

Q1.4/ **Potential Influence** of these objects on the ego vehicle, explaining how they might affect its driving decisions, such as slowing down, changing lanes, or stopping.  

Q2. 3D Location Understanding in Numbers. In C1, we provide a list of all objects detected in the scene by an external detector. While the detector may not be perfectly accurate, it offers a reasonable understanding of the 3D distances and locations of these objects. Try to ground these objects within the image, and complement the description of the objects in (Q1.1, Q1.2) with the numerical data.

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
    "R_analysis": "(Write the above analysis of critical objects in the tone of ego car at current time in ONE PARAGRAPH. Introduce the objects you see, their characteristics, and their future influence on the ego vehicle. Keep the quantitative analysis of the previous sections (Q2, Q3, C1), such as locations or velocities, but do not include content from meta plans (C2).) DO NOT INCLUDE THE META PLANS (such as slow down, speed up, etc.) IN YOUR OUTPUT. DO NOT REPEAT THE META PLANS AT THE END." // Keep the numbers within the previous keys. Follow the actual meta plans as the influence of the critical objects on the ego vehicle. Do NOT use phases like `To execute the planned actions` or `according to the external detector`. Just act as if you think up all the context information, including the meta plans by yourself.
}}
```

Please include the '`' symbol at the beginning and end of the JSON format to ensure proper formatting. Please ensure the keys are kept as is to ensure proper evaluation.

Q5. After solving all above questions, self-check if you have summarized the keys in the JSON format. If not, please do so. If this meets, please end your answer here with "STOP" token. Do NOT output any code or results or `final answers` paragraph below this cell.

## Context

## C1. External Detector Detected Objects

In this section, I will provide a list of all objects detected in the scene by an external detector. While the detector may not be perfectly accurate, it offers a reasonable understanding of the 3D distances and locations of these objects. Try to ground these objects within the image. There may be additional objects not listed, so if you identify more, please include them as well.

(Positive X is forward, Positive Y is left. This applies for both the location and the velocity of the objects.)

%%c1%%

## C2. Meta Plans

An external expert planner gives the following meta plans for the ego vehicle.

%%c2%%

Try to guess why such meta plans are deduced to identify the critical objects in the scene. In other words, you can use these meta plans to analyze the potential influence of the critical objects on the ego vehicle.

## Answer

Please think step by step to solve this problem. Subtitle your solutions by stages, i.e., A0, A1, etc. Note that A4 shall include ```json and ```.\n"""


        text_template = self.query_prompt
        
        c1_txt = ""
        
        #! C1-1. read from record
        sample_data_token = self.get_token(container_in)
        sample_record = self.nuscenes.get('sample', sample_data_token)
        sd_record = self.nuscenes.get('sample_data', sample_record['data']['CAM_FRONT'])
        cs_record = self.nuscenes.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = self.nuscenes.get('ego_pose', sd_record['ego_pose_token'])

        #! C1-2. consider in ego space, and camera space. WE USE EGO SPACE coordinates instead of Lidar space.
        from pyquaternion import Quaternion
        all_boxes = []
        all_box_metas = []
        e2g_r = pose_record['rotation']    
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        
        for k in range(len(sample_record['anns'])):
            box = self.nuscenes.get_box(sample_record['anns'][k])
            # 从世界坐标系->车身坐标系, 方向为“前左上”，原点在“后轴中心”
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)
            
            # box metadata
            box_metadata = {}
            box_velocity = self.nuscenes.box_velocity(sample_record['anns'][k])[:2]
            # convert velo from global to ego car
            velo = np.array([*box_velocity, 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T
            box_velocity = velo[:2]
            box_metadata["velocity"] = box_velocity
            box_metadata["name"] = box.name
            
            # 从车身坐标系->相机坐标系
            cam_box = box.copy()
            cam_box.translate(-np.array(cs_record['translation']))
            cam_box.rotate(Quaternion(cs_record['rotation']).inverse)
    
            camera_intrinsic_list = [
                container_in['cam_intrinsic'][0]
            ]
            in_image = False
            for intrinsic in camera_intrinsic_list:
                # in_image = box_in_image(cam_box, intrinsic=intrinsic, imsize=(1600, 900), vis_level=1) or in_image
                imsize = (1600, 900)
                corners_3d = cam_box.corners()
                from nuscenes.utils.geometry_utils import view_points
                corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]  # (2, 8)
                corners_img_avg = corners_img.mean(axis=1)  # array([564.53139382, 475.24563114])
                # print(corners_img[0, :], corners_img[1, :])

                visible1 = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
                visible2 = np.logical_and(visible1, corners_img[1, :] < imsize[1])
                visible3 = np.logical_and(visible2, corners_img[1, :] > 0)
                visible4 = np.logical_and(visible3, corners_3d[2, :] > 1)
                in_image = any(visible4) and all(corners_3d[2, :] > 0.1)
                if in_image:
                    all_boxes.append(box)
                    all_box_metas.append(box_metadata)
                    break
            
        # calculate for each box
        CAMERA_MAPPING = {}
        CAMERA_VIEWS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for k in range(len(sample_record['anns'])):
            box_ann_id = sample_record['anns'][k]
            if box_ann_id in self.scene_frame_ann_2d:
                continue
            
            CAMERA_MAPPING[box_ann_id] = {}
            
            box = self.nuscenes.get_box(sample_record['anns'][k])
            # Transform box to ego space
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)
            
            # Initialize the entry for this box in CAMERA_MAPPING

                
            # Check visibility in each camera view
            for cam in CAMERA_VIEWS:
                sd_record = self.nuscenes.get('sample_data', sample_record['data'][cam])
                cs_record = self.nuscenes.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                
                # Transform box to camera space
                cam_box = box.copy()
                cam_box.translate(-np.array(cs_record['translation']))
                cam_box.rotate(Quaternion(cs_record['rotation']).inverse)
                
                # Camera intrinsic
                intrinsic = container_in['cam_intrinsic'][CAMERA_VIEWS.index(cam)]
                
                # Check if box is visible in the image
                imsize = (1600, 900)
                corners_3d = cam_box.corners()
                corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]
                corners_img_avg = corners_img.mean(axis=1)
                
                keep_mask = corners_img_avg[0] > 0 and corners_img_avg[0] < imsize[0] and corners_img_avg[1] < imsize[1] and corners_img_avg[1] > 0
                keep_mask = keep_mask and all(corners_3d[2, :] > 1)
                
                corners_img_avg = corners_img_avg.tolist()
                box_center = box.center.tolist()
                
                if keep_mask:
                    # Store the relative coordinates and pose in ego coordinates
                    CAMERA_MAPPING[box_ann_id][cam] = (corners_img_avg[0] / 1600, corners_img_avg[1] / 900, box_center)
     
        # update self.scene_frame_ann_2d
        for k in CAMERA_MAPPING:
            if k not in self.scene_frame_ann_2d:
                self.scene_frame_ann_2d[k] = CAMERA_MAPPING[k]

        #! C1-3 visualize: draw boxes
        if False:
            from PIL import Image
            image_path = container_out['images'][0]  # Replace with the actual image array
            image = Image.open(image_path)
            image = np.array(image)
            from matplotlib import pyplot as plt
            plt.clf()
            plt.imshow(image)
            for box in all_boxes:
                cam_box = box.copy()
                cam_box.translate(-np.array(cs_record['translation']))
                cam_box.rotate(Quaternion(cs_record['rotation']).inverse)
                
                corners_3d = cam_box.corners()
                from nuscenes.utils.geometry_utils import view_points
                intrinsic = container_in['cam_intrinsic'][0]
                corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]
                corners_img_avg = corners_img.mean(axis=1)  # array([564.53139382, 475.24563114])
            
                # print(corners_img_avg)
                plt.scatter(corners_img_avg[0], corners_img_avg[1], color='red', marker='x', s=5)
            # lim x to 1600, y to 900
            plt.xlim(0, 1600)
            plt.ylim(900, 0)
            plt.savefig('trash.png', dpi=300)
        
        
        # ! C1-4. write to text
        number = 0
        for box, box_metadata in zip(all_boxes, all_box_metas):
            #! conduct a final filtering.
            box_xy = box.center[:2]
            distance = np.linalg.norm(box_xy)
            if distance > 45:
                continue
            box_xy = np.round(box_xy, 2)
            box_metadata['velocity'] = np.round(box_metadata['velocity'], 2)
            
            #! write to txt
            number += 1
            this_txt = f"[{number}] "
            this_txt += f"{box_metadata['name']}: "
            this_txt += f"Location: {str(box_xy.tolist())}, Velocity {str(box_metadata['velocity'].tolist())}\n"
            
            c1_txt += this_txt
        
        if c1_txt == "":
            c1_txt = "(No objects detected in the scene within 45 meters of ego vehicle)"
            
        else:
            c1_txt += "(Above are the objects detected in the scene within 45 meters of ego vehicle. The location is in the format of [x, y] in meters, and the velocity is in the format of [vx, vy] in m/s.)"

        
        text_template = text_template.replace("%%c1%%", c1_txt)
        
        c2_txt = container_out["buffer_container"]["meta_planning"]
        text_template = text_template.replace("%%c2%%", c2_txt)
        if False:
            print(text_template, container_out["images"])
            exit(-1)
        return text_template, container_out["images"]

    
    def format_output(self, helper_ret, container_out, container_in):
        road_agent_analysis_str = ""
        road_agent_analysis_str += "<Road Agent Analysis>\n"
        road_agent_analysis_str += helper_ret['R_analysis'] + "\n"
        road_agent_analysis_str += "</Road Agent Analysis>\n"
        return road_agent_analysis_str

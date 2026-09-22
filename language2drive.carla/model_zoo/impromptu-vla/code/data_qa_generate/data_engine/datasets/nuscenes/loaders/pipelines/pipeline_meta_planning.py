import re
import os
import torch
import numpy as np

from data_engine.datasets.nuscenes.loaders.pipelines.pipeline_blueprint import PromptNuScenesBlueprint


class PromptNuScenesMetaPlanning(PromptNuScenesBlueprint):
    def __init__(self, nuscenes, use_query_gt_status=True, **kwargs):
        default_params = {
            'cache_filename': 'nuscenes_meta_planning.json',
            'container_out_key': 'meta_planning',
            'need_helper': True,
        }
        
        default_params.update(kwargs)
        
        # 调用父类的 __init__ 方法，并传递更新后的参数
        super().__init__(nuscenes=nuscenes, **default_params)
        
        # super().__init__(nuscenes=nuscenes, cache_filename="nuscenes_meta_planning.json", container_out_key="meta_planning", need_helper=True, **kwargs)
        self.use_query_gt_status = use_query_gt_status

  
    def get_query(self, container_out, container_in):        
        self.query_prompt = """## Problem Statement
You are an expert in autonomous driving planning analysis. You are shown front-view image of a driving scene:
I1. Current time <image>

You will analyze the driving scene and determine the most appropriate actions for the ego vehicle based on the following chain of thought. Please answer EACH question proposed for a better understanding of the scene and the ego vehicle's future movements.

Q0. Descibe the image of current time (I1). Which road lane is the ego vehicle currently in? What are the road conditions? Are the road lines straight or curved? If curved, is it turning left or right? Are there any traffic signs or signals? Are there any obstacles on the road?

Q1. **Previous** and **Future Ground Truth** (GT) Movement (Location and ego status): 
You will be provided with **Previous** and **Future ground truth** location and ego status of the ego vehicle. This serves as the starting point for your analysis. 

Q1.1 Consider the previous ego status and the future ground truth movement. Compared with previous ego status, are the velocities in the future speeding up, keeping a constant speed, or slowing down? (Extract these numbers for EVERY TIMESTEP and comparing them. You must describe the changes in velocity through time. e.g. You can say that the ego agent is first speeding up, then maintaining the speed, and then slowing down. 

Q1.2 Is the direction changing? Please look at the y coordinates in the future (positive indicates left and negative indicates right). If it is changing, is it a fix to stay in the middle of the lane, turning left, turning right, or turning around? What is its difference through time? If it is changing, please look at the image (I1), is it still keeping in the same lane or changing lanes? If it is changing lanes, is it changing to the left or to the right? 

You can use steering angle values to help you determine the direction. However, the difference between the steering angles and y coordinates is that steering angles provide an immediate indication of the direction in which the vehicle is steering. It's important to accumulate these angles over time to understand the overall direction. For example, if there are multiple right steers followed by a sudden left steer, the actual path is still turning right. In contrast, the y coordinates reflect the resultant position over time, showing the actual path taken by the vehicle.

(Note that there might be chances where previous movements are not available. In such cases, try your best to deduce the movements based on the current scene I1.)

Q2. Meta Planning: Deduce the meta action (speed, direction, and lane control) that aligns with the GT velocity and direction. NO NEED TO consider safety and traffic rules YOURSELF. JUST USE THE GT FUTURE MOVEMENT (C1, C2) to deduce the meta plans. You can deduce multiple meta actions for different spans time frames (e.g., in short term, maintain current speed, in long term, speed up).
You need to use number parameters based on the language options for better details. (e.g., how much the velocity changed when speeding up (per 0.5 seconds)? turn left to what degree?)

Q2.1. **Speed-control actions** (discerned from acceleration and braking signals within the ego state data):
    - speed up rapidly
    - speed up
    - slow down
    - slow down rapidly
    - go straight slowly
    - go straight at a constant speed
    - stop
    - wait
    - reverse
    - maintain current speed

Q2.2. **Turning actions** (deduced from steering wheel signals and y coordinate of the ego vehicle):
    - fix to the left to stay in the middle of the lane
    - fix to the right to stay in the middle of the lane
    - turn left - starting phase
    - turn left - ending phase
    - turn right - starting phase
    - turn right - ending phase
    - turn around - starting phase
    - turn around - ending phase
    - maintain current direction

Q2.3. **Lane-control actions** (encompassing lane selection decisions derived from a combination of steering wheel signals and either map or perception data):
    - change lane to the left - starting phase
    - change lane to the left - ending phase
    - change lane to the right - starting phase
    - change lane to the right - ending phase
    - shift slightly to the left
    - shift slightly to the right
    - maintain current lane

Q3. Decision Description. Decision description D articulates the more fine-grained driving strategy the ego vehicle should adopt. It contains three elements: Action A, Subject S, and Duration D. Action pertains to meta actions above. Subject refers to the interacting object, such as a pedestrian, a traffic signal, or a specific lane (please refer to captioned I1 in Q0). Duration indicates the temporal aspect of the action, specifying how long it should be carried out or when it should start.

For example:
- Slow down and wait for the cyclist to pass before continuing to turn right.
- Ensure a safe distance from the vehicles in front and on both sides while moving forward slowly.

Please output the decision description in a natural language form, including the numbers, based on the meta plans deduced in Q2 and subjects in the scene I1.

Q4. Aggregate Results. You need to provide "M_speed", "M_direction", "M_lane" for the meta planning, "M_decision" for the decision description. You need also provide "M_meta_planning", in which you need to tell the decision description in the tone of the ego vehicle.

Please first follow the chain-of-thought above to solve the problem, and finally provide your reasoning and the selected actions **in the following JSON format** (DO NOT USE `\boxed` to wrap it. Use wrap it with the following copyable JSON format):
```json
{
    "M_speed": "(Q2), (if applicable, use the numbers to detailedly describe)",
    "M_direction": "(Q2), (if applicable, use the numbers to detailedly describe)",
    "M_lane": "(Q2), (if applicable, use the numbers to detailedly describe)",
    "M_decision": "(Q3)",
    "M_meta_planning": "I will give a meta plan in this stage. (*answer* from Q2 + Q3)  // Re-tell the above M_decision in the tone of the ego agent at now (t-0). Keep the numbers within the previous keys. NO NEED TO consider safety and traffic rules YOURSELF. JUST engage the previous conent into one detailed natural language form. This paragraph can be a little longer. Do NOT use phases like `This plan aligns with the ground truth future movement and status provided` as conclusion or `according to the ground truth future movement and status` as your reason. Do not add these phrases. JUST GIVE THE PLAN.
}
```

Please include the '`' symbol at the beginning and end of the JSON format to ensure proper formatting. Please ensure the keys are kept as "M_speed", "M_direction", "M_lane", and "M_meta_planning" to ensure proper evaluation. Replace all (Q2) or (Q3) with your answers in natural language.

Q5. After solving all above questions, self-check if you have summarized the previous three keys in the JSON format. If not, please do so. If this meets, please end your answer here with stop token. Do NOT output any code or results or `final answers` paragraph below this cell.

## Context\n"""


        # 1. get current_sample, future_sample
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        current_image_token = this_sample['data']['CAM_FRONT']
        current_image = self.nuscenes.get('sample_data', current_image_token)['filename']
        current_image = os.path.join(self.nuscenes.dataroot, current_image)

        
        # sample_t_tp6 = []  # [t, t+3s]
        # sample_t_tp6.append(this_sample)
        # while len(sample_t_tp6) < 7 and this_sample['next'] != '':
        #     this_sample = self.nuscenes.get('sample', this_sample['next'])
        #     sample_t_tp6.append(this_sample)
        
        # if len(sample_t_tp6) < 7:
        #     # not enough samples. pad from the left side with previous samples
        #     this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        #     while len(sample_t_tp6) < 7 and this_sample['prev'] != '':
        #         this_sample = self.nuscenes.get('sample', this_sample['prev'])
        #         sample_t_tp6.insert(0, this_sample)
        
        # current_image_token, after_image_token, future_image_token = sample_t_tp6[0]['data']['CAM_FRONT'], sample_t_tp6[3]['data']['CAM_FRONT'], sample_t_tp6[6]['data']['CAM_FRONT']
        # current_image = self.nuscenes.get('sample_data', current_image_token)['filename']
        # after_image = self.nuscenes.get('sample_data', after_image_token)['filename']
        # future_image = self.nuscenes.get('sample_data', future_image_token)['filename']
        # current_image = os.path.join(self.nuscenes.dataroot, current_image)
        # after_image = os.path.join(self.nuscenes.dataroot, after_image)
        # future_image = os.path.join(self.nuscenes.dataroot, future_image)
        
        # prompt_t_tp6 = []
        # for sample in sample_t_tp6:
        #     prompt_t_tp6.append(self.previous_stage_cache[sample['token']])
        
        # 2. construct more information for query
        # 2.1 ego status
        query_prev_ego_status = "### C1. Ego status for the previous seconds\n"
        start_idx = container_out['messages'][0]['content'].find("Provided are the previous ego vehicle status recorded over")
        query_prev_ego_status += container_out['messages'][0]['content'][start_idx:]
        
        query_prev_ego_status += "\n(In case this part is missing, please try your best to deduce the movements based on the current scene.)\n"

        
        # 2.3. gt trajectory
        query_gt_trajectory = "### C2. Ground truth movement per 0.5 second for the ego vehicle in the upcoming 3 seconds\n"
        query_gt_trajectory_candidate = container_out['buffer_container']["planning"]
        query_gt_trajectory_candidate = query_gt_trajectory_candidate[query_gt_trajectory_candidate.find("<PLANNING>"):].replace("Predicted", "")
        query_gt_trajectory += query_gt_trajectory_candidate
        
        query_gt_status = "### C3. Ground truth status for the ego vehicle in the upcoming seconds\n"
        this_sample_idx = container_out['buffer_container']['this_sample_idx']
        all_status = container_out['buffer_container']['all_ego_status']
        
        if all_status is not None:    
            future_status = all_status[this_sample_idx + 1 : this_sample_idx + 7]
        
            def convert_status(status) -> str:
                acc_x, acc_y = round(status[0], 2), round(status[1], 2)
                vel_x, _ = round(status[6], 2), round(status[7], 2)
                steering_angle = round(status[9], 2)
                return f"Acceleration: X {acc_x}, Y {acc_y} m/s^2, Velocity: {vel_x} m/s, Steering angle: {steering_angle} (positive: left turn, negative: right turn)"
        
            for i, status in enumerate(future_status):
                query_gt_status += f"(t+{(i+1)*0.5}s) {convert_status(status)}\n"
        else:
            query_gt_status += "(The ground truth status is not available. Please try to continue your answer with information from other contexts.)"

        if not self.use_query_gt_status:
            query_gt_status = ""
        
        
        query_final = f"{self.query_prompt}\n\n{query_prev_ego_status}\n{query_gt_trajectory}\n{query_gt_status}\n## Answer\nPlease think step by step to solve this problem. Subtitle your solutions by stages, i.e., A0, A1, etc. Note that A4 shall include ```json and ```."
        return query_final, [current_image,]

    
    def format_output(self, helper_ret, container_out, container_in):
        # helper_ret = '```
        meta_planning_str = ""
        meta_planning_str += "<Meta Planning>\n"
        meta_planning_str += helper_ret['M_meta_planning'] + "\n"
        meta_planning_str += "</Meta Planning>\n"
        return meta_planning_str
    
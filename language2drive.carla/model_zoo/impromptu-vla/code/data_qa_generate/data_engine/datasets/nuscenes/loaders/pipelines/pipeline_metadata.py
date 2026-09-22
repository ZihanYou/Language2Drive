import os
from data_engine.datasets.nuscenes.loaders.pipelines.pipeline_blueprint import PromptNuScenesBlueprint

class PromptNuScenesMetadata(PromptNuScenesBlueprint):
    def __init__(self, nuscenes=None, use_image="6v", **kwargs):
        super().__init__(nuscenes=nuscenes, container_out_key="", cache_response_filename=None, need_helper=False, **kwargs)
        
        # assert self.nuscenes is not None
        
        self.use_image = use_image
        assert self.use_image in ["6v", "1v", "t-6v", "t-1v", "none"]

    
    def extract_scene_samples(self, container_out, container_in):
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        this_sample_idx = 0  # store the index of this_sample in the scene_samples
        scene_samples = []
        scene_samples.append(this_sample)
        while this_sample['next'] != '':
            this_sample = self.nuscenes.get('sample', this_sample['next'])
            scene_samples.append(this_sample)
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        while this_sample['prev'] != '':
            this_sample = self.nuscenes.get('sample', this_sample['prev'])
            scene_samples.insert(0, this_sample)
            this_sample_idx += 1
        container_out["buffer_container"]["scene_samples"] = scene_samples
        container_out["buffer_container"]["this_sample_idx"] = this_sample_idx
        return container_out
    
    def get_past_images(self, container_out, container_in):
        assert "scene_samples" in container_out["buffer_container"]
        assert "this_sample_idx" in container_out["buffer_container"]
        
        scene_samples = container_out["buffer_container"]["scene_samples"]
        this_sample_idx = container_out["buffer_container"]["this_sample_idx"]
        
        sample_now = scene_samples[this_sample_idx]
        sample_1_5s = scene_samples[max(0, this_sample_idx - 3)]
        sample_3s = scene_samples[max(0, this_sample_idx - 6)]
        
        
        sample_list = []
        if not "t-" in self.use_image:
            sample_list = [sample_now]
        else:
            if this_sample_idx - 6 >= 0:
                sample_list.append(scene_samples[this_sample_idx - 6])
            if this_sample_idx - 3 >= 0:
                sample_list.append(scene_samples[this_sample_idx - 3])
            sample_list.append(sample_now)

        
        
        def get_image_path(sample):
            cam_tokens = [sample['data'][x] for x in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']]
            cam_samples = [self.nuscenes.get('sample_data', x) for x in cam_tokens]
            cam_image_paths = [x['filename'] for x in cam_samples]
            return cam_image_paths
        
        images = []
        if "6v" in self.use_image:
            images = []
            for sample in sample_list:
                images.extend([os.path.join(self.nuscenes.dataroot, x[x.find("samples"):]) for x in get_image_path(sample)])
        else:
            for sample in sample_list:
                images.extend([os.path.join(self.nuscenes.dataroot, get_image_path(sample)[0])])
        return images
    
    def get_user_prompt(self, container_out, container_in, images):
        
        
        len_images = len(images)
        if "6v" in self.use_image:
            len_images = len(images) // 6
        
        prompt_dict = {
            "6v": "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle: (1) front view (which you should focus on with the most attention) <image>, (2) front right view <image>, (3) front left view <image>, (4) back view <image>, (5) back left view <image>, and (6) back right view <image>. ",
            "1v": "You are an autonomous driving agent. You have access to a front view camera image of a vehicle <image>. ",
            "none": "You are an autonomous driving agent.",
            # "t-6v": "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle, listed in the order of the past 3 seconds, 1.5 seconds, and now: (1) front view (which you should focus on with the most attention) <image> <image> <image>, (2) front right view <image> <image> <image>, (3) front left view <image> <image> <image>, (4) back view <image> <image> <image>, (5) back left view <image> <image> <image>, and (6) back right view <image> <image> <image>. ",
            # "t-1v": "You are an autonomous driving agent. You have access to a front view camera image of a vehicle, listed in the order of the past 3 seconds, 1.5 seconds, and now: <image> <image> <image>. "
        }
        prompt_dict["t-6v"] = ""
        if len_images == 3:
            prompt_dict["t-6v"] = "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle, listed in the order of the past 3 seconds, 1.5 seconds, and now: (1) front view (which you should focus on with the most attention) <image> <image> <image>, (2) front right view <image> <image> <image>, (3) front left view <image> <image> <image>, (4) back view <image> <image> <image>, (5) back left view <image> <image> <image>, and (6) back right view <image> <image> <image>. "
        elif len_images == 2:
            prompt_dict["t-6v"] = "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle, listed in the order of the past 1.5 seconds and now: (1) front view (which you should focus on with the most attention) <image> <image>, (2) front right view <image> <image>, (3) front left view <image> <image>, (4) back view <image> <image>, (5) back left view <image> <image>, and (6) back right view <image> <image>. "
        elif len_images == 1:
            prompt_dict["t-6v"] = "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle, listed in the order of the past 3 seconds and now: (1) front view (which you should focus on with the most attention) <image>, (2) front right view <image>, (3) front left view <image>, (4) back view <image>, (5) back left view <image>, and (6) back right view <image>. "
        
        prompt_dict["t-1v"] = ""
        if len_images == 3:
            prompt_dict["t-1v"] = "You are an autonomous driving agent. You have access to a front view camera image of a vehicle, listed in the order of the past 3 seconds, 1.5 seconds, and now: <image> <image> <image>. "
        elif len_images == 2:
            prompt_dict["t-1v"] = "You are an autonomous driving agent. You have access to a front view camera image of a vehicle, listed in the order of the past 1.5 seconds and now: <image> <image>. "
        elif len_images == 1:
            prompt_dict["t-1v"] = "You are an autonomous driving agent. You have access to a front view camera image of a vehicle, listed in the order of the past 3 seconds and now: <image>. "
        
        prompt_input = prompt_dict[self.use_image]
        suffix = "Your task is to do your best to predict future waypoints for the vehicle over the next 3 timesteps, given the vehicle's intent inferred from the images."
        return prompt_input + suffix
    

    def __call__(self, container_out, container_in):
        
        images = []
        container_out = {
            "id": container_in['img_metas'].data['token'],
            "images": [],
            "messages": [
                {"role": "user", "content": ""}, 
                {"role": "assistant", "content": ""}
            ],
            "buffer_container": {}
        }  # init
        
        if 't-' in self.use_image:
            container_out = self.extract_scene_samples(container_out, container_in)
            images = self.get_past_images(container_out, container_in)
            container_out["images"] = images
        else:
            container_out["images"] = container_in['img_metas'].data['img_filename']
            for idx in range(len(container_out["images"])):
                container_out["images"][idx] = os.path.join("data_engine/data_storage/external_datasets/nuscenes/", container_out["images"][idx][container_out["images"][idx].find("samples"):])
            if self.use_image == "1v":
                container_out["images"] = [container_out["images"][0]]
            elif self.use_image == "6v":
                container_out["images"] = container_out["images"]
            elif self.use_image == "none":
                container_out["images"] = []
        
        container_out["messages"][0]["content"] = self.get_user_prompt(container_out, container_in, images)
        return container_out


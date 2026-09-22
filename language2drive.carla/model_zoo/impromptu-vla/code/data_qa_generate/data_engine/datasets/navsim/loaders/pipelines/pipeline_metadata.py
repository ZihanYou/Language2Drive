import os
from data_engine.datasets.navsim.loaders.pipelines.pipeline_blueprint import PromptNavsimBlueprint

class PromptNavsimMetadata(PromptNavsimBlueprint):
    def __init__(self, navsim=None, use_image="3v", **kwargs):
        super().__init__(navsim=navsim, container_out_key="", cache_response_filename=None, need_helper=False, **kwargs)
        
        assert use_image in ["3v", "8v"]
        self.use_image = use_image
    
    
    def extract_scene_samples(self, container_out, container_in):
        import IPython; IPython.embed()
        return container_out
    
    def get_past_images(self, container_out, container_in):
        cameras = container_in['frame_data'][3]['cameras']
        # {'cam_f0': {}, 'cam_l0': {}, 'cam_r0': {}, 'cam_l1': {}, 'cam_r1': {}, 'cam_l2': {}, 'cam_r2': {}, 'cam_b0': {}}
        camera_names = ["cam_f0", "cam_r0", "cam_l0", "cam_r1", "cam_l1", "cam_r2", "cam_l2", "cam_b0"]
        
        if self.use_image == "3v":
            camera_names = ["cam_f0", "cam_r0", "cam_l0"]
            
        images = []
        
        for camera_name in camera_names:
            camera = cameras[camera_name]
            image_path = camera['image_path']
            images.append(image_path)
        
        return images
        
    
    def get_user_prompt(self, container_out, container_in):
        prompt_dict = {
            "3v": "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle: (1) front view (which you should focus on with the most attention) <image>, (2) front right view <image>, and (3) front left view <image>. ",
            "8v": "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle: (1) front view (which you should focus on with the most attention) <image>, (2) front right view <image>, (3) front left view <image>, (4) middle right view <image>, (5) middle left view <image>, (6) back right view <image>, (7) back left view <image>, and (8) back view <image>. "
        }
        prompt_input = prompt_dict[self.use_image]
        suffix = "Your task is to do your best to predict future waypoints for the vehicle over the next 10 timesteps, given the vehicle's intent inferred from the images."
        return prompt_input + suffix
        
        
    def __call__(self, container_out, container_in):
        
        images = []
        container_out = {
            "id": self.get_token(container_in),
            "images": [],
            "messages": [
                {"role": "user", "content": ""}, 
                {"role": "assistant", "content": ""}
            ],
            "buffer_container": {}
        }  # init
        
        images = self.get_past_images(container_out, container_in)
        container_out["images"] = images
        
        container_out["messages"][0]["content"] = self.get_user_prompt(container_out, container_in)
        return container_out

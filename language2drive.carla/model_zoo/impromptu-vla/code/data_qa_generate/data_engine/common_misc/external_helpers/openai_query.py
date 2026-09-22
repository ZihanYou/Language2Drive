from PIL import Image
import requests

def fix_helper_ret(helper_ret: str):
    """
    A function for fixing JSON responses from the external model.
    Input:
        - helper_ret: str. The JSON response from the external model.
    Desc:
        There might be `"` characters in the response that are not properly escaped.
        For example:
            "characteristics": "A white van with "FACTORY OUTLET" writt..."}
        This function will make it to:
            {"characteristics": "A white van with FACTORY OUTLET writt..."}
    """
    helper_ret_lines = helper_ret.split('\n')  # Split by newline
    processed_lines = []

    for line in helper_ret_lines:
        quote_indices = [i for i, char in enumerate(line) if char == '"']  # Find all quote positions
        
        if len(quote_indices) > 4:
            # Keep the first three quotes and the last one
            keep_indices = quote_indices[:3] + [quote_indices[-1]]
            kill_indices = []
            for idx in quote_indices:
                if idx not in keep_indices:
                    kill_indices.append(idx)
            # REMOVE all quotes that are in kill_indices, from right to left
            new_line = line
            for i in kill_indices[::-1]:
                new_line = new_line[:i] + new_line[i + 1:]
                
            processed_lines.append(new_line)
        else:
            # Keep the line as is if it has 4 or fewer quotes
            processed_lines.append(line)

    helper_ret_fixed = '\n'.join(processed_lines)
    return helper_ret_fixed

class ExternalQueryOpenAI:
    def __init__(self, api_key: str, base_url="", model_name="gpt-4o"):
        from openai import OpenAI
        
        self.api_key = api_key
        self.base_url = base_url
        if self.base_url == "":
            self.base_url = "https://api.openai.com/v1"
        self.model_name = model_name
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def query(self, query: str,):
        response = self.client.chat.completions.create(
            model=self.model_name, # ModelScope Model-Id
            messages=[{'role': 'user', 'content': query}],
            stream=True
        )
        return response
    
    def query_with_context(self, query: str, **context):
        if 'img' in context:
            img_paths = context['img']
            if not isinstance(img_paths, list):
                img_paths = [img_paths]
            
            from copy import deepcopy
            img_paths = deepcopy(img_paths)

            
            if isinstance(img_paths[0], str) and not img_paths[0].startswith("data:image"):
                # make it to base64
                for i, img_path in enumerate(img_paths):
                    import base64
                    with open(img_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    img_paths[i] = f"data:image/jpeg;base64,{encoded_image}"

            elif isinstance(img_paths[0], Image.Image):
                # make it to base64
                for i, img_path in enumerate(img_paths):
                    import base64
                    from io import BytesIO
                    buffered = BytesIO()
                    img_path.save(buffered, format="png")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    img_paths[i] = f"data:image/png;base64,{img_str}"


            response = self.client.chat.completions.create(
                model=self.model_name, # ModelScope Model-Id
                messages=[
                    {'role': 'user', 'content': [
                        *[
                            {"type": "image_url", "image_url": {"url": img_path}}
                            for img_path in img_paths
                        ],
                        {"type": "text", "text": query}
                    ]}
                ],
                stream=True
            )
            return response
        else:
            return self.query(query)
    
    def fix_helper_ret(self, helper_ret: str):
        return fix_helper_ret(helper_ret)

def construct_external_query(model_name: str, **kwargs):
    if model_name in ["Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/QVQ-72B-Preview"]:
        return ExternalQueryOpenAI(
            api_key=<your api_key>, 
            base_url='https://api-inference.modelscope.cn/v1/',
            model_name=model_name)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    construct_external_query(model_name="Qwen/Qwen2.5-VL-72B-Instruct")   
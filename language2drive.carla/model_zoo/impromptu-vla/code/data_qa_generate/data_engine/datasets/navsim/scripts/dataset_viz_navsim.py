import os
import json
from tqdm import tqdm

dataset_file = "data/navsim/navsim_test_full.json"
INTERVAL = 100

viz_path = dataset_file.replace(".json", "")
os.makedirs(viz_path, exist_ok=True)
print(f"Saving visualization to {viz_path}")

with open(dataset_file, "r") as f:
    dataset = json.load(f)
    
for idx, data in enumerate(tqdm(dataset)):
    if idx % INTERVAL != 0:
        continue
    
    save_name = f"{idx:06d}.html"
    id = data['id']
    images = data['images']
    query = data['messages'][0]['content']
    gt = data['messages'][1]['content']
    
    def escape_html(text):
        return text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    
    from PIL import Image
    image_paths = images
    target_size = (576, 384)
    output_dir_imgs = os.path.join(viz_path, f"images_{idx}")
    output_img_names = []

    os.makedirs(output_dir_imgs, exist_ok=True)

    # Process each image
    for image_path in image_paths:
        # Extract the camera status from the path
        camera_status = image_path.split('/')[-2]
        
        # Open the image
        with Image.open(image_path) as img:
            # Resize the image
            img_resized = img.resize(target_size, Image.Resampling.NEAREST)
            
            # Create the output file name
            output_file_name = f"{camera_status}.jpg"
            output_file_path = os.path.join(output_dir_imgs, output_file_name)
            output_img_names.append(output_file_name)
            
            # Save the resized image
            img_resized.save(output_file_path, "JPEG")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>All In One</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h2 {{
                color: #333;
            }}
            p {{
                font-family: monospace;
            }}
            .section {{
                margin-bottom: 20px;
            }}
            .images {{
                display: flex;
                flex-wrap: wrap;
            }}
            .images img {{
                margin: 5px;
                max-width: 200px;
            }}
        </style>
    </head>
    <body>
        <div class="section">
            <h2>Query</h2>
            <p>{escape_html(query)}</p>
        </div>
        <div class="section">
            <h2>Ground Truth</h2>
            <p>{escape_html(gt)}</p>
        </div>
        <div class="section">
            <h2>Images</h2>
            <div class="images">
    """
    # Add the images to the HTML content
    for image_file in output_img_names:
        image_path = os.path.join(output_dir_imgs, image_file)
        html_content += f'            <img src="images_{idx}/{image_file}" alt="{image_file}">\n'

    # Close the HTML tags
    html_content += """</div>
        </div>
    </body>
    </html>"""
    
    
    with open(os.path.join(viz_path, save_name), "w") as f:
        f.write(html_content)
        
# zip viz_path to viz_path.zip
os.system(f"zip -qr {viz_path}.zip {viz_path}")

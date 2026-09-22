import copy
import json
import argparse

from tqdm import tqdm
import cv2
import numpy as np

from llava.model.builder import load_pretrained_model, load_senna_pretrained_model
from llava.mm_utils import get_model_name_from_path
from data_tools.senna_qa_utils import eval_model_wo_init
from data_tools.senna_qa_utils import load_image

image_prompt = "<FRONT VIEW>:\n<image>\n" \
               "<FRONT LEFT VIEW>:\n<image>\n" \
               "<FRONT RIGHT VIEW>:\n<image>\n" \
               "<BACK LEFT VIEW>:\n<image>\n" \
               "<BACK RIGHT VIEW>:\n<image>\n" \
               "<BACK VIEW>:\n<image>\n"

desc_question = "Suppose you are driving, and I'm providing you with the image " \
            "captured by the car's XXX, generate a description of the driving scene " \
            "which includes the key factors for driving planning, including the positions " \
            "and movements of vehicles and pedestrians; prevailing weather conditions; " \
            "time of day, distinguishing between daylight and nighttime; road conditions, " \
            "indicating smooth surfaces or the presence of obstacles; and the status of traffic lights " \
            "which influence your decision making, specifying whether they are red or green. " \
            "The description should be concise, providing an accurate understanding " \
            "of the driving environment to facilitate informed decision-making."


def draw_text_on_image(image, text, font_scale=1, font_thickness=2, line_spacing=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    img_height, img_width = image.shape[:2]
    max_text_width = img_width - 20 

    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + word + ' '
        test_size = cv2.getTextSize(test_line, font, font_scale, font_thickness)[0]
        if test_size[0] <= max_text_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + ' '
    
    lines.append(current_line.strip())
    total_text_height = (len(lines) * cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1]) + ((len(lines) - 1) * line_spacing)
    text_x = 10 
    text_y = img_height - total_text_height - 10
    
    for line in lines:
        cv2.putText(image, line, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        text_y += cv2.getTextSize(line, font, font_scale, font_thickness)[0][1] + line_spacing
    
    return image


def add_text_below_image(image, text):
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Choose font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    # Define margins
    margin = 10  # pixels
    max_text_width = img_width - 2 * margin
    
    # Function to wrap text into lines that fit within max_width
    def wrap_text(text, font, font_scale, font_thickness, max_width):
        words = text.split(' ')
        lines = []
        current_line = ''
        for word in words:
            # Build the test line
            if current_line:
                test_line = current_line + ' ' + word
            else:
                test_line = word
            # Get the size of the test_line
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
            if text_width <= max_width:
                # If the test_line fits, assign it to current_line
                current_line = test_line
            else:
                # Current line is full, add to lines
                lines.append(current_line)
                current_line = word  # Start new line
        # Add the last line
        if current_line:
            lines.append(current_line)
        return lines
    
    # Wrap text into lines
    lines = wrap_text(text, font, font_scale, font_thickness, max_text_width)
    
    # Get text height
    (line_width, line_height), _ = cv2.getTextSize('Test', font, font_scale, font_thickness)
    line_spacing = 5  # pixels between lines
    text_area_height = len(lines) * (line_height + line_spacing) + 2 * margin  # Add top and bottom margins
    
    # Create a new image with extra space at the bottom for text
    new_img_height = img_height + text_area_height
    new_image = np.ones((new_img_height, img_width, 3), dtype=np.uint8) * 255  # White background
    
    # Copy the original image into the new image
    new_image[0:img_height, 0:img_width] = image
    
    # Starting y position for text (after the image)
    y0 = img_height + margin + line_height
    for i, line in enumerate(lines):
        y = y0 + i * (line_height + line_spacing)
        # Put the text
        cv2.putText(new_image, line, (margin, y), font, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
    
    return new_image


def visualization(eval_data, tokenizer, model, image_processor, save_path):

    save_cnt = 1
    gap_cnt = 0
    save_tot_limit =100
    gap_interval = 2
    for sample in tqdm(eval_data[100:]):
        img_path = sample['image']
        question = sample['conversations'][0]['value']
        question = question.replace('\n<image>', '')
        question = question.replace('<image>\n', '')
        
        if 'SPEED plan' in question:
            gap_cnt = gap_cnt + 1
            if gap_cnt % gap_interval == 0:

                image = load_image(img_path)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                size = (1920, 1080)
                image = cv2.resize(image, size)
                size = (960, 540)
                image = cv2.resize(image, size)

                args = type('Args', (), {
                    "model_path": model_path,
                    "model_base": None,
                    "query": question,
                    "conv_mode": None,
                    "image_file": img_path,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 512
                })()

                answer = eval_model_wo_init(args, tokenizer, model, image_processor)
                answer = answer.replace('\n', '')

                font_scale = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_thickness = 3
                (text_width, text_height), baseline = cv2.getTextSize(answer, font, font_scale, font_thickness)
                text_x, text_y = 10, 10 + text_height
                cv2.rectangle(image, (text_x, text_y - text_height), (text_x + text_width, text_y + baseline), (0, 0, 0), thickness=cv2.FILLED)
                cv2.putText(image, answer, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

                desc_q = desc_question.replace('XXX', 'front camera')
                desc_q = '<image>\n' + desc_q
                args = type('Args', (), {
                    "model_path": model_path,
                    "model_base": None,
                    "query": desc_q,
                    "conv_mode": 'llava_v1',
                    "image_file": img_path,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 512
                })()

                answer = eval_model_wo_init(args, tokenizer, model, image_processor)
                answer = answer.replace('\n', '')
                image = add_text_below_image(image, answer)

                cv2.imwrite(f'{save_path}/{save_cnt}.jpg', image)
                save_cnt = save_cnt + 1

                if save_cnt == save_tot_limit:
                    break


parser = argparse.ArgumentParser(description='Senna visualization')
parser.add_argument(
    '--eval-data-path',
    type=str,
    default='nusc_plan_cmd_qa.json',
    help='specify the eval data path')
parser.add_argument(
    '--model-path',
    type=str,
    default='senna-llava-v1.5-7b',
    help='specify the model path')
parser.add_argument(
    '--save-path',
    type=str,
    default='./vis/',
    help='specify the image save path')
args = parser.parse_args()

if __name__ == '__main__':

    eval_data_path = args.eval_data_path
    model_path = args.model_path
    save_path = args.save_path

    tokenizer, model, image_processor, context_len = load_senna_pretrained_model(
        model_path, None, model_name="llava", device_map=0)

    with open(eval_data_path, 'r') as file:
        eval_data = json.load(file)

    visualization(eval_data, tokenizer, model, image_processor, save_path)


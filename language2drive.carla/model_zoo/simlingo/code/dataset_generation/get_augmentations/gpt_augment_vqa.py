"""
This script is used to augment the VQA dataset using GPT-4o.
It generates multiple variations of questions, answers, and objects by calling the OpenAI API.
The script reads the original dataset, processes the data to replace certain keywords with placeholders,
and then uses the OpenAI API to generate augmented data.
The augmented data is saved in JSON files for questions, answers, and objects.
Existing augmentations are loaded if available, and new augmentations are added to them.
"""

import numpy as np
import random
import json
import gzip
from multiprocessing import Pool
from openai import AzureOpenAI
import openai
from retry import retry
from collections import Counter
from pathlib import Path


def initialize_client():
    openai.api_key = "INSERT_YOUR_KEY"
    return openai

@retry(tries=5, delay=1, backoff=1, jitter=(0, 5), max_delay=10)
def call_chatgpt(client, chatgpt_messages, max_tokens=40, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens
    )
    reply = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    return reply, total_tokens

def prepare_chatgpt_message(prompt):
    system_message = "You are an expert in text editing and are helping in rephrasing the text so that the content always stays the same but the variety of the sentence is high.  Output as a list where each item is on its own line. Do not include any numbering, bullet points, dashes, or quotation marks. Only plain text, separated by newlines."
    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": "{}".format(prompt)})
    return messages

def gpt_forward(data):
    try:

        # Initialize the client inside the worker process
        client = initialize_client()

        prompts = ("Give me 20 different ways to say the following sentence. "
                   "Add the original sentence as the first item at the beginning of the list. "
                   "Keep the placeholders (text inside <>) as they are. "
                "Original sentence: " + data)

        messages = prepare_chatgpt_message(prompts)
        reply, total_tokens = call_chatgpt(client, messages, max_tokens=3000)
    except Exception as e:
        print(f"Error: {e}")
        return -1
    return reply

if __name__ == "__main__":
    
    import glob
    import re
    import tqdm
    
    dataset = 'all'
    as_augmented_path = 'data/augmented_templates/drivelm_train_augmented/all_as_augmented.json'
    qs_augmented_path = 'data/augmented_templates/drivelm_train_augmented/all_qs_augmented.json'
    objs_augmented_path = 'data/augmented_templates/drivelm_train_augmented/all_objs_augmented.json'
    
    locations = [
        'nearby to the front of the ego vehicle',
        'nearby to the front right of the ego vehicle',
        'nearby to the front left of the ego vehicle',
        'nearby on the left side of the ego vehicle',
        'far to the front of the ego vehicle',
        'far to the front right of the ego vehicle',
        'far to the front left of the ego vehicle',
        'far on the left side of the ego vehicle',
        'to the front of the ego vehicle',
        'to the front of it',
        'to the front right of the ego vehicle',
        'to the front left of the ego vehicle',
        'on the left side of the ego vehicle',
        'nearby',
        'far',
    ]
    
    save_path = f'data/{dataset}_augmented'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    path_to_graph = 'database/simlingo/drivelm/simlingo'
    
    # read all jsons
    all_files = glob.glob(f'{path_to_graph}/**/*.json.gz', recursive=True)
    # all_files = all_files[:1000]
    
    all_data = []
    all_questions = []
    all_answers = []
    all_objects = []
    failed_files = []
    for file in tqdm.tqdm(all_files):
        with gzip.open(file) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file}")
                failed_files.append(file)
                continue
            
            objects = [value['Visual_description'] for key, value in data['key_object_infos'].items()]
            
            questions = [data['QA'][types][i]['Q'] for types in data['QA'] for i in range(len(data['QA'][types]))]
            answers = [data['QA'][types][i]['A'] for types in data['QA'] for i in range(len(data['QA'][types]))]
            for object_type in objects:
                questions = [q.replace(object_type, '<OBJECT>') for q in questions]
                answers = [q.replace(object_type, '<OBJECT>') for q in answers]
            for location in locations:
                questions = [q.replace(location, '<LOCATION>') for q in questions]
                answers = [q.replace(location, '<LOCATION>') for q in answers]
                
            # find 'in XX m' and replace with '<DISTANCE>'
            questions = [re.sub(r'in \d+ m', 'in <DISTANCE>', q) for q in questions]
            answers = [re.sub(r'in \d+ m', 'in <DISTANCE>', q) for q in answers]

            all_questions.extend(questions)
            all_answers.extend(answers)
            all_objects.extend(objects)
            
            if len(all_answers) > 100000:
            
                print(f"Number of questions: {len(all_questions)}")
                print(f"Number of answers: {len(all_answers)}")
                print(f"Number of objects: {len(all_objects)}")
                print("Removing duplicates...")
                #remove duplicates
                all_questions = list(set(all_questions))
                all_answers = list(set(all_answers))
                all_objects = list(set(all_objects))
                print(f"Number of questions: {len(all_questions)}")
                print(f"Number of answers: {len(all_answers)}")
                print(f"Number of objects: {len(all_objects)}")
                
    print(f"Number of questions: {len(all_questions)}")
    print(f"Number of answers: {len(all_answers)}")
    print(f"Number of objects: {len(all_objects)}")
    print("Removing duplicates...")
    #remove duplicates
    all_questions = list(set(all_questions))
    all_answers = list(set(all_answers))
    all_objects = list(set(all_objects))
    print(f"Number of questions: {len(all_questions)}")
    print(f"Number of answers: {len(all_answers)}")
    print(f"Number of objects: {len(all_objects)}")
    
    if as_augmented_path is not None and Path(as_augmented_path).exists():
        print("Loading existing augmentation...")
        with open(as_augmented_path, 'r') as f:
            all_as_augmented = json.load(f)
    else:
        all_as_augmented = {}
    if qs_augmented_path is not None and Path(qs_augmented_path).exists():
        print("Loading existing augmentation...")
        with open(qs_augmented_path, 'r') as f:
            all_qs_augmented = json.load(f)
    else:
        all_qs_augmented = {}
    if objs_augmented_path is not None and Path(objs_augmented_path).exists():
        print("Loading existing augmentation...")
        with open(objs_augmented_path, 'r') as f:
            all_objs_augmented = json.load(f)
    else:
        all_objs_augmented = {}

    all_questions_augm = list(all_qs_augmented.keys())
    all_answers_augm = list(all_as_augmented.keys())
    all_objects_augm = list(all_objs_augmented.keys())
    print("Starting augmentation...")

    # remove strings in all_questions that are in all_questions_augm
    all_questions = [q for q in all_questions if q not in all_questions_augm]
    all_answers = [a for a in all_answers if a not in all_answers_augm and not "The important object" in a]
    all_objects = [o for o in all_objects if o not in all_objects_augm]
    print(f"Number of questions: {len(all_questions)}")
    print(f"Number of answers: {len(all_answers)}")
    print(f"Number of objects: {len(all_objects)}")

    print(f"Number of failed files: {len(failed_files)}")

    # breakpoint()


    all_qs = {}
    with Pool(8) as p:
        all_questions_augm = p.map(gpt_forward, all_questions)
    for i, q in enumerate(all_questions):
        a = all_questions_augm[i].replace("\n\n", "\n").split("\n")
        # remove leading and trailing whitespaces
        a = [x.strip() for x in a]
        all_qs[q] = a

    # add existing augmented questions to all_qs
    all_qs.update(all_qs_augmented)
    
    print("Saving...")
    with open(f'{save_path}/all_qs_augmented.json', 'w') as f:
        json.dump(all_qs, f, indent=4)
        
    print("Starting augmentation...")
    all_as = {}
    with Pool(8) as p:
        all_answers_augm = p.map(gpt_forward, all_answers)
    for i, a in enumerate(all_answers):
        a_tmp = all_answers_augm[i].replace("\n\n", "\n").split("\n")
        a_tmp = [x.strip() for x in a_tmp]
        all_as[a] = a_tmp

    # add existing augmented answers to all_as
    all_as.update(all_as_augmented)
    
    with open(f'{save_path}/all_as_augmented.json', 'w') as f:
        json.dump(all_as, f, indent=4)
        
    all_objs = {}
    with Pool(8) as p:
        all_objects = p.map(gpt_forward, all_objects)
    for i, o in enumerate(all_objects):
        a_tmp = all_objects[i].replace("\n\n", "\n").split("\n")
        a_tmp = [x.strip() for x in a_tmp]
        all_objs[o] = a_tmp
    
    # add existing augmented objects to all_objs
    all_objs.update(all_objs_augmented)
    
    print("Saving...")
    with open(f'{save_path}/all_objs_augmented.json', 'w') as f:
        json.dump(all_objs, f, indent=4)
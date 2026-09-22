"""
The augmentations for the subsentences of the commentaries were generated
manually using chatpgt. Since now the generation of the commentaty changed and we need
augmentations of the full sentences, we need to merge the augmentations of the subsentences.
This is done by this script.
"""


import glob
import gzip
import random
import ujson
from pathlib import Path
from tqdm import tqdm


save_path = 'data/augmented_templates'
path = 'database/simlingo/commentary'
augmented_sub_sentence_file = 'data/augmented_templates/commentary_subsentence.json'
all_files = glob.glob(path + '/**/*.json.gz', recursive=True)

print(f"Number of files: {len(all_files)}")

all_commentaries = []
for file in tqdm(all_files):

    with gzip.open(file, 'rt') as f:
        data = ujson.load(f)
        if 'commentary' in data:
            com_template = data['commentary_template']
            
            parent_folder = str(Path(file).parent)
            file_name = str(Path(file).name)
                
            if com_template not in all_commentaries:
                all_commentaries.append(com_template)   

print(len(all_commentaries))      

# get dictionary with subsentences and their augmentations
with open(augmented_sub_sentence_file, 'r') as f:
    augmented_sub_sentences = ujson.load(f)

augmentation_dict = {}
for key, value in augmented_sub_sentences.items():

    new_key = value[0]
    if new_key not in augmentation_dict:
        augmentation_dict[new_key] = value


# Augment all commentaries by looking for all new keys in all all_commentaries and replacing them with a randomly chosen augmentation
# generate 20 augmented commentaries for each original commentary
augmented_commentaries = {}
for com in tqdm(all_commentaries):
    for i in range(20):
        augmented_com = com
        for key, value in augmentation_dict.items():
            if key in augmented_com:
                augmented_com = augmented_com.replace(key, random.choice(value))
            elif key.lower() in augmented_com:
                augmented_com = augmented_com.replace(key.lower(), random.choice(value).lower())
        if com not in augmented_commentaries:
            augmented_commentaries[com] = [augmented_com]
        else:
            augmented_commentaries[com].append(augmented_com)


with open(f"{save_path}/commentary_augmented.json", 'w') as f:
    ujson.dump(augmented_commentaries, f, indent=4)
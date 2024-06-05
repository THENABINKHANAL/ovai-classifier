"""
Download datasets at https://huggingface.co/datasets/ovai-purdue/sea-creature-imgs/tree/main
inaturalist-2k, wikimedia-2k, and yahoo-2k contain images for the ~2000 concepts from Fathomnet scraped from iNaturalist, Wikimedia, and Yahoo Images respectively

Unzip the dataset into file structure:
dataset/
    train/
        inaturalist/
        wikimedia/
        yahoo/

- inaturalist contains more accurate photos, but doesn't cover all concepts from Fathomnet
- wikimedia contains many images of text and diagrams instead of photos of the creature
- yahoo contains images that are not of the creature, but it's ordered by relevance

Set DATASETS to the list of datasets you want to use
Run this script in the dataset/ directory to update label_counts.json, labels.json, train_labels.json
"""

#DATASETS = ['inaturalist', 'wikimedia', 'yahoo']
DATASETS = ['inaturalist']
ID_START = 3000000
EXCLUDED_LABELS = ['test']


import os
import json


with open('label_templates/label_counts.json') as f:
    label_counts = json.load(f)

with open('label_templates/labels.json') as f:
    labels = json.load(f)

with open('label_templates/train_labels.json') as f:
    train_labels = json.load(f)


cur_id = ID_START
for dataset in DATASETS:
    basedir = os.path.join('train', dataset)
    for label in os.listdir(basedir):
        label = label.replace(' sp ', ' sp. ')
        if label.endswith(' sp'):
            label = label + '.'
            
        if label in EXCLUDED_LABELS:
            continue
            
        key = label.upper()
        if key not in labels:
            continue
        
        imgs = os.listdir(os.path.join(basedir, label))
        if key not in label_counts:
            label_counts[key] = 0
        label_counts[key] = label_counts[key] + len(imgs)
        
        for img in imgs:
            train_labels.append(
                {
                    "id": cur_id,
                    "path": "./dataset/train/"+dataset+'/'+label+'/'+img,
                    "label": label
                }
            )
            cur_id = cur_id+1
        

with open("label_counts.json", "w") as outfile:
    json.dump(label_counts, outfile, sort_keys=True, indent=4)

with open("labels.json", "w") as outfile:
    json.dump(labels, outfile, sort_keys=True, indent=4)

with open("train_labels.json", "w") as outfile:
    json.dump(train_labels, outfile, sort_keys=True, indent=4)


import os
import json
import random

DATASETS = ['inaturalist', 'wikimedia', 'yahoo']
ID_START = 3000000
EXCLUDED_LABELS = ['test']

with open('./dataset/label_counts.json') as f:
    label_counts = json.load(f)

with open('./dataset/labels.json') as f:
    labels = json.load(f)

with open('./dataset/train_labels.json') as f:
    train_labels = json.load(f)

with open('./dataset/test_labels.json') as f:
    test_labels = json.load(f)

def list_files_in_directory(directory):
    try:
        files_and_directories = os.listdir(directory)
        files = [f for f in files_and_directories if os.path.isfile(os.path.join(directory, f))]
        return files
    except FileNotFoundError:
        return "Directory not found."
    except Exception as e:
        return str(e)

cur_id = ID_START
for dataset in DATASETS:
    basedir = os.path.join('./dataset/train', dataset)
    for dirLabel in os.listdir(basedir):
        label = dirLabel.replace(' sp ', ' sp. ')
        if label.endswith(' sp'):
            label = label + '.'
        if label in EXCLUDED_LABELS:
            continue

        label = label.upper()
        if label not in label_counts:
            print(label)
            continue
        files = list_files_in_directory("./dataset/train/" + dataset + '/' + dirLabel)
        label_counts[label] = label_counts[label] + len(files)

        random.shuffle(files)

        split_index = int(len(files) * 0.8)
        train_files = files[:split_index]
        test_files = files[split_index:]

        for file in train_files:
            train_labels.append(
                {
                    "id": cur_id,
                    "path": "./dataset/train/" + dataset + '/' + dirLabel + '/' + file,
                    "label": label.upper()
                }
            )
            cur_id += 1

        for file in test_files:
            test_labels.append(
                {
                    "id": cur_id,
                    "path": "./dataset/train/" + dataset + '/' + label + '/' + file,
                    "label": label.upper()
                }
            )
            cur_id += 1

with open("./dataset/label_counts2.json", "w") as outfile:
    json.dump(label_counts, outfile, sort_keys=True, indent=4)

with open("./dataset/labels2.json", "w") as outfile:
    json.dump(labels, outfile, sort_keys=True, indent=4)

with open("./dataset/train_labels2.json", "w") as outfile:
    json.dump(train_labels, outfile, sort_keys=True, indent=4)

with open("./dataset/test_labels2.json", "w") as outfile:
    json.dump(test_labels, outfile, sort_keys=True, indent=4)

import pymssql
import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import json
import random
import shutil

FOLDER_DIRECTORY = "./images"
TRAIN_FOLDER = "./dataset/train"
TEST_FOLDER = "./dataset/test"
JSON_FILE_TRAIN = "./dataset/train_labels.json"
JSON_FILE_TEST = "./dataset/test_labels.json"

# Modify the connection details as per your configuration
connection = pymssql.connect(server="128.46.81.40", user="forbeslab", password="r^%o*3zMzsgzSxADi9", database="Fathom")
cursor = connection.cursor()

# Fetch all IDs from the database
cursor.execute("SELECT id FROM bounding_boxes ORDER BY id")
all_ids = [row[0] for row in cursor.fetchall()]

# Shuffle IDs for random split
random.shuffle(all_ids)

# Split IDs for training and testing
split_idx = int(len(all_ids) * 0.8)  # 80% for training, 20% for testing
train_ids = all_ids[:split_idx]
test_ids = all_ids[split_idx:]

print(len(train_ids))

def process_and_save_images(ids, folder, json_file):
    BATCH_SIZE = 10000  # Process 10,000 IDs at a time
    data_for_json = []

    # Function to process each batch
    def process_batch(batch_ids):
        # Convert list of IDs to a format suitable for the SQL query (e.g., (id1, id2, ...))
        ids_format = ','.join(map(str, batch_ids))
        query = f"SELECT id, image_id, concept, x, y, width, height FROM bounding_boxes WHERE id IN ({ids_format})"
        cursor.execute(query)
        rows = cursor.fetchall()

        for row in rows:
            bbox_id, image_id, concept, x, y, width, height = row
            image_path = os.path.join(FOLDER_DIRECTORY, f"{image_id}.png")
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    roi = image.crop((x, y, x + width, y + height))
                    save_path = os.path.join(folder, f"{bbox_id}.png")
                    roi.save(save_path)
                    data_for_json.append({"id": bbox_id, "path": save_path, "label": concept})
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")

    # Iterate over the IDs in batches
    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        process_batch(batch_ids)

    # Save the collected data to a JSON file
    with open(json_file, 'w') as f:
        json.dump(data_for_json, f, indent=4)


# Ensure the dataset folders exist
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# Process and save training and testing images, and generate JSON files
process_and_save_images(train_ids, TRAIN_FOLDER, JSON_FILE_TRAIN)
process_and_save_images(test_ids, TEST_FOLDER, JSON_FILE_TEST)

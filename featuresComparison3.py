import pymssql
import os
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_L_Weights  # Import EfficientNet V2 L Weights
from PIL import Image
import sqlite3  # Import SQLite library
from scipy.spatial.distance import cosine
import gc  # Garbage Collector interface
import time
import datetime

FOLDER_DIRECTORY = "./images"

# Use EfficientNet V2 L with optional pretrained weights
model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize database connection
connection = pymssql.connect(
            server="128.46.81.40",
            user="forbeslab",
            password="r^%o*3zMzsgzSxADi9",
            database="Fathom"
)
cursor = connection.cursor()

# Fetch all IDs from the database
cursor.execute("SELECT id FROM bounding_boxes ORDER BY id")
all_ids = [row[0] for row in cursor.fetchall()]

total_ids = len(all_ids)
totalTime = total_ids *(total_ids-1)/2

grid_size = (total_ids + 1) * (total_ids + 1)
float_size = 4  # Size of a float in bytes
file_path = './similarity_scores.bin'

def getDifference(elapsed_time):
    days, remainder = divmod(elapsed_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"Remaining time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

with open(file_path, 'wb') as file:
    zeros_per_chunk = 100000  # Number of zeros to write in each chunk
    chunk_size = zeros_per_chunk * float_size  # Size of each chunk in bytes
    total_chunks = grid_size // zeros_per_chunk  # Total number of full chunks
    remaining = grid_size % zeros_per_chunk  # Remaining floats to write after the full chunks

    for _ in range(total_chunks):
        file.write(bytearray(chunk_size))
    
    if remaining > 0:
        file.write(bytearray(remaining * float_size))

    file.seek(0)
    file.write(np.float32(0).tobytes())

    for id in all_ids:
        file.write(np.float32(id).tobytes())

    for i in range(total_ids):
        file.seek((i + 1) * (total_ids + 1) * float_size)
        file.write(np.float32(all_ids[i]).tobytes())

    def save_similarity_score(index1, index2, score):
        
        offset = ((index1 + 1) * (total_ids + 1) + (index2 + 1)) * float_size
        
        file.seek(offset)
        file.write(np.float32(score).tobytes())

    def batch_extract_features(image_paths, bboxes):
        batch_tensors = []
        feature_vectors = [None] * len(image_paths)

        for img_path, bbox in zip(image_paths, bboxes):
            try:
                image = Image.open(img_path).convert('RGB')
                roi = image.crop(bbox)
                tensor = preprocess(roi).unsqueeze(0)
                batch_tensors.append(tensor)
            except:
                print(f"Cannot identify image file {img_path}")
                batch_tensors.append(None)

        
        if not any(t is not None for t in batch_tensors):
            return feature_vectors

        valid_tensors = torch.cat([t for t in batch_tensors if t is not None], 0).to(device)
        
        with torch.no_grad():
            batch_features = model(valid_tensors)
            
        batch_features = batch_features.cpu()
        features_split = batch_features.split(1, 0)

        j = 0
        for i, tensor in enumerate(batch_tensors):
            if tensor is not None:
                feature_vectors[i] = features_split[j].squeeze(0)
                j += 1
        
        return feature_vectors

    BATCH_SIZE = 5000

    query_template = "SELECT id, image_id, x, y, width, height FROM bounding_boxes ORDER BY id OFFSET {} ROWS FETCH NEXT {} ROWS ONLY"

    def process_images_in_batch(all_ids):
        total_ids = len(all_ids)
        for start_idx in range(0, total_ids, BATCH_SIZE):
            start_time = time.time()
            query = query_template.format(start_idx, BATCH_SIZE)
            cursor.execute(query)
            rows = cursor.fetchall()

            image_paths = []
            bboxes = []
            bounding_box_ids1 = []

            for i, row in enumerate(rows):
                bounding_box_id, image_id, x, y, width, height = row
                filename = os.path.join(FOLDER_DIRECTORY, f"{image_id}.png")
                if not os.path.exists(filename) or width <= 0 or height <= 0:
                    print(f"Skipping bounding_box_id {bounding_box_id}.")
                    bounding_box_ids1.append(bounding_box_id)
                    image_paths.append(None)
                    continue
                
                image_paths.append(filename)
                bboxes.append((x, y, x + width, y + height))
                bounding_box_ids1.append(bounding_box_id)
            feature_vectors1 = batch_extract_features(image_paths, bboxes)
            print(start_idx, start_idx)


            for x in range(len(feature_vectors1)):
                for y in range(len(feature_vectors1)):
                    if feature_vectors1[x] is not None and feature_vectors1[y] is not None:
                        sim = 1-cosine(feature_vectors1[x], feature_vectors1[y])
                        save_similarity_score(start_idx + x,start_idx + y,sim)

            end_time = time.time()
            timeDiff = end_time - start_time
            curN = (start_idx)*total_ids+1
            print(getDifference((totalTime - curN)/BATCH_SIZE*timeDiff))

            gc.collect()

            for start_idx1 in range(start_idx+BATCH_SIZE, total_ids, BATCH_SIZE):
                start_time = time.time()

                query = query_template.format(start_idx1, BATCH_SIZE)
                cursor.execute(query)
                rows2 = cursor.fetchall()

                image_paths = []
                bboxes = []
                bounding_box_ids2 = []

                for ij, row in enumerate(rows2):
                    bounding_box_id, image_id, x, y, width, height = row
                    filename = os.path.join(FOLDER_DIRECTORY, f"{image_id}.png")
                    if not os.path.exists(filename) or width <= 0 or height <= 0:
                        print(f"Skipping bounding_box_id {bounding_box_id}.")
                        image_paths.append(None)
                        bounding_box_ids2.append(bounding_box_id)
                        continue
                    
                    image_paths.append(filename)
                    bboxes.append((x, y, x + width, y + height))
                    bounding_box_ids2.append(bounding_box_id)
                feature_vectors2 = batch_extract_features(image_paths, bboxes)

                for x in range(len(feature_vectors1)):
                    for y in range(len(feature_vectors2)):
                        if feature_vectors1[x] is not None and feature_vectors2[y] is not None:
                            sim = 1-cosine(feature_vectors1[x], feature_vectors2[y])
                            #ws.cell(row=start_idx + x + 2, column=start_idx1 + y + 2, value=sim)
                            #ws.cell(row=start_idx1 + y + 2, column=start_idx + x + 2, value=sim)
                            save_similarity_score(start_idx + x,start_idx1 + y,sim)
                            save_similarity_score(start_idx1 + y, start_idx + x,sim)

                end_time = time.time()
                timeDiff = end_time - start_time
                curN = (start_idx)*total_ids+start_idx1+1
                print(getDifference((totalTime - curN)/BATCH_SIZE*timeDiff))
                gc.collect()



    process_images_in_batch(all_ids)


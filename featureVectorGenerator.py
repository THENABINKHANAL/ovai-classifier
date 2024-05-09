import pymssql
import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

# Constants
BATCH_SIZE = 10000
VECTOR_SIZE = 1000  # Assuming the feature vector size is 1000
FOLDER_DIRECTORY = "./images"
IMAGE_BATCH_SIZE = 512  # Adjust based on your GPU memory

# Database connection
connection = pymssql.connect(
            server="127.0.0.1",
            database="Fathom",
)
connection.autocommit(True)
cursor = connection.cursor()

# Load EfficientNet-B7 pre-trained model
model = models.efficientnet_b7(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the cropped bounding box image to the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def batch_extract_features(image_paths, bboxes):
    batch_tensors = []
    feature_vectors = [None] * len(image_paths)  # Initialize the list with None

    for img_path, bbox in zip(image_paths, bboxes):
        try:
            image = Image.open(img_path).convert('RGB')
            roi = image.crop(bbox)
            tensor = preprocess(roi).unsqueeze(0)  # Add a batch dimension
            batch_tensors.append(tensor)
        except UnidentifiedImageError:
            print(f"Cannot identify image file {img_path}")
            batch_tensors.append(None)  # Append None for failed image reads
    
    if not any(t is not None for t in batch_tensors):
        return feature_vectors  # Return early if no valid tensors

    valid_tensors = torch.cat([t for t in batch_tensors if t is not None], 0)
    
    with torch.no_grad():
        batch_features = model(valid_tensors)
    
    features_split = batch_features.split(1, 0)  # Split the batched tensor into individual tensors

    # Align the extracted features with the original sequence, maintaining order and length
    j = 0  # Index for features_split
    for i, tensor in enumerate(batch_tensors):
        if tensor is not None:
            feature_vectors[i] = features_split[j].squeeze(0)
            j += 1
    
    return feature_vectors


def update_magnitude(bounding_box_id, magnitude):
    try:
        magnitude_value = float(magnitude)  # Convert to a float
        cursor.execute(
            "UPDATE bounding_boxes SET magnitude = %s WHERE id = %s",
            (magnitude_value, bounding_box_id)
        )
        connection.commit()
    except Exception as e:
        print(f"Error updating magnitude for bounding_box_id {bounding_box_id}: {e}")

def update_feature_vectors(batch_data):
    try:
        cursor.executemany(
            """
            INSERT INTO bounding_box_image_feature_vectors (bounding_box_id, vector_index, vector_value) 
            VALUES (%s, %s, %s)
            """,
            batch_data
        )
        connection.commit()
    except Exception as e:
        print(f"Error updating feature vectors: {e}")


def update_database(bounding_box_ids, feature_vectors):
    batch_data = []
    for bounding_box_id, feature_vector in zip(bounding_box_ids, feature_vectors):
        if feature_vector is None:
            continue
        feature_vector = feature_vector.cpu().numpy().flatten()
        magnitude = np.linalg.norm(feature_vector)
        update_magnitude(bounding_box_id, magnitude)
        for i in range(VECTOR_SIZE):
            batch_data.append((bounding_box_id, i + 1, float(feature_vector[i])))
    update_feature_vectors(batch_data)

def process_rows(rows):
    image_paths = []
    bboxes = []
    bounding_box_ids = []

    for row in rows:
        bounding_box_id, image_id, x, y, width, height = row
        filename = os.path.join(FOLDER_DIRECTORY, f"{image_id}.png")
        if not os.path.exists(filename) or width <= 0 or height <= 0:
            print(f"Skipping bounding_box_id {bounding_box_id} due to missing file or invalid dimensions.")
            continue
        
        image_paths.append(filename)
        bboxes.append((x, y, x + width, y + height))
        bounding_box_ids.append(bounding_box_id)
    
    feature_vectors = batch_extract_features(image_paths, bboxes)
    update_database(bounding_box_ids, feature_vectors)

# Main processing loop
offset = 0
query = f"SELECT id, image_id, x, y, width, height FROM bounding_boxes ORDER BY id OFFSET {offset} ROWS FETCH NEXT {BATCH_SIZE} ROWS ONLY"

while True:
    cursor.execute(query)
    rows = cursor.fetchall()
    if not rows:
        break  # No more rows to fetch
    for i in range(0, len(rows), IMAGE_BATCH_SIZE):
        batch_rows = rows[i:i + IMAGE_BATCH_SIZE]
        process_rows(batch_rows)
    offset += BATCH_SIZE
    query = f"SELECT id, image_id, x, y, width, height FROM bounding_boxes ORDER BY id OFFSET {offset} ROWS FETCH NEXT {BATCH_SIZE} ROWS ONLY"

# Close the cursor and connection
cursor.close()
connection.close()

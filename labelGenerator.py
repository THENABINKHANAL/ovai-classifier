import pymssql
import json

# Define the path for the labels JSON file
JSON_FILE_TRAIN = "./dataset/labels.json"

# Establish database connection
connection = pymssql.connect(server="128.46.81.40", user="forbeslab", password="r^%o*3zMzsgzSxADi9", database="Fathom")
cursor = connection.cursor()

# Execute SQL query to fetch distinct concepts
cursor.execute("SELECT DISTINCT concept FROM bounding_boxes ORDER BY concept")
concepts = [row[0] for row in cursor.fetchall()]

# Create a mapping from concept to integer index
concept_to_idx = {concept.upper(): idx for idx, concept in enumerate(concepts)}

# Ensure the directory exists (optional)
import os
os.makedirs(os.path.dirname(JSON_FILE_TRAIN), exist_ok=True)

# Open the JSON file in write mode and dump the mapping
with open(JSON_FILE_TRAIN, 'w') as file:
    json.dump(concept_to_idx, file, indent=4)

# Close cursor and connection
cursor.close()
connection.close()

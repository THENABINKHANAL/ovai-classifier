
import pymssql
import requests
import os
import time

# Constants
BATCH_SIZE = 10000
FOLDER_DIRECTORY = "./images"
DELAY_AFTER_EACH_FILE = 1  # seconds
DELAY_AFTER_BATCH = 40  # seconds

# Ensure the folder exists
os.makedirs(FOLDER_DIRECTORY, exist_ok=True)
# Database connection
connection = pymssql.connect(
            server="127.0.0.1",
            user="forbeslab",
            password="r^%o*3zMzsgzSxADi9",
            database="Fathom"
)
cursor = connection.cursor()

# Function to download and save an image
def download_image(url, filename, image_id):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(filename, 'wb') as file:
            file.write(response.content)
    except Exception as e:
        print(image_id)


# Function to process rows
def process_rows(rows):
    for row in rows:
        image_id = row[0]
        image_url = row[1]
        filename = os.path.join(FOLDER_DIRECTORY, f"{image_id}.png")
        download_image(image_url, filename, image_id)
        time.sleep(DELAY_AFTER_EACH_FILE)  # Wait for 1s after each file retrieval

# Modify the query to fetch rows in chunks
offset = 0
query = f"SELECT id, url FROM images ORDER BY id OFFSET {offset} ROWS FETCH NEXT {BATCH_SIZE} ROWS ONLY"

while True:
    cursor.execute(query)
    rows = cursor.fetchall()
    if not rows:
        break  # No more rows to fetch
    process_rows(rows)
    offset += BATCH_SIZE
    query = f"SELECT id, url FROM images ORDER BY id OFFSET {offset} ROWS FETCH NEXT {BATCH_SIZE} ROWS ONLY"
    time.sleep(DELAY_AFTER_BATCH)  # Wait for 40s after 10000 files retrieval

# Close the cursor and connection
cursor.close()
connection.close()

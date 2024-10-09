import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
import os
import pytesseract
import pandas as pd
from sklearn.metrics import f1_score
import concurrent.futures
import cupy as cp

# Persistent session for requests
session = requests.Session()
count=0
# Function to download a batch of images
def download_images_batch(image_urls):
    def download_image(image_url):
        os.system("cls")
        global count
        print(f"Image:{count}")
        count+=1
        try:
            response = session.get(image_url, timeout=20)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        image_contents = list(executor.map(download_image, image_urls))
    return image_contents

# Function to preprocess a batch of images
def preprocess_images_batch(image_contents):
    preprocessed_images = []
    for content in image_contents:
        if content is None:
            preprocessed_images.append(None)
            continue
        try:
            # Open image using PIL
            img = Image.open(BytesIO(content))
            # Convert to NumPy array
            img_array = np.array(img)

            # Handle different image modes
            if img_array.ndim == 3:  # RGB or RGBA
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            elif img_array.ndim == 2:  # Grayscale
                gray = img_array
            else:
                raise ValueError("Unsupported image format")

            # Convert to CuPy array for GPU acceleration
            gray_cupy = cp.asarray(gray)
            # Apply binary thresholding using CuPy
            _, thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)

            preprocessed_images.append(thresh)  # Append CuPy array
        except Exception as e:
            print(f"Error processing image: {e}")
            preprocessed_images.append(None)
    
    return preprocessed_images

# Function to perform OCR and extract text from a batch of images
def extract_texts_batch(preprocessed_images):
    extracted_texts = []
    for image in preprocessed_images:
        if image is None:
            extracted_texts.append("")
            continue
        try:
            # Convert CuPy array to NumPy for OCR processing
            image_np = cp.asnumpy(image)
            raw_text = pytesseract.image_to_string(image_np, config='--psm 6')
            # Clean the extracted text
            cleaned_text = re.sub(r'[^\w\s.]+', '', raw_text)
            extracted_texts.append(cleaned_text)
        except Exception as e:
            print(f"Error extracting text: {e}")
            extracted_texts.append("")
    
    return extracted_texts
# Mapping entity to units (same as before)
entity_unit_map = {
    'centimetre': 'centimetre', 'foot': 'foot', 'inch': 'inch', 'metre': 'metre',
    'millimetre': 'millimetre', 'yard': 'yard', 'gram': 'gram', 'kilogram': 'kilogram',
    'microgram': 'microgram', 'milligram': 'milligram', 'ounce': 'ounce', 'pound': 'pound',
    'ton': 'ton', 'kilovolt': 'kilovolt', 'millivolt': 'millivolt', 'volt': 'volt',
    'kilowatt': 'kilowatt', 'watt': 'watt', 'centilitre': 'centilitre', 'cubic foot': 'cubic foot',
    'cubic inch': 'cubic inch', 'cup': 'cup', 'decilitre': 'decilitre', 'fluid ounce': 'fluid ounce',
    'gallon': 'gallon', 'imperial gallon': 'imperial gallon', 'litre': 'litre', 'microlitre': 'microlitre',
    'millilitre': 'millilitre', 'pint': 'pint', 'quart': 'quart', 'cm': 'centimetre', 'kg': 'kilogram',
    'g': 'gram', 'ml': 'millilitre', 'l': 'litre', 'oz': 'ounce', 'ft': 'foot', 'in': 'inch',
    'yd': 'yard', 'lbs': 'pound', 'kW': 'kilowatt', 'V': 'volt', 'gal': 'gallon', 'pt': 'pint', 'qt': 'quart'
}
# Parse entity value from text
def parse_entity_value(text, entity_name):
    pattern = r'(\d*\.?\d+)\s*([a-zA-Z]+)'  # Regex to find values and units
    matches = re.findall(pattern, text)
    
    for value, unit in matches:
        unit = unit.lower()
        if unit in entity_unit_map:
            return f"{value} {entity_unit_map.get(unit)}"
    
    return ""  # Return empty string if no valid match is found

# Process a batch of rows
def process_batch(batch):
    image_urls = batch['image_link'].tolist()
    entity_names = batch['entity_name'].tolist()

    # Step 1: Download the batch of images
    image_contents = download_images_batch(image_urls)

    # Step 2: Preprocess the batch of images
    preprocessed_images = preprocess_images_batch(image_contents)

    # Step 3: Extract text using OCR for the batch of images
    extracted_texts = extract_texts_batch(preprocessed_images)

    # Step 4: Parse entity values from extracted text
    predictions = [parse_entity_value(text, entity_name) for text, entity_name in zip(extracted_texts, entity_names)]

    # Return predictions as a DataFrame
    return pd.DataFrame({'index': batch['index'], 'prediction': predictions})

# Predict entity values in batches
def predict_entity_values_batch(test_data, batch_size=100, max_workers=8):
    results = []
    num_batches = (len(test_data) + batch_size - 1) // batch_size  # Calculate total number of batches
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process each batch in parallel
        futures = [
            executor.submit(process_batch, test_data.iloc[i * batch_size: (i + 1) * batch_size])
            for i in range(num_batches)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # Concatenate all the results into a single DataFrame
    return pd.concat(results, ignore_index=True)

# Load test data (adjust batch size or sampling for testing)
test_data = pd.read_csv("dataset/train.csv")
test_data['index'] = range(0, len(test_data))
test_data=test_data.iloc[:200]

# Predict in parallel with batch processing
batch_size = 50  # Set appropriate batch size
output = predict_entity_values_batch(test_data, batch_size=batch_size, max_workers=16)
#output['index']=test_data['index'].to_numpy()

# Save to CSV
output.to_csv('test_out.csv', index=False)

#Uncomment to calculate F1 score if needed
y_true = test_data['entity_value'].tolist()  # Replace with correct column name
y_pred = output['prediction'].tolist()
f1 = f1_score(y_true, y_pred, average='micro')  # Choose appropriate average
print(f"F1 Score:{f1:.6f}")
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
import pytesseract
import pandas as pd
from sklearn.metrics import f1_score
import concurrent.futures

# Persistent session for requests
session = requests.Session()

def preprocess_image(image_url):
    try:
        response = session.get(image_url, timeout=10)  # Set a timeout for image download
        response.raise_for_status()  # Check for any request issues

        # Open image using PIL
        img = Image.open(BytesIO(response.content))
        # Convert the image to a NumPy array
        img_array = np.array(img)

        # Handle different image modes
        if img_array.ndim == 3:  # RGB or RGBA
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        elif img_array.ndim == 2:  # Grayscale
            gray = img_array
        else:
            raise ValueError("Unsupported image format")

        # Apply binary thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return thresh
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None  # Return None if there's an issue

def extract_text_from_image(image):
    config = '--psm 6'  # Set OCR config to assume a single uniform block of text
    raw_text = pytesseract.image_to_string(image, config=config)
    # Clean up unwanted characters
    cleaned_text = re.sub(r'[^\w\s.]+', '', raw_text)
    return cleaned_text

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

def parse_entity_value(text, entity_name):
    # Regex to find values and units
    pattern = r'(\d*\.?\d+)\s*([a-zA-Z]+)'
    matches = re.findall(pattern, text)
    
    for value, unit in matches:
        unit = unit.lower()
        if unit in entity_unit_map:
            return f"{value} {entity_unit_map.get(unit)}"
    
    return ""  # Return empty string if no valid match is found

def process_row(row):
    image_url = row['image_link']
    entity_name = row['entity_name']
    
    # Preprocess the image
    image = preprocess_image(image_url)
    
    if image is not None:
        # Extract text using OCR
        extracted_text = extract_text_from_image(image)
        
        # Parse entity value from extracted text
        prediction = parse_entity_value(extracted_text, entity_name)
    else:
        prediction = ""  # Return empty string if the image couldn't be processed
    
    return {'index': row['index'], 'prediction': prediction}

def predict_entity_values_parallel(test_data, max_workers=8):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_row, [row for _, row in test_data.iterrows()]))
    return pd.DataFrame(results)

# Load test data (adjust the batch size or sampling for testing)
test_data = pd.read_csv('dataset/test.csv')
test_data['index'] = range(0, len(test_data))
#test_data=test_data.iloc[:20]

# Predict in parallel
output = predict_entity_values_parallel(test_data, max_workers=32)  # Adjust workers based on CPU

# Save to CSV
output.to_csv('test_out.csv', index=False)

# Calculate F1 score
'''y_true = test_data['entity_value'].tolist()  # Replace with correct column name
y_pred = output['prediction'].tolist()
f1 = f1_score(y_true, y_pred, average='micro')  # Choose appropriate average
print(f"F1 Score: {f1:.6f}")'''

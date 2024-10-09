import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
from sklearn.metrics import f1_score
def preprocess_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)  # Set a timeout for image download
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
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        return thresh
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None  # Return None if there's an issue

import pytesseract

def extract_text_from_image(image):
    # OCR to extract text
    raw_text = pytesseract.image_to_string(image)
    # Clean up unwanted characters
    cleaned_text = re.sub(r'[^\w\s.]+', '', raw_text)
    #print(cleaned_text)
    return cleaned_text

# Map entity to the allowed units
'''entity_unit_map = {
  "width": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "depth": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "height": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "item_weight": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
  "maximum_weight_recommendation": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
  "voltage": {"millivolt", "kilovolt", "volt"},
  "wattage": {"kilowatt", "watt"},
  "item_volume": {"cubic foot", "microlitre", "cup", "fluid ounce", "centilitre", "imperial gallon", 
                  "pint", "decilitre", "litre", "millilitre", "quart", "cubic inch", "gallon"}
                  
}'''
entity_unit_map={
    'centimetre': 'centimetre',
    'foot': 'foot',
    'inch': 'inch',
    'metre': 'metre',
    'millimetre': 'millimetre',
    'yard': 'yard',
    'gram': 'gram',
    'kilogram': 'kilogram',
    'microgram': 'microgram',
    'milligram': 'milligram',
    'ounce': 'ounce',
    'pound': 'pound',
    'ton': 'ton',
    'kilovolt': 'kilovolt',
    'millivolt': 'millivolt',
    'volt': 'volt',
    'kilowatt': 'kilowatt',
    'watt': 'watt',
    'centilitre': 'centilitre',
    'cubic foot': 'cubic foot',
    'cubic inch': 'cubic inch',
    'cup': 'cup',
    'decilitre': 'decilitre',
    'fluid ounce': 'fluid ounce',
    'gallon': 'gallon',
    'imperial gallon': 'imperial gallon',
    'litre': 'litre',
    'microlitre': 'microlitre',
    'millilitre': 'millilitre',
    'pint': 'pint',
    'quart': 'quart',
    'cm': 'centimetre',
    'kg': 'kilogram',
    'g': 'gram',
    'ml': 'millilitre',
    'l': 'litre',
    'oz': 'ounce',
    'ft': 'foot',
    'in': 'inch',
    'yd': 'yard',
    'lbs': 'pound',
    'kW': 'kilowatt',
    'V': 'volt',
    'gal': 'gallon',
    'pt': 'pint',
    'qt': 'quart'
}

def parse_entity_value(text, entity_name):
    # Regex to find values and units
    pattern = r'(\d*\.?\d+)\s*([a-zA-Z]+)'
    matches = re.findall(pattern, text)
    
    for value, unit in matches:
        unit = unit.lower()
        #print(unit)
        # Check if the unit is valid for the entity
       # if unit in entity_unit_map.get(entity_name, set()):
            #return f"{value} {unit}"
        if unit in entity_unit_map:
            return f"{value} {entity_unit_map.get(unit)}"
    
    return ""  # Return empty string if no valid match is found
import pandas as pd

# Load test data
test_data = pd.read_csv('dataset/train.csv')

test_data['index'] = range(0,263859)
test_data=test_data.iloc[:200]
# Function to predict entity value for each image in test data
def predict_entity_values(test_data):
    predictions = []
    
    for idx, row in test_data.iterrows():
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
        
        # Append result to the list
        predictions.append({'index': row['index'], 'prediction': prediction})
    
    return pd.DataFrame(predictions)


# Run predictions
output = predict_entity_values(test_data)

# Save to CSV
output.to_csv('test_out.csv', index=False)
 # Assuming 'true_entity_values' column contains the true labels in test_data
y_true = test_data['entity_value'].tolist()  # Replace with the correct column name
y_pred = output['prediction'].tolist()
# Calculate the F1 score
f1 = f1_score(y_true, y_pred, average='micro')  # Choose appropriate average
print(f"F1 Score: {f1:.6f}")
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import regex as re

def preprocess_image(image):
    # Convert to grayscale
    width = int(image.shape[1] * 50 / 100)
    height = int(image.shape[0] * 50 / 100)
    dimensions = (width, height)
    rescaled_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    return thresh

def extract_text_from_image(image):
    # Configuring Tesseract for OCR
    config = '--psm 6'  # Assume a uniform block of text
    
    # Extract text from the image using pytesseract
    raw_text = pytesseract.image_to_string(image, config=config)
    
    # Clean up unwanted characters using regex
    cleaned_text = re.sub(r'[^\w\s.]+', '', raw_text)
    
    return cleaned_text

def find_and_extract_text_from_contours(image):
    # Preprocess the image to get a thresholded version
    thresholded_image = preprocess_image(image)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_texts = []

    # Loop over the contours
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out very small or very large contours
        if w>30 and h>30:  # You can tune these values based on the image
            # Extract the region of interest (ROI) from the image
            roi = thresholded_image[y:y+h, x:x+w]

            # Apply OCR on the ROI
            text = extract_text_from_image(roi)

            # Append extracted text to the list if it contains any valid content
            if text.strip():
                extracted_texts.append(text)

            # Optionally draw the rectangle around the contour (for visualization)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, extracted_texts

# Example usage
image_path = 'dataset/img.jpg'
image = cv2.imread(image_path)

# Find contours and extract text from them
contoured_image, texts = find_and_extract_text_from_contours(image)

# Display the original image with contours drawn
plt.imshow(cv2.cvtColor(contoured_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Contours')
plt.show()

# Print extracted text
print("Extracted Texts from Contours:")
for idx, text in enumerate(texts):
    print(f"Text {idx + 1}: {text}")

import cv2
import numpy as np

# Load the background image
background_image = cv2.imread("background1.jpg")

# Read the input image
input_image = cv2.imread("input.jpg")

# Convert the input image to grayscale
grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Segment the foreground object from the input image
foreground_mask = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY_INV)[1]

# Apply morphological operations to clean up the foreground mask
foreground_mask = cv2.dilate(foreground_mask, None, iterations=2)
foreground_mask = cv2.erode(foreground_mask, None, iterations=2)

# Extract the foreground object
foreground_object = cv2.bitwise_and(input_image, input_image, mask=foreground_mask)

# Paste the foreground object onto the background image
output_image = background_image.copy()
output_image[foreground_mask != 0] = foreground_object[foreground_mask != 0]

# Save the output image
cv2.imwrite("output.png", output_image)

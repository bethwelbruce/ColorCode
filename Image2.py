import cv2
import numpy as np

# Read the input image
image = cv2.imread("input.jpg")

# Convert image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur the image slightly
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply median blur to further smooth the image
median_blurred_image = cv2.medianBlur(blurred_image, 5)

# Sharpen the image slightly
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(grayscale_image, -1, kernel)

# Save the output image
cv2.imwrite("output.png", sharpened_image)

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 07:14:39 2023

@author: Admin
"""


# ---------------- TASK 1 ----------------

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology, measure

# Load the image
image = io.imread('120-0002x2.png')

# Check the number of color channels
if image.shape[-1] == 4:
    # If the image has an alpha channel, remove it
    image = color.rgba2rgb(image)

# Convert the image to HSV
hsv_image = color.rgb2hsv(image)

# Display the original and HSV images side-by-side
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 8))
ax1.imshow(image)
ax1.set_title('Original Image')
ax2.imshow(hsv_image)
ax2.set_title('HSV Image')
plt.show()



# =============================================================================
# ---------------- TASK 2 ---------------- 
# =============================================================================

# Define the hue threshold range for red
hue_threshold = 0.02  # adjust as needed

# Define the saturation and value thresholds
saturation_threshold = 0.3  # adjust as needed
value_threshold = 0.2  # adjust as needed


# Create a mask for pixels that meet the threshold criteria
hue_mask = np.logical_or(hsv_image[:, :, 0] < hue_threshold, hsv_image[:, :, 0] > (1 - hue_threshold))
saturation_mask = hsv_image[:, :, 1] > saturation_threshold
value_mask = hsv_image[:, :, 2] > value_threshold
mask = np.logical_and.reduce((hue_mask, saturation_mask, value_mask))

# Plotting Mask
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(13, 13))
ax1.imshow(hue_mask)
ax1.set_title('Hue Mask')
ax2.imshow(saturation_mask)
ax2.set_title('Saturation Mask')
ax3.imshow(value_mask)
ax3.set_title('Value Mask')
ax4.imshow(mask)
ax4.set_title('Mask')
io.show()


# =============================================================================
# ---------------- TASK 3 ---------------- 
# =============================================================================


# Apply morphological operations to refine the mask
mask = morphology.binary_erosion(mask, morphology.disk(2))
mask = morphology.binary_dilation(mask, morphology.disk(2))

# Find the contours of the mask
contours = measure.find_contours(mask, 0.5)
#
# Label the connected components in the mask
labelled_mask = measure.label(mask)
#
# Display the original image, the mask, and the masked refined image
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 8))
ax1.imshow(image)
ax1.set_title('Original Image')
ax2.imshow(mask, cmap='gray')
ax2.set_title('Mask')


for contour in contours:
    ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
plt.show()


# =============================================================================
# ---------------- TASK 4 ---------------- 
# =============================================================================



# Find connected components in the mask
labels = measure.label(mask, connectivity=2)

# Create a figure with two subplots, one for the original image and one for the masked image with boxes around regions
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

# Show the masked image with boxes around regions in the second subplot
ax1.imshow(mask)
ax1.set_title('Boxed Masked Image')
ax2.imshow(image)
ax2.set_title('Boxed Image')


# Loop over each region and draw a box around it on the masked image
for region in measure.regionprops(labels):
    
    # Get the coordinates of the region's bounding box
    min_row, min_col, max_row, max_col = region.bbox
    # Draw the bounding box on the masked image
    rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, fill=False, edgecolor='red', linewidth=1)
    ax1.add_patch(rect)

# Loop over each region and draw a box around it on the Image
for region in measure.regionprops(labels):
    
    # Get the coordinates of the region's bounding box
    min_row, min_col, max_row, max_col = region.bbox
    # Draw the bounding box on the masked image
    rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, fill=False, edgecolor='red', linewidth=1)
    ax2.add_patch(rect)

# Show the figure
plt.show()


# =============================================================================
# ---------------- TASK 5 ---------------- 
# =============================================================================

# Get the properties of each connected component in the mask
props = measure.regionprops(labels)

# Initialize lists to store the circle components and their coordinates
circle_components = []
circle_coords = []

# Loop over each component and filter by its properties
for prop in props:
    # Check if the component is roughly circular
    if prop.area > 100 and prop.perimeter > 50 and prop.major_axis_length / prop.minor_axis_length > 0.8:
        # Add the component to the list of circle components
        circle_components.append(prop)
        # Get the coordinates of the component's bounding box
        min_row, min_col, max_row, max_col = prop.bbox
        # Add the coordinates to the list of circle coordinates
        circle_coords.append((min_row, min_col, max_row, max_col))

# Create new images with only the circle components
circle_images = []
for coords in circle_coords:
    min_row, min_col, max_row, max_col = coords
    circle_images.append(image[min_row:max_row, min_col:max_col])
    
    

# Display and save the circle images
for i, circle_image in enumerate(circle_images):
    plt.imshow(circle_image)    
    io.imsave(f"RoI_{i}.png", circle_image)
    plt.title(f"RoI {i+1}")
    plt.show()
    
    
    
# =============================================================================
# ---------------- TASK 6 ---------------- 
#   Converting a RoI to a vector
# =============================================================================

from PIL import Image, ImageEnhance

# Open the image file
image = Image.open("RoI_2.png")

# Convert the image to grayscale
grayscale_image = image.convert("L")

# Resize the image to 64x64 pixels
resized_image = grayscale_image.resize((64, 64))

# Save the resulting image
resized_image.save("64x64.jpg")






# ------------ contrast enhance the resized image to ensure it uses the full intensity range


# Enhance the contrast of the image
contrast_enhancer = ImageEnhance.Contrast(resized_image)
enhanced_image = contrast_enhancer.enhance(1.5)

# Save the resulting image
enhanced_image.save("64x64_contrast.jpg")





# ------------ subtract off the mean pixel value

# Load the contrast-enhanced grayscale image as a NumPy array

# Calculate the mean pixel value of the image
mean_pixel_value = np.mean(enhanced_image)

# Subtract off the mean pixel value from the image
image_normalized = image - mean_pixel_value

# Save the resulting image
Image.fromarray(image_normalized.astype(np.uint8)).save("64x64_normalized.jpg")




# -----------  flatten to a 4096-element vector (by concatenating image rows, use the flatten method from Numpy). (N.B. 4096 = 64x64)

# Load the normalized grayscale image as a NumPy array
image = np.array(Image.open("64x64_normalized.jpg"))

# Flatten the image into a 4096-element vector
image_vector = image.flatten()

# Save the resulting image vector as a NumPy array
np.save("RoI_64x64_vector.npy", image_vector)



# ------- now normalize the resulting vector to generate a unit vector in 4096 dimensions


# Load the flattened image vector from the NumPy file
image_vector = np.load("RoI_64x64_vector.npy")

# Normalize the image vector to generate a unit vector in 4096 dimensions
image_unit_vector = image_vector / np.linalg.norm(image_vector)

# Save the resulting unit vector as a NumPy array
np.save("RoI_64x64_unit_vector.npy", image_unit_vector)




# --------------- measure the distance of the result from all the exemplar vectors. Return the closest


# Load the unit vectors of the exemplars from the NumPy file
exemplars = np.load("1-NN-descriptor-vects.npy")

# Remove the extra column from the exemplars array
exemplars = exemplars[:, :4096]

# Load the unit vector of the image from the NumPy file
image_unit_vector = np.load("RoI_64x64_unit_vector.npy")

# Reshape the image_unit_vector to match the shape of exemplars
image_unit_vector = np.reshape(image_unit_vector, (1, -1))

# Remove the extra dimension from image_unit_vector
image_unit_vector = image_unit_vector[:, :4096]

# Repeat the image unit vector 62 times to match the number of exemplars
image_unit_vector = np.tile(image_unit_vector, (exemplars.shape[0], 1))

# Calculate the Euclidean distance between the image vector and each exemplar vector
distances = np.linalg.norm(exemplars - image_unit_vector, axis=1)

# Find the index of the closest exemplar
closest_index = np.argmin(distances)

# Retrieve the closest exemplar vector
closest_exemplar = exemplars[closest_index]

# Print the index of the closest exemplar and its distance from the image vector
print("Closest exemplar:", closest_index)
print("Distance:", distances[closest_index])

# You can now use the closest exemplar vector for further analysis or classification



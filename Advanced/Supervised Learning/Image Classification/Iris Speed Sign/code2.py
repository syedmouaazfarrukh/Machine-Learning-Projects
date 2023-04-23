# ---------------- TASK 1 ----------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology, measure

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

# Plotting Mask
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 8))
ax1.imshow(hue_mask)
ax1.set_title('Hue Mask')
ax2.imshow(saturation_mask)
ax2.set_title('Saturation Mask')
ax3.imshow(value_mask)
ax3.set_title('Value Mask')
io.show()


mask = np.logical_and.reduce((hue_mask, saturation_mask, value_mask))
fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 8))
ax1.imshow(mask)
ax1.set_title('Mask')


## Apply the mask to each color channel separately
#filtered_channels = []
#for i in range(3):
#    channel = image[:, :, i]
#    filtered_channel = channel.copy()
#    #filtered_channel[mask] = filters.median(channel[mask], selem=np.ones((3, 3)))
#    filtered_channels.append(filtered_channel)
##
### Combine the filtered channels into a single image
#filtered_image = np.stack(filtered_channels, axis=2)
##
### Display the filtered image
#io.imshow(filtered_image)
#io.show()


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


# Find connected components in the mask
labels = measure.label(mask, connectivity=2)

# Create a blank image to draw the rectangles on
box_image = np.zeros_like(mask)

# Loop over each region and draw a rectangle around it on the box image
for region in measure.regionprops(labels):
    
    # Get the coordinates of the region's bounding box
    min_row, min_col, max_row, max_col = region.bbox
    
    # Draw the rectangle on the box image
    cv2.rectangle(box_image, (min_col, min_row), (max_col, max_row), (255, 255, 255), 2)
    
# Save the box image
cv2.imwrite('box_image.png', box_image)
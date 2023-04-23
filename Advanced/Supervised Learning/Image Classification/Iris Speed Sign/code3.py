# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:36:31 2023

@author: Admin
"""

from skimage import measure

# Label each connected component in the mask
labels = measure.label(mask)

# Get the bounding boxes for each region
regions = []
for label in range(1, labels.max() + 1):
    # Get the indices of the non-zero pixels for this label
    indices = np.nonzero(labels == label)
    
    # Find the minimum and maximum indices along each axis
    bbox = (np.min(indices[1]), np.min(indices[0]), np.max(indices[1]), np.max(indices[0]))
    
    # Add the bounding box to the list of regions
    regions.append(bbox)
    
# Print the bounding boxes for each region
for i, bbox in enumerate(regions):
    print(f"Region {i+1}: {bbox}")
    
    
    
    
    
    
    
    
    
    
    
    
    
# Find connected components in the mask
labels = measure.label(mask, connectivity=1)

# Create a figure with two subplots, one for the original image and one for the masked image with boxes around regions
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

# Show the original image in the first subplot
ax1.imshow(image)
ax1.set_title('Original Image')

# Show the masked image with boxes around regions in the second subplot
ax2.imshow(image)
ax2.set_title('Masked Image')

# Loop over each region and draw a box around it on the masked image
for region in measure.regionprops(labels):
    # Get the coordinates of the region's bounding box
    min_row, min_col, max_row, max_col = region.bbox
    
    # Draw the bounding box on the masked image
    rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, fill=False, edgecolor='red', linewidth=2)
    ax2.add_patch(rect)

# Show the figure
plt.show()
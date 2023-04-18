# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 04:23:03 2023

@author: Admin
"""


"""
Task - 1
To find Harris interest points by thresholding the Harris response images for two images in Python, you can follow these steps:
- Load the images and convert them to grayscale.
- Define parameters for Harris corner detection, such as block size, aperture size, and k value.
- Compute the Harris response image for each input image using the defined parameters.
- Threshold the Harris response images by comparing each pixel value to a threshold value that is a fraction of the maximum pixel value.
- Locate the Harris interest points by finding the coordinates of the pixels that are above the threshold in the thresholded Harris response images.

"""


import numpy as np
from scipy import signal
from PIL import Image
import random

# Load images and convert them to grayscale
img1 = np.array(Image.open('arch1.png').convert('L'))
img2 = np.array(Image.open('arch2.png').convert('L'))

# Define parameters for Harris corner detection
window_size = 3
k = 0.04
threshold = 0.01

# Define the Sobel operators for x and y derivatives
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Compute x and y derivatives of input images
Ix1 = signal.convolve2d(img1, sobel_x, mode='same')
Iy1 = signal.convolve2d(img1, sobel_y, mode='same')
Ix2 = signal.convolve2d(img2, sobel_x, mode='same')
Iy2 = signal.convolve2d(img2, sobel_y, mode='same')

# Compute elements of the Harris matrix at each pixel
Ixx1 = signal.convolve2d(Ix1 * Ix1, np.ones((window_size, window_size)), mode='same')
Iyy1 = signal.convolve2d(Iy1 * Iy1, np.ones((window_size, window_size)), mode='same')
Ixy1 = signal.convolve2d(Ix1 * Iy1, np.ones((window_size, window_size)), mode='same')
Ixx2 = signal.convolve2d(Ix2 * Ix2, np.ones((window_size, window_size)), mode='same')
Iyy2 = signal.convolve2d(Iy2 * Iy2, np.ones((window_size, window_size)), mode='same')
Ixy2 = signal.convolve2d(Ix2 * Iy2, np.ones((window_size, window_size)), mode='same')

# Compute the Harris response image for each input image
det1 = Ixx1 * Iyy1 - Ixy1 ** 2
trace1 = Ixx1 + Iyy1
harris1 = det1 - k * trace1 ** 2
det2 = Ixx2 * Iyy2 - Ixy2 ** 2
trace2 = Ixx2 + Iyy2
harris2 = det2 - k * trace2 ** 2

# Threshold the Harris response images
harris1_thresh = harris1 > threshold * np.max(harris1)
harris2_thresh = harris2 > threshold * np.max(harris2)

# Find Harris interest points in image 1 and image 2
harris1_points = np.argwhere(harris1_thresh)
harris2_points = np.argwhere(harris2_thresh)

print("Harris Points for first image is: ",harris1_points)
print("Harris Points for Second image is: ",harris2_points)

"""
Task - 2
form "normalised patch descriptor vector" for all the Hips in both the images
- To form normalized patch descriptor vectors for all the Hips in both images, we need to extract a small patch around each Hip and convert it into a descriptor vector. We can then normalize the descriptor vector by dividing it by its L2 norm.
- The resulting descriptors1 and descriptors2 lists will contain normalized descriptor vectors for all the Hips in each image. Each descriptor vector will be a 1D numpy array of size 64, corresponding to a flattened 16x16 patch around the Hip.
"""



patch_size = 16  # Size of patch around each Hip
descriptor_size = 64  # Size of descriptor vector for each patch



# Loop through Hips in image 1
descriptors1 = []
for point in harris1_points:
    x, y = point[1], point[0]  # Extract x and y coordinates from point
    patch = img1[y-patch_size//2:y+patch_size//2, x-patch_size//2:x+patch_size//2]
    descriptor = patch.flatten()  # Flatten patch into 1D array
    if descriptor.size == patch_size**2:  # Make sure patch was extracted correctly
        descriptor = descriptor.astype(np.float64)
        descriptor /= np.linalg.norm(descriptor, ord=2) # Normalize descriptor vector
        descriptors1.append(descriptor)

# Loop through Hips in image 2
descriptors2 = []
for point in harris2_points:
    x, y = point[1], point[0]  # Extract x and y coordinates from point
    patch = img2[y-patch_size//2:y+patch_size//2, x-patch_size//2:x+patch_size//2]
    descriptor = patch.flatten()  # Flatten patch into 1D array
    if descriptor.size == patch_size**2:  # Make sure patch was extracted correctly
        descriptor = descriptor.astype(np.float64)
        descriptor /= np.linalg.norm(descriptor, ord=2) # Normalize descriptor vector
        descriptors2.append(descriptor)

'''
print("HIPS in first image: ", descriptors1)
print("HIPS in Seconds image: ", descriptors2)
'''


"""
Task - 3
Now match these normalised patch descriptor vectors using inner product op and threshold for strong matches. sort by match strength(strongest first). Result is a list of point correspondences [(r1i,c1i) to (r2j,c2j)]
    - To match the normalized patch descriptor vectors using the inner product and threshold for strong matches, you can use the following code:
        -- Here, desc1 and desc2 are the lists of normalized patch descriptor vectors for image 1 and image 2, respectively. The threshold variable is a threshold value for strong matches, which you can adjust to get the desired number of matches.
        -- The matches list contains tuples of point correspondences in the form [(r1i,c1i) to (r2j,c2j)]. The first element of the tuple is the index of the point in image 1, and the second element is the index of the corresponding point in image 2. The matches are sorted by match strength in descending order, so the first match in the list is the strongest.
"""

matches = []  # List of point correspondences

# Calculate inner product of normalized patch descriptor vectors
for i, d1 in enumerate(descriptors1):
    best_match = (-1, -1, float('-inf'))  # (index in desc2, distance) for best match
    for j, d2 in enumerate(descriptors2):
        distance = np.dot(d1, d2)
        if len(best_match) >= 3 and distance > best_match[2]:
            best_match = (j, distance)
    if best_match[1] > threshold:
        matches.append((i, best_match[0]))

# Sort matches by match strength (strongest first)
matches.sort(key=lambda x: x[1], reverse=True)


print("List of Point Correspondences: ",matches)

"""
Task - 4
To apply exhaustive RANSAC to filter outliers from the list of point correspondences and return the best translation between the images, you can follow these steps:
    - Define a function to calculate the translation given a set of point correspondences. This function should take two arguments - a list of point correspondences and a threshold value.
    - Use the function to calculate the translation for all possible combinations of point correspondences.
    - Calculate the number of inliers for each translation and choose the translation with the largest number of inliers.
    - Return the translation with the largest number of inliers.
    
"""



def calculate_translation(matches, threshold):
    # Create arrays of points
    points1 = np.array([harris1_points[i] for i, _ in matches])
    points2 = np.array([harris2_points[j] for _, j in matches])

    # Calculate translation
    translation = np.mean(points2 - points1, axis=0)

    # Calculate number of inliers
    num_inliers = sum(np.linalg.norm(points2 - (points1 + translation), axis=1) < threshold)

    return translation, num_inliers


def exhaustive_ransac(matches, threshold, iterations):
    best_translation = None
    best_num_inliers = 0

    for i in range(iterations):
        # Choose a random subset of matches
        subset = random.sample(matches, 3)

        # Calculate translation for subset
        translation, num_inliers = calculate_translation(subset, threshold)

        # Update best translation
        if num_inliers > best_num_inliers:
            best_translation = translation
            best_num_inliers = num_inliers

    return best_translation


# Set threshold and number of RANSAC iterations
threshold = 5
iterations = 1000

# Apply RANSAC to matches
best_translation = exhaustive_ransac(matches, threshold, iterations)

# Print best translation
print("Best translation:", best_translation)


"""
Task - 5
Use the above best translation to make a composite image and return this
    
"""


from PIL import Image
import numpy as np

# Load images and convert them to grayscale
img1 = np.array(Image.open('arch1.png').convert('RGBA'))
img2 = np.array(Image.open('arch2.png').convert('RGBA'))

# Convert RGBA images to RGB images
img1 = img1[:, :, :3]
img2 = img2[:, :, :3]

def create_composite_image(img1, img2, dr, dc):
    # Determine size of composite image
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape
    height = max(height1, height2)
    width = width1 + width2

    # Create empty composite image
    composite = np.zeros((height, width, 3), dtype=np.uint8)

    # Copy first image to left side of composite
    composite[0:height1, 0:width1] = img1

    # Copy second image to right side of composite, translated by (dr, dc)
    x_min = max(0, dc)
    x_max = min(width2, width2 + dc)
    y_min = max(0, dr)
    y_max = min(height2, height2 + dr)

    # Crop second image to overlapping region
    img2_cropped = img2[y_min - dr:y_max - dr, x_min - dc:x_max - dc]

    # Compute new row and column indices based on translation
    new_row = max(0, dr)
    new_col = max(0, dc)

    # Add second image to composite
    composite[new_row:new_row+height2-abs(int(dr)), new_col:new_col+width2-abs(int(dc))] += img2_cropped

    # Add overlapping region to composite
    composite[y_min:y_max, width1 + x_min:width1 + x_max] = img2[y_min - dr:y_max - dr, x_min - dc:x_max - dc]

    return composite



dr = abs(int(best_translation[0]))
dc = abs(int(best_translation[1]))
print(dr,dc)
# Generate composite image with translation offset (10, 20)
composite = create_composite_image(img1, img2, -30, 40)

# Save composite image to file
Image.fromarray(composite).save('composite-arch.png')











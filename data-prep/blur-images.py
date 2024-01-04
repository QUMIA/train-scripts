#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Read the data
data_dir = '/projects/0/einf6214/data'
df = pd.read_csv(os.path.join(data_dir, 'merged.csv'))
print(df.shape)


def blur_area_with_soft_edges(image, bbox, blur_intensity=15, feather_amount=15):
    # Ensure kernel sizes are odd
    blur_intensity = blur_intensity + 1 if blur_intensity % 2 == 0 else blur_intensity
    feather_amount = feather_amount + 1 if feather_amount % 2 == 0 else feather_amount

    # Extract coordinates from bbox
    x, y, w, h = bbox

    # Blur the specified area
    roi = image[y:y+h, x:x+w]
    blurred_roi = cv2.GaussianBlur(roi, (blur_intensity, blur_intensity), 0)

    # Create a mask for feathering
    mask = np.zeros_like(roi)
    cv2.rectangle(mask, (0, 0), (w, h), (255, 255, 255), thickness=cv2.FILLED)
    feathered_mask = cv2.GaussianBlur(mask, (feather_amount, feather_amount), 0) / 255

    # Blend the original and blurred areas
    feathered_roi = (roi * (1 - feathered_mask)) + (blurred_roi * feathered_mask)

    # Place the blended area back into the image
    image[y:y+h, x:x+w] = feathered_roi.astype(np.uint8)

    return image


def process_row(row):
    image_path = os.path.join(data_dir, 'merged', row['exam_id'], row['image_file'])

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the bounding box
    x = max(row['li_x'], row['re_x'], row['id_x'])
    y = max(row['li_y'], row['re_y'], row['id_y'])
    bbox = (0, y - 5, x + 32, 23) if x > -1 and y > -1 else None

    # Blur the image
    if bbox is not None:
        blurred_image = blur_area_with_soft_edges(image, bbox, 21, 15)
    else:
        blurred_image = image

    # Save the image
    exam_dir = os.path.join(data_dir, 'blurred', row['exam_id'])
    if not os.path.exists(exam_dir):
        os.makedirs(exam_dir)
    cv2.imwrite(os.path.join(exam_dir, row['image_file']), blurred_image)

    return blurred_image


# Debugging only
# cv2.rectangle(image, bbox[:2], (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
# plt.imshow(image)
# plt.show()


# Debugging
blurred_image = process_row(df.iloc[0])
plt.imshow(blurred_image, cmap='gray')
plt.show()


# Iterate through the rows and show progress
for i in tqdm(range(df.shape[0])):
    process_row(df.iloc[i])








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


def mask_region(image, bbox):
    # Extract coordinates from bbox
    x, y, w, h = bbox

    # Mask the specified area
    image[y:y+h, x:x+w] = 0

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

    # Debugging only
    # cv2.rectangle(image, bbox[:2], (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
    # plt.imshow(image)
    # plt.show()

    # Blur the image
    if bbox is not None:
        output_imate = mask_region(image, bbox)
    else:
        output_imate = image

    # Save the image
    exam_dir = os.path.join(data_dir, 'masked', row['exam_id'])
    print(exam_dir)
    if not os.path.exists(exam_dir):
        os.makedirs(exam_dir)
    cv2.imwrite(os.path.join(exam_dir, row['image_file']), output_imate)

    return output_imate


# # Debugging
# blurred_image = process_row(df.iloc[0])
# plt.imshow(blurred_image, cmap='gray')
# plt.show()


# Iterate through the rows and show progress
for i in tqdm(range(df.shape[0])):
    process_row(df.iloc[i])


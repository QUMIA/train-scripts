#!/usr/bin/env python
# coding: utf-8

# The data is provided as three csv files, corresponding to the three original SPSS files. 
# Here, the data is merged, duplicates are removed and some filtering (burnt-in print, h-scores) is performed.
# Data is written to merged.csv


import os
import pandas as pd
import hashlib


data_dir = '/projects/0/einf6214/data'


# Load the csv files for the three datasets
df1 = pd.read_csv(os.path.join(data_dir, 'data_1', 'output.csv'))
df2 = pd.read_csv(os.path.join(data_dir, 'data_2', 'output.csv'))
df3 = pd.read_csv(os.path.join(data_dir, 'data_3', 'output.csv'))
print (df1.shape, df2.shape, df3.shape)
# first extraction: (13075, 13) (39933, 13) (37345, 13)


df1.head()


# Test if there are any exam_ids that are in more than one dataset
print (set(df1['exam_id']).intersection(set(df2['exam_id'])))
print (set(df1['exam_id']).intersection(set(df3['exam_id'])))
print (set(df2['exam_id']).intersection(set(df3['exam_id'])))

# Only duplicates are between sets 2 and 3
duplicates = set(df2['exam_id']).intersection(set(df3['exam_id']))
print(len(duplicates))


# Check if the images are the same
for sample_dupe in duplicates:
  # calculate the checksum of the image file
  img_path_1 = os.path.join(data_dir, 'data_2', sample_dupe, '05.png')
  img_path_2 = os.path.join(data_dir, 'data_3', sample_dupe, '05.png')
  checksum_1 = hashlib.md5(open(img_path_1, 'rb').read()).hexdigest()
  checksum_2 = hashlib.md5(open(img_path_2, 'rb').read()).hexdigest()
  print (checksum_1, checksum_2)


# Test if there are any exam_ids that have the same value as the anon_ids

# for the first set, all exam_ids are equal to the anon_ids, because they lack an exam date
#print (set(df1['exam_id']).intersection(set(df1['anon_id'])))

print (set(df2['exam_id']).intersection(set(df2['anon_id'])))
print (set(df3['exam_id']).intersection(set(df3['anon_id'])))


# merge the three datasets
df = pd.concat([df1, df2, df3])
print(df.shape)
# first extraction: (90353, 13)


# remove the duplicate rows
df = df.drop_duplicates()
print(df.shape)
# first extraction: (89866, 13)


# count the number of unique exam_ids
print(len(set(df['exam_id']))) # 2188


# count the number of unique patients
print(len(set(df['anon_id']))) # 1845


# Show the distribution of the h_score
print(df['h_score'].value_counts())

# Filter out entries with wrong h_scores
valid_h_scores = [1.0, 2.0, 3.0, 4.0]
df = df[df['h_score'].isin(valid_h_scores)]
print(df.shape)

# h_score
# 1.0     61489
# 2.0     20090
# 3.0      7548
# 4.0       711
# 0.0         9
# 2.1         4
# 0.5         3
# 1.4         3
# 22.0        3
# 2.6         3
# 0.9         3
# Name: count, dtype: int64
# (89838, 13)


# filter out the rows that have markers
print(str(len(df[df['has_markers'] == True])) + " images have markers")
df = df[df['has_markers'] == False]
print(df.shape)

# Count entries for each h_score
print(df['h_score'].value_counts())

# First extraction:
# (82060, 13)
# h_score
# 1.0    57064
# 2.0    17973
# 3.0     6445
# 4.0      578
# Name: count, dtype: int64


# write the merged dataset to a csv file
df.to_csv(os.path.join(data_dir, 'merged.csv'), index=False)





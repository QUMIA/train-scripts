#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The data is provided as three csv files, corresponding to the three original SPSS files. 
# Here, the data is merged, duplicates are removed and some filtering (burnt-in print, h-scores) is performed.
# Data is written to merged.csv


# In[2]:


import os
import pandas as pd
import hashlib


# In[3]:


data_dir = '/projects/0/einf6214/data'


# In[4]:


# Load the csv files for the three datasets
df1 = pd.read_csv(os.path.join(data_dir, 'data_1', 'output.csv'))
df2 = pd.read_csv(os.path.join(data_dir, 'data_2', 'output.csv'))
df3 = pd.read_csv(os.path.join(data_dir, 'data_3', 'output.csv'))
print (df1.shape, df2.shape, df3.shape)


# In[5]:


df1.head()


# In[6]:


# Test if there are any exam_ids that are in more than one dataset
print (set(df1['exam_id']).intersection(set(df2['exam_id'])))
print (set(df1['exam_id']).intersection(set(df3['exam_id'])))
print (set(df2['exam_id']).intersection(set(df3['exam_id'])))
duplicates = set(df2['exam_id']).intersection(set(df3['exam_id']))


# In[7]:


# Check if the images are the same
for sample_dupe in duplicates:
  # calculate the checksum of the image file
  img_path_1 = os.path.join(data_dir, 'data_2', sample_dupe, '05.png')
  img_path_2 = os.path.join(data_dir, 'data_3', sample_dupe, '05.png')
  checksum_1 = hashlib.md5(open(img_path_1, 'rb').read()).hexdigest()
  checksum_2 = hashlib.md5(open(img_path_2, 'rb').read()).hexdigest()
  print (checksum_1, checksum_2)


# In[8]:


# Test if there are any exam_ids that have the same value as the anon_ids

# for the first set, all exam_ids are equal to the anon_ids, because they lack an exam date
#print (set(df1['exam_id']).intersection(set(df1['anon_id'])))

print (set(df2['exam_id']).intersection(set(df2['anon_id'])))
print (set(df3['exam_id']).intersection(set(df3['anon_id'])))


# In[9]:


# merge the three datasets
df = pd.concat([df1, df2, df3])
print(df.shape)


# In[10]:


# remove the duplicate rows
df = df.drop_duplicates()
print(df.shape)


# In[11]:


# count the number of unique exam_ids
print(len(set(df['exam_id'])))


# In[12]:


# count the number of unique patients
print(len(set(df['anon_id'])))


# In[13]:


# Show the distribution of the h_score
print(df['h_score'].value_counts())

# Filter out entries with wrong h_scores
valid_h_scores = [1.0, 2.0, 3.0, 4.0]
df = df[df['h_score'].isin(valid_h_scores)]
print(df.shape)


# In[14]:


# filter out the rows that have print
df = df[df['has_print'] == False]
print(df.shape)

# Count entries for each h_score
print(df['h_score'].value_counts())


# In[15]:


# write the merged dataset to a csv file
df.to_csv(os.path.join(data_dir, 'merged.csv'), index=False)


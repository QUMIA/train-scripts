#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[2]:


# Read the data
data_dir = '/projects/0/einf6214/data'
df = pd.read_csv(os.path.join(data_dir, 'merged.csv'))
print(df.shape)


# In[3]:


from sklearn.model_selection import GroupShuffleSplit 

def get_split(df, groups, test_size, random_state=42):
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    split = splitter.split(df, groups=groups)
    inds1, inds2 = next(split)
    df1 = df.iloc[inds1]
    df1.reset_index(drop=True, inplace=True)
    df2 = df.iloc[inds2]
    df2.reset_index(drop=True, inplace=True)
    return df1, df2

def split_n_way(df, group_column, sizes, random_state=42):
    assert abs(sum(sizes) - 1.0) < 1e-6
    df2 = df.copy()
    results = []
    for i in range(len(sizes)-1):
        next_fraction = sum(sizes[i+1:]) / sum(sizes[i:])
        df1, df2 = get_split(df2, df2[group_column], next_fraction, random_state=random_state)
        results.append(df1)
    results.append(df2)
    return results

df_train, df_val, df_test = split_n_way(df, group_column='anon_id', sizes=[0.6, 0.2, 0.2])
print(df_train.shape, df_val.shape, df_test.shape)


# In[4]:


# Check if there are any anon_ids that are in more than one dataset
print (set(df_train['anon_id']).intersection(set(df_val['anon_id'])))
print (set(df_train['anon_id']).intersection(set(df_test['anon_id'])))
print (set(df_val['anon_id']).intersection(set(df_test['anon_id'])))


# In[5]:


# For each set, print the distribution of h_scores
print(df_train['h_score'].value_counts())
print(df_val['h_score'].value_counts())
print(df_test['h_score'].value_counts())


# In[6]:


# write the train, val, and test datasets to csv files
df_train.to_csv(os.path.join(data_dir, 'split_train.csv'), index=False)
df_val.to_csv(os.path.join(data_dir, 'split_val.csv'), index=False)
df_test.to_csv(os.path.join(data_dir, 'split_test.csv'), index=False)


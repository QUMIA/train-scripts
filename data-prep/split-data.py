#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import hashlib


# In[2]:


data_dir = '/projects/0/einf6214/data'


# In[3]:


# Load the csv files for the three datasets
df1 = pd.read_csv(os.path.join(data_dir, 'data_1', 'output.csv'))
df2 = pd.read_csv(os.path.join(data_dir, 'data_2', 'output.csv'))
df3 = pd.read_csv(os.path.join(data_dir, 'data_3', 'output.csv'))
print (df1.shape, df2.shape, df3.shape)


# In[4]:


df1.head()


# In[5]:


# Test if there are any exam_ids that are in more than one dataset
print (set(df1['exam_id']).intersection(set(df2['exam_id'])))
print (set(df1['exam_id']).intersection(set(df3['exam_id'])))
print (set(df2['exam_id']).intersection(set(df3['exam_id'])))
duplicates = set(df2['exam_id']).intersection(set(df3['exam_id']))


# In[6]:


for sample_dupe in duplicates:
  # calculate the checksum of the image file
  img_path_1 = os.path.join(data_dir, 'data_2', sample_dupe, '05.png')
  img_path_2 = os.path.join(data_dir, 'data_3', sample_dupe, '05.png')
  checksum_1 = hashlib.md5(open(img_path_1, 'rb').read()).hexdigest()
  checksum_2 = hashlib.md5(open(img_path_2, 'rb').read()).hexdigest()
  print (checksum_1, checksum_2)


# In[7]:


# Test if there are any exam_ids that have the same value as the anon_ids
#print (set(df1['exam_id']).intersection(set(df1['anon_id'])))
print (set(df2['exam_id']).intersection(set(df2['anon_id'])))
print (set(df3['exam_id']).intersection(set(df3['anon_id'])))


# In[8]:


# merge the three datasets
df = pd.concat([df1, df2, df3])
print(df.shape)


# In[9]:


# remove the duplicate rows
df = df.drop_duplicates()
print(df.shape)


# In[10]:


# count the number of unique exam_ids
print(len(set(df['exam_id'])))


# In[11]:


# count the number of unique patients
print(len(set(df['anon_id'])))


# In[12]:


# write the merged dataset to a csv file
df.to_csv(os.path.join(data_dir, 'merged.csv'), index=False)


# In[19]:


# Filter out entries with wrong h_scores
valid_h_scores = [1.0, 2.0, 3.0, 4.0]
df = df[df['h_score'].isin(valid_h_scores)]
print(df.shape)


# In[23]:


# filter out the rows that have print
df_no_print = df[df['has_print'] == False]
print(df_no_print.shape)

# Count entries for each h_score
print(df_no_print['h_score'].value_counts())


# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split

def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if 'columns' in df_input and stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


# In[22]:


df_train, df_val, df_test = \
    split_stratified_into_train_val_test(df_no_print, stratify_colname='h_score', frac_train=0.60, frac_val=0.20, frac_test=0.20)

print(df_train.shape, df_val.shape, df_test.shape)


# In[ ]:


# No this doesn't work
# ids = df_no_print['anon_id'].unique()
# ids_train, ids_val, ids_test = \
#     split_stratified_into_train_val_test(ids, stratify_colname='anon_id', frac_train=0.60, frac_val=0.20, frac_test=0.20)
# print(len(ids_train), len(ids_val), len(ids_test))


# In[36]:


from sklearn.model_selection import GroupShuffleSplit 

def get_split(df):
    splitter = GroupShuffleSplit(test_size=.40, n_splits=1, random_state = 7)
    split = splitter.split(df_no_print, groups=df['anon_id'])
    train_inds, temp_indices = next(split)
    train_df = df.iloc[train_inds]
    train_df.reset_index(drop=True)
    temp_df = df.iloc[temp_indices]
    temp_df.reset_index(drop=True)

    splitter2 = GroupShuffleSplit(test_size=.50, n_splits=1, random_state = 7)
    split2 = splitter2.split(temp_df, groups=temp_df['anon_id'])
    test_inds, val_inds = next(split2)
    test_df = df.iloc[test_inds]
    test_df.reset_index(drop=True)
    val_df = df.iloc[val_inds]
    val_df.reset_index(drop=True)

    return train_df, val_df, test_df

df_train, df_val, df_test = get_split(df_no_print)
print(df_train.shape, df_val.shape, df_test.shape)


# In[38]:


# For each set, print the distribution of h_scores
print(df_train['h_score'].value_counts())
print(df_val['h_score'].value_counts())
print(df_test['h_score'].value_counts())


# In[37]:


# write the train, val, and test datasets to csv files
df_train.to_csv(os.path.join(data_dir, 'split_train.csv'), index=False)
df_val.to_csv(os.path.join(data_dir, 'split_val.csv'), index=False)
df_test.to_csv(os.path.join(data_dir, 'split_test.csv'), index=False)


# In[ ]:





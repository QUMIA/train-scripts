#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import torch


# Automatically reload modules
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# Read and concatenate the data sets (just use all of them for now)
df1 = pd.read_csv("df_validation_predictions.csv")
df2 = pd.read_csv("df_train_predictions.csv")
#df3 = pd.read_csv("/projects/0/einf6214/output/2024-03-06_18-23-27_test-test-set/output/df_test_predictions.csv")
df = pd.concat([df1, df2])
print(df.shape)
print(df.columns.values)


# Input
input_column = 'muscle'
input_value_column = 'h_score'
print(df[input_column].value_counts())
input_values = df[input_column].unique() # all muscles
#input_values = input_values[input_values != 'Extensors'] # remove Extensors
#input_values = ['Biceps','RF','TA','GM','VL','Deltoid'] # six most common muscles
value_to_index = {value: idx for idx, value in enumerate(input_values)}
print(value_to_index)


# Output
use_diagnosis = True

# Remove diagnosis 99
df = df[df['diagnosis'] != 99]
print(df['diagnosis'].value_counts())


# Aggregate the h_scores for each muscle
df_agg = df.groupby(['exam_id', input_column]).agg({'prediction': 'max', 'h_score': 'max', 'diagnosis': 'first'}).reset_index()
# df_agg['h_score'] = np.random.normal(2, 1.0, df_agg.shape[0])
# df_agg['h_score'] = 1.0
df_agg.head(25)


def create_vectors(df):
    input_vectors = {id: np.full(len(input_values), np.nan) for id in df['exam_id'].unique()}
    label_vectors = {id: False for id in df['exam_id'].unique()}

    # Populate the vectors
    for _, row in df.iterrows():
        exam_id = row['exam_id']
        muscle = row[input_column]
        if muscle in input_values:
            value_index = value_to_index[muscle]
            input_vectors[exam_id][value_index] = row[input_value_column]
            #diagnosis_index = diagnosis_to_index[row['diagnosis']]
            if use_diagnosis:
                label_vectors[exam_id] = row['diagnosis']
            else:
                label_vectors[exam_id] = row['diagnosis'] != 0

    # Convert the dictionary to a DataFrame for easy handling/viewing
    input_df = pd.DataFrame.from_dict(input_vectors, orient='index', columns=input_values)
    label_df = pd.DataFrame.from_dict(label_vectors, orient='index', columns=['diagnosis'])
    return (input_df, label_df)

def create_max_vectors(df):
    n = 4
    input_vectors = {id: np.full(n, np.nan) for id in df['exam_id'].unique()}
    label_vectors = {id: -1 for id in df['exam_id'].unique()}

    grouped = df.groupby('exam_id')

    # Iterate and perform operations
    for name, group in grouped:
        # print(f"Group: {name}, Value count: {group['prediction'].count()}")
        # print(group.nlargest(n, input_value_column)[input_value_column].values)
        if group.shape[0] >= n:
            largest_n = group.nlargest(n, input_value_column)[input_value_column].values
            for i in range(n):
                input_vectors[name][i] = largest_n[i]
            label_vectors[name] = group['diagnosis'].values[0]
        else:
            # remove from dictionary
            del input_vectors[name]
            del label_vectors[name]

    # Convert the dictionary to a DataFrame for easy handling/viewing
    input_df = pd.DataFrame.from_dict(input_vectors, orient='index', columns=[f'{input_value_column}_{i}' for i in range(n)])
    label_df = pd.DataFrame.from_dict(label_vectors, orient='index', columns=['diagnosis'])
    return (input_df, label_df)

input_df, label_df = create_vectors(df_agg)

# Handling missing values in 'input_df' (simple imputation with mean of each column)
# input_df = input_df.apply(lambda x: x.fillna(x.mean()), axis=1)
# input_df = input_df.fillna(0)
#input_df_filled = pd.DataFrame(0, index=np.arange(input_df.shape[0]), columns=input_df.columns)
#print(input_df_filled.head(10))

print(input_df.shape, label_df.shape)
display(input_df.head(20))
display(label_df.head(20))

# input_df, label_df = create_max_vectors(df_agg)
# display(input_df.head(10))
# display(label_df.head(10))


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_df, label_df, test_size=0.2, random_state=42)

# Initializing and training the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced')
clf.fit(X_train, y_train)

def print_accuracy(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Accuracy: {accuracy}")

print_accuracy(clf, X_train, y_train)
print_accuracy(clf, X_test, y_test)
print(f"Maximum depth of the tree: {clf.tree_.max_depth}")
print(f"Total number of nodes: {clf.tree_.node_count}")
print(f"Number of leaves: {clf.tree_.n_leaves}")


# Accuracy: 0.7378917378917379
# Accuracy: 0.4431818181818182
# Maximum depth of the tree: 10
# Total number of nodes: 143
# Number of leaves: 72

# Without diagnosis 99
# Accuracy: 0.7670807453416149
# Accuracy: 0.4691358024691358
# Maximum depth of the tree: 10
# Total number of nodes: 137
# Number of leaves: 69

# Without extensor muscles
# Accuracy: 0.7670807453416149
# Accuracy: 0.4567901234567901
# Maximum depth of the tree: 10
# Total number of nodes: 141
# Number of leaves: 71

# Before fillna(mean)
# Accuracy: 0.9004702194357367
# Accuracy: 0.70625
# Maximum depth of the tree: 10
# Total number of nodes: 273
# Number of leaves: 137

# After fillna(mean)
# Accuracy: 0.8612852664576802
# Accuracy: 0.7
# Maximum depth of the tree: 10
# Total number of nodes: 181
# Number of leaves: 91

# Accuracy: 0.8934169278996865
# Accuracy: 0.6875
# Maximum depth of the tree: 10
# Total number of nodes: 325
# Number of leaves: 163

# Accuracy: 0.8103448275862069
# Accuracy: 0.696875
# Maximum depth of the tree: 10
# Total number of nodes: 255
# Number of leaves: 128


label_values = label_df['diagnosis'].unique()
label_values = [str(x) for x in label_values]
print(input_values, label_values)


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(100,50))
plot_tree(clf, filled=True, rounded=True, class_names=label_values, feature_names=input_values)
plt.show()


print (X_train)
print (y_train)


from sklearn.metrics import f1_score

def print_f1_score(X_train, Y_train, X_test, y_test):
    y_train_pred = mlp_classifier.predict(X_train)
    y_test_pred = mlp_classifier.predict(X_test)
    f1_score_train = f1_score(y_train, y_train_pred, average='weighted')
    f1_score_test = f1_score(y_test, y_test_pred, average='weighted')
    print(f"F1 score (train): {f1_score_train:.2f}")
    print(f"F1 score (test): {f1_score_test:.2f}")


# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split


# # Initialize the MLPClassifier
# mlp_classifier = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=500)

# # Train the model
# mlp_classifier.fit(X_train, y_train)

# # Evaluate the model
# print_f1_score(X_train, y_train, X_test, y_test)


# import shap

# # explain all the predictions in the test set
# explainer = shap.KernelExplainer(sklearn.nn.predict_proba, X_train)
# shap_values = explainer.shap_values(X_test)
# shap.force_plot(explainer.expected_value[0], shap_values[..., 0], X_test)


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
print(type(X_train.values))


print(X_train.head())
print(y_train.head())


def df_to_masked_array(df):
    # Convert the dataframe to a numpy array
    array = df.to_numpy()

    # Create a mask array, 1.0 where the original array is not NaN, and 0.0 where it is NaN
    mask = np.where(np.isnan(array), 0.0, 1.0)

    # Stack the original array and the mask array to form a new array with 2 channels
    two_channel_array = np.stack((array, mask), axis=-1)

    # Replace NaN values in the original data with 0.0 (or another value of your choice)
    two_channel_array[np.isnan(two_channel_array[..., 0]), 0] = 0.0

    return two_channel_array

X_train_masked = df_to_masked_array(X_train)
print(X_train_masked.shape)
#print(X_train_masked[0:10])
X_test_masked = df_to_masked_array(X_test)
print(X_test_masked.shape)


def boolean_df_to_int_array(df):
    return df.astype(int).to_numpy()

if use_diagnosis:
    y_train_int = y_train.astype(int).to_numpy()
    print(y_train_int[0:10])
    y_test_int = y_test.astype(int).to_numpy()
    print(y_test_int[0:10])
else:
    y_train_int = boolean_df_to_int_array(y_train)
    print(y_train_int[0:10])
    y_test_int = boolean_df_to_int_array(y_test)
    print(y_test_int[0:10])


from torch.utils.data import Dataset, DataLoader

X_train_tensor = torch.tensor(X_train_masked, dtype=torch.float).permute(0, 2, 1)  # Swap the last two dimensions
y_train_tensor = torch.tensor(y_train_int, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_masked, dtype=torch.float).permute(0, 2, 1)  # Swap the last two dimensions
y_test_tensor = torch.tensor(y_test_int, dtype=torch.long)

# Create DataLoader for batching
train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


import torch
import torch.nn as nn
import torch.optim as optim
from qumia_model2 import MaskedClassifier
from torchsummary import summary

# Define the model, loss function, and optimizer
vector_length = len(input_values)
num_classes = len(label_values) if use_diagnosis else 1
model = MaskedClassifier(vector_length, num_classes)
criterion = nn.CrossEntropyLoss() if use_diagnosis else nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# print model summary
summary(model, (2, vector_length))

# Training the model
num_epochs = 20  # Set the number of epochs
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Forward pass
        outputs = model(inputs)
        target = labels.squeeze() if use_diagnosis else labels.float()
        loss = criterion(outputs, target)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



# Evaluate the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_preds = model(X_test_tensor)
    _, predicted_classes = torch.max(test_preds, 1)
    matches = (predicted_classes == y_test_tensor.squeeze())
    accuracy = matches.sum().float().item() / y_test_tensor.size(0)
    print(f'Accuracy: {accuracy:.4f}')


import torch
from sklearn.metrics import f1_score

# Set the model to evaluation mode
model.eval()

# Store all the predictions and true labels
all_predictions = []
all_labels = []
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)

        if use_diagnosis:
            # Get the predicted classes by finding the max logit value
            _, predicted_classes = torch.max(outputs, 1)
            
            # Collect the predictions and true labels
            all_predictions.extend(predicted_classes.tolist())
            all_labels.extend(labels.tolist())

            correct_predictions += (predicted_classes == labels.squeeze()).sum().item()
            total_predictions += labels.size(0)

        else:

            # Apply sigmoid to the outputs to get the probabilities
            probs = torch.sigmoid(outputs.squeeze())  # Squeeze to remove unnecessary dimensions
            # Convert probabilities to binary predictions using a threshold of 0.5
            predictions = (probs >= 0.5).long()  # Convert to long to match labels type
            
            # Collect the predictions and true labels (for f1 score)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

            # Compare predictions to true labels (for accuracy)
            correct_predictions += (predictions == labels.squeeze()).sum().item()
            total_predictions += labels.size(0)

if use_diagnosis:
    # Calculate the F1 score (choose 'macro', 'micro', or 'weighted' as per your requirement)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_micro = f1_score(all_labels, all_predictions, average='micro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')

    print(f'F1 Score (Macro): {f1_macro:.4f}')
    print(f'F1 Score (Micro): {f1_micro:.4f}')
    print(f'F1 Score (Weighted): {f1_weighted:.4f}')
    
    from sklearn.metrics import classification_report
    # all_classes = np.array(label_values)
    # unique_labels = np.unique(np.concatenate((all_classes, all_predictions)))
    # print(unique_labels)
    report = classification_report(all_labels, all_predictions)
    print(report)
    
else:
    # Calculate the F1 score
    f1 = f1_score(all_labels, all_predictions, average='binary')  # 'binary' for binary classification tasks
    print(f'F1 Score: {f1:.4f}')

print(f'Accuracy: {correct_predictions / total_predictions:.4f}')


label_values.astype(int)


import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")





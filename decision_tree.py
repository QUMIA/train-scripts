#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


df = pd.read_csv("/projects/0/einf6214/output/2023-12-13_21-51-57_baseline-4/output/df_val_predictions.csv")
df.head(25)


input_column = 'muscle'
input_value_column = 'prediction'
print(df[input_column].value_counts())
input_values = df[input_column].unique()
value_to_index = {value: idx for idx, value in enumerate(input_values)}
print(value_to_index)


# Remove diagnosis 99
df = df[df['diagnosis'] != 99]


print(df['diagnosis'].value_counts())


# Aggregate the h_scores for each muscle
df_agg = df.groupby(['exam_id', input_column]).agg({'prediction': 'mean', 'h_score': 'mean', 'diagnosis': 'first'}).reset_index()
df_agg.head(25)


# Remove extensor muscles
print(df_agg['muscle'].value_counts())
df_agg = df_agg[df_agg['muscle'] != 'Extensors']
print(df_agg['muscle'].value_counts())
input_values = input_values[input_values != 'Extensors']


input_values


def create_vectors(df):
    input_vectors = {id: np.full(len(input_values), np.nan) for id in df['exam_id'].unique()}
    label_vectors = {id: False for id in df['exam_id'].unique()}

    # Populate the vectors
    for _, row in df.iterrows():
        value_index = value_to_index[row[input_column]]
        input_vectors[row['exam_id']][value_index] = row[input_value_column]
        #diagnosis_index = diagnosis_to_index[row['diagnosis']]
        label_vectors[row['exam_id']] = row['diagnosis'] != 0

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
display(input_df.head(10))
display(label_df.head(10))

# input_df, label_df = create_max_vectors(df_agg)
# display(input_df.head(10))
# display(label_df.head(10))


# Handling missing values in 'input_df' (simple imputation with mean of each column)
input_df_filled = input_df.fillna(0)
#input_df_filled = pd.DataFrame(0, index=np.arange(input_df.shape[0]), columns=input_df.columns)
#print(input_df_filled.head(10))

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_df_filled, label_df, test_size=0.2, random_state=42)

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


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# Initialize the MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)

# Train the model
mlp_classifier.fit(X_train, y_train)

# Evaluate the model
print(f"Classification accuracy: {mlp_classifier.score(X_train, y_train):.2f}")
print(f"Classification accuracy: {mlp_classifier.score(X_test, y_test):.2f}")


from sklearn.metrics import f1_score

y_train_pred = mlp_classifier.predict(X_train)
y_test_pred = mlp_classifier.predict(X_test)
f1_score_train = f1_score(y_train, y_train_pred, average='weighted')
f1_score_test = f1_score(y_test, y_test_pred, average='weighted')
print(f"F1 score (train): {f1_score_train:.2f}")
print(f"F1 score (test): {f1_score_test:.2f}")





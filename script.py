import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn import tree
from sklearn.decomposition import PCA


labels = pd.read_csv("label.csv",header=None)

gene_expression = pd.read_csv("gene_expression.csv",delimiter=";",header=None)

# PART 1
# Replace commas with periods and convert strings to float values
gene_expression.replace(',', '.', regex=True, inplace=True)
gene_expression = gene_expression.astype(float)

X = gene_expression  # Features
y = labels # Target variable

# Flatten the target variable
y = np.ravel(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Select top 100 features based on mutual information score
#selector = SelectKBest(score_func=mutual_info_classif, k=100)
#X_train_selected = selector.fit_transform(X_train, y_train)
#X_test_selected = selector.transform(X_test)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier()

# Train the decision tree classifier on the training data
clf.fit(X_train, y_train)

# Calculate training accuracy
training_accuracy = clf.score(X_train, y_train)
print("Training Accuracy:", training_accuracy)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate real accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the decision tree
tree.plot_tree(clf, filled=True)
plt.show()

# PART 2
# Initialize PCA to optimize number of components
pca_val = PCA(n_components=None)

# Fit PCA to the training data
pca_val.fit(X_train)

# Calculate cumulative explained variance ratio
explained_variance_ratio = pca_val.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Find the number of components that capture 90% of the variance
n_components_90 = np.argmax(cumulative_variance_ratio >= 0.90) + 1 
n_components = n_components_90
print("N_components:",n_components)

# Initialize PCA with the desired number of components
pca = PCA(n_components=n_components)

# Fit PCA to the training data
pca.fit(X_train)

# Get the basis matrix V
basis_matrix_V = pca.components_

best_accuracy = 0
best_tree = None
best_k = None

for k in range(1, n_components+1):
    # Project the original data X to the top K components of V
    X_train_reduced = pca.transform(X_train)[:, :k]
    X_test_reduced = pca.transform(X_test)[:, :k]

    # Create a decision tree out of these reduced data
    clf2 = DecisionTreeClassifier()
    clf2.fit(X_train_reduced, y_train)

    # Make predictions on the test data
    y_pred = clf2.predict(X_test_reduced)
    # Calculate validation accuracy
    validation_accuracy = accuracy_score(y_test, y_pred)

    # Keep track of the best model
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_tree = clf2
        best_k = k

# Visualize the best decision tree
tree.plot_tree(best_tree, filled=True)
plt.title(f'Best decision tree with {best_k} components')
plt.show()

# Evaluate the best tree on the testing set
X_test_reduced = pca.transform(X_test)[:, :best_k]
y_pred_final = best_tree.predict(X_test_reduced)
testing_accuracy = accuracy_score(y_test,y_pred_final)
print(f"Testing Accuracy of the best decision tree with {best_k} components:", testing_accuracy)
# Heart Attack Prediction Project

This repository contains three Jupyter Notebooks that explore different approaches to predict heart attacks using machine learning techniques. The three notebooks focus on different methodologies: Artificial Neural Network (ANN), Decision Tree, and K-Means clustering.

## Dataset
The dataset used in this project is the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

## Notebooks Overview

### 1. Heart Attack Prediction using ANN
- The first notebook utilizes an Artificial Neural Network (ANN) to predict heart attacks.
- Data is loaded from a cleaned dataset.
- Data preparation involves separating features and labels, one-hot encoding categorical variables, and rescaling numeric features.
- An ANN model is built using the Keras library with multiple dense layers and dropout regularization.
- The model is trained and evaluated on the training and validation sets.
- Visualizations include training and validation loss/accuracy plots and a confusion matrix.

### 2. Heart Attack Risk Prediction with Decision Tree
- The second notebook employs a Decision Tree classifier for heart attack risk prediction.
- Initial exploratory data analysis is performed to visualize the distribution of key features.
- Data preparation involves one-hot encoding categorical variables and rescaling numeric features.
- The Decision Tree model is trained and evaluated on the training and validation sets.
- Fine-tuning is performed to address potential overfitting issues.
- Random Forest and XGBoost classifiers are also explored.

### 3. Heart Attack Risk Clustering with K-Means
- The third notebook focuses on clustering using the K-Means algorithm.
- Initially, the optimal number of clusters is determined using the elbow method.
- The K-Means algorithm is applied with two clusters due to ambiguity in the elbow method.
- Visualization includes a scatter plot of the clustered data points and centroids.
- The performance of the clustering is evaluated using adjusted Rand Index (ARI), normalized mutual information (NMI), and accuracy metrics.


## Instructions
1. Load and run each notebook in a Jupyter environment sequentially to execute the code and observe the output.
2. Explore and experiment with the provided code to gain insights into different machine learning approaches for heart attack prediction.

Feel free to reach out for any questions or improvements!

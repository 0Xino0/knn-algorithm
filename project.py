import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the data from a CSV file
data = pd.read_csv("Cancer_Data.csv")
# print(data.shape)  # Check the shape of the dataset
# data.head()  # View the first few rows of the dataset

# 2. Plot a histogram for the "diagnosis" column
data["diagnosis"].hist(bins=10)  # Divide the range of values into 10 intervals (bins)
plt.show()

# 3. Convert diagnosis values to numeric (Malignant -> 1, Benign -> 0)
data["diagnosis"].replace('M', 1, inplace=True)  # Replace 'M' with 1
data["diagnosis"].replace('B', 0, inplace=True)  # Replace 'B' with 0

# 4. Drop unnecessary columns
data.drop(columns=['Unnamed: 32', 'id'], inplace=True)  # Remove redundant columns

# 5. Extract features (x) and labels (y)
y = data.values[:, 0]  # Labels (cancer type)
x = data.values[:, 1:28]  # Features

# 6. Split data into training and testing sets
def train_test_split(x, y, test_size=0.3, random_state=1):
    np.random.seed(random_state)  # Set random seed for reproducibility
    total_samples = x.shape[0]  # Total number of samples in the dataset
    test_samples = int(total_samples * test_size)  # Number of samples for the test set
    indices = np.random.permutation(total_samples)  # Randomly shuffle the sample indices
    
    test_indices = indices[:test_samples]  # Select indices for the test set
    train_indices = indices[test_samples:]  # Select indices for the training set
    
    x_train, x_test = x[train_indices], x[test_indices]  # Split features
    y_train, y_test = y[train_indices], y[test_indices]  # Split labels
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(x, y)  # Perform the split

# 7. Define a function to compute the Euclidean distance between two points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))  # Compute the square root of the sum of squared differences

# 8. Define the K-Nearest Neighbors algorithm
def knn_predict(x_train, y_train, x_test, k=5):
    predictions = []  # List to store predictions for each test point
    for test_point in x_test:
        # Calculate the distance between the test point and all training points
        distances = [euclidean_distance(test_point, train_point) for train_point in x_train]
        
        # Sort distances and select indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_indices]  # Get the labels of the nearest neighbors
        
        # Predict the class based on majority voting
        prediction = np.bincount(k_nearest_labels.astype(int)).argmax()  # Convert to int and find the most common label
        predictions.append(prediction)  # Add the prediction to the list
    
    return np.array(predictions)  # Return predictions as a NumPy array

# 9. Make predictions for the test data
y_pred = knn_predict(x_train, y_train, x_test, k=5)

# 10. Define a function to calculate accuracy
def calAccuracy(y_test, y_pred):
    accuracy = np.sum(y_test == y_pred) / len(y_test) * 100  # Calculate percentage of correct predictions
    return accuracy

accuracy = calAccuracy(y_test, y_pred)  # Calculate accuracy of the model
print(f'Accuracy: {accuracy}%')  # Print the accuracy

# 11. Predict for a specific sample
sample = np.array([13.73, 22.61, 93.6, 578.3, 0.1131, 0.2293,
                   0.2128, 0.08025, 0.2069, 0.07682, 0.2121, 1.169, 2.061, 
                   19.21, 0.006429, 0.05936, 0.05501, 0.01628, 0.01961, 
                   0.008093, 15.03, 32.01, 108.8, 697.7, 0.1651, 0.7725, 0.6943])

# Get sample data from the user
# user_input = input("Enter the sample attributes (separate numbers with commas): ")

# Convert user input to a numeric array
# try:
#     sample = np.array([float(x) for x in user_input.split(',')])  # Convert input to an array of floats
#     if sample.size != 27:  # Ensure the sample has exactly 27 features
#         raise ValueError("Please enter exactly 27 values.")
# except ValueError as e:
#     print(f"Error in input: {e}")
#     exit()

# Predict for the given sample
y_pred_sample = knn_predict(x_train, y_train, sample.reshape(1, -1), k=5)
print(f'Prediction for sample: {y_pred_sample[0]}')  # 1 = Malignant, 0 = Benign

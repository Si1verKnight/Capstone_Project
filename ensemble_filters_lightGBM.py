import cv2
import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hinge_loss


# Load the image
data_path = "C:/Users/msnar/OneDrive/Desktop/DataSet2"


# Using os.listdir()
folders = [folder for folder in os.listdir(data_path)]

c_array = []
f_array = []

for image_folder in folders:
    # List all files in the image folder
    image_files = [file for file in os.listdir("C:/Users/msnar/OneDrive/Desktop/DataSet2/"+image_folder)]
   
    for image_file in image_files:

        image_path = "C:/Users/msnar/OneDrive/Desktop/DataSet2/"+image_folder+'/'+image_file

        image = cv2.imread(image_path)

        # Convert to gray scale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gabor Filter
        gabor_kernel = cv2.getGaborKernel((31, 31), sigma = 5, theta = np.pi/4, lambd = 10, gamma = 0.5, psi = 0, ktype=cv2.CV_32F)
        gabor_filtered = cv2.filter2D(gray_image, cv2.CV_8U, gabor_kernel)

        # Apply Gaussian Filter
        gaussian_filtered = cv2.GaussianBlur(gabor_filtered, (15, 15), 0)

        # Apply triangle thresholding 
        _, thresholded_image = cv2.threshold(gaussian_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
        c_array.append(len(contours))
        if image_folder in ['Cotton','Denim','Leather','Linen','Silk','Wool']:
            f_array.append(1)
        else:
            f_array.append(0)

x = np.array(c_array)
X = x.reshape(-1,1)
y = np.array(f_array)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create dataset for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters for LightGBM
params = {
    'objective': 'binary',  # Binary classification problem
    'metric': 'binary_error',  # Metric to optimize (binary classification error)
    'boosting_type': 'gbdt',  # Gradient boosting decision tree
    'num_leaves': 31,  # Number of leaves in each tree
    'learning_rate': 0.05,  # Step size shrinkage used to prevent overfitting
    'feature_fraction': 0.9,  # Percentage of features to be used in each boosting round
    'bagging_fraction': 0.8,  # Percentage of data to be used for each boosting round
    'bagging_freq': 5,  # Frequency for bagging
    'verbose': 0  # Verbosity (0 means no output during training)
}

# Train the model
num_round = 1000  # Number of boosting rounds
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# Make predictions on the test set
predictions = bst.predict(X_test, num_iteration=bst.best_iteration)
binary_predictions = [1 if p > 0.5 else 0 for p in predictions]  # Convert probabilities to binary predictions

#Check the performance of the model
accuracy = accuracy_score(y_test, binary_predictions)
f1 = f1_score(y_test, binary_predictions)
loss = hinge_loss(y_test, binary_predictions)

print("Accuracy:", accuracy)
print("F1-Score:", f1)
print('Loss:',loss)


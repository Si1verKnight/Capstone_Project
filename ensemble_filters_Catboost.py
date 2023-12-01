import cv2
import os
import numpy as np
from catboost import CatBoostClassifier
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

# Initialize the CatBoostClassifier
model = CatBoostClassifier(iterations = 70,  # Number of boosting iterations
                           learning_rate=0.1,  # Learning rate
                           depth=6,  # Depth of the trees
                           loss_function='Logloss',  # Loss function for binary classification
                           custom_metric='AUC',  # Metric to monitor during training
                           random_seed=42)  # Random seed for reproducibility

# Train the model
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=100)

# Make predictions on the test set
y_pred = model.predict(X_test)

#Check the performance of the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
loss = hinge_loss(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1-Score:", f1)
print('Loss:',loss)

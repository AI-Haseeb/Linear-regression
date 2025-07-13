import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

# print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

# print(diabetes.data.shape)
# (442, 10)

# print(diabetes.data)
# [[ 0.03807591  0.05068012  0.06169621 ... -0.0442235   -0.03482076
#   -0.00259226]

# print(diabetes.DESCR)

# diabetes_x = diabetes.data[:, np.newaxis, 2]  # Use only one feature (BMI), BMI is the 3rd feature (index 2)
diabetes_x = diabetes.data

diabetes_x_train = diabetes_x[:-30]  # Use all but the last 30 samples for training
diabetes_x_test = diabetes_x[-30:]  # Use the last 30 samples for testing

diabetes_y_train = diabetes.target[:-30]  # Use all but the last 30 target values for training
diabetes_y_test = diabetes.target[-30:]  # Use the last 30 target values for testing

model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)  # Train the model

diabetes_y_predicted = model.predict(diabetes_x_test)  # Make predictions

print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))  # Calculate mean squared error
print("Weights: ", model.coef_)  # Print the weights (coefficients)
print("Intercept: ", model.intercept_)  # Print the intercept term of the model 

plt.scatter(diabetes_x_test, diabetes_y_test)  # Plot the test data
plt.plot(diabetes_x_test, diabetes_y_predicted) # Plot the predicted line
plt.show()  # Show the plot

# Output: for only one feature (BMI)
# Mean squared error is:  3035.060115291269
# Weights:  [941.43097333]
# Intercept:  153.39713623331644

#output: for all features
# Mean squared error is:  1826.4841712795044
# Weights:  [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
#   458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
# Intercept:  153.05824267739402
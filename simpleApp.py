
import streamlit as st
import webbrowser
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



# Assuming this is a placeholder for your more complex ML model.
def BILSTM(stock_code):
    # Simulate processing. In a real scenario, this function would involve data processing and model prediction.
    return f"Processed {stock_code}"

# Create a text input field for user input.
stockcode = st.text_input("Enter stock code", '')

# Create a button that users can press to submit their input.
if st.button("Submit"):
    if stockcode:  # Check if the input is not empty
        st.write('Processing input...')  # Corrected typo from 'precessing' to 'processing'
        # Display the input for verification
        st.write("Stock code is:", stockcode)
        # Call the BILSTM function with the user's input and display the result
        result = BILSTM(stockcode)
        st.write(result)  # Display the output from the BILSTM function
    else:
        st.error("Please enter a stock code to proceed.")  # Show an error message if the input is empty



# Step 1: Generate synthetic data
np.random.seed(0)
X = 2.5 * np.random.randn(100) + 1.5   # Array of 100 values with mean = 1.5, stddev = 2.5
res = 0.5 * np.random.randn(100)       # Generate 100 residual terms
y = 2 + 0.3 * X + res                  # Actual values of Y

# Step 2: Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Reshape data
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Step 3: Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions using the testing set
y_pred = model.predict(X_test)

# Step 5: Plot outputs
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Predicted')
plt.title('Linear Regression Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

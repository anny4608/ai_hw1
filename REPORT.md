# Project Report: Linear Regression Streamlit Application

## 1. Project Overview

The objective of this project was to develop a web-based application to demonstrate linear regression. The application, built with Python and Streamlit, provides two main functionalities:
1.  **Synthetic Data Analysis**: Allows users to generate their own data, fit a linear regression model, and visualize the results, including the identification of outliers.
2.  **Iris Dataset Analysis**: Allows users to perform linear regression on the classic Iris dataset to predict one of its numerical features based on the others.

## 2. Implemented Features

The application includes the following features:

### Part A: Synthetic Data
- **Interactive Controls**: Sliders and input fields to control the data generation parameters (slope, noise, sample size, and random seed).
- **Linear Regression Modeling**: Fits a scikit-learn `LinearRegression` model to the generated data.
- **Performance Metrics**: Reports the learned slope, intercept, R² score, and Root Mean Squared Error (RMSE).
- **Interactive Plot**: A scatter plot of the data points with the fitted regression line drawn in red.
- **Outlier Detection**: Identifies and labels the top 5 outliers based on their absolute residuals.
- **Data Export**: A download button to export the generated dataset (including predictions and residuals) as a CSV file.

### Part B: Iris Dataset
- **Target Variable Selection**: A dropdown menu to select which of the four numerical features of the Iris dataset to predict.
- **Preprocessing Pipeline**: Uses a `StandardScaler` to normalize the features before fitting the model.
- **Model Coefficients**: Displays the learned coefficients and intercept of the model.
- **Performance Metrics**: Shows the R² score and RMSE for the model's predictions on the test set.
- **Diagnostic Plots**:
    - **Predicted vs. Actual Plot**: To visually assess the model's accuracy.
    - **Residuals vs. Predicted Plot**: To analyze the variance of the model's errors.
- **Data Export**: A download button to export the test set predictions and residuals as a CSV file.

## 3. Development and Debugging Process

The development process involved creating the Streamlit application from scratch, including the UI, data generation, modeling, and plotting components.

During the deployment phase, several technical challenges were encountered and resolved:
- **Environment Issues**: The initial attempts to run the application were hampered by issues related to the system's PATH and multiple Python installations. This was resolved by identifying the correct Python executable.
- **Interactive Prompts**: The application failed to start due to an interactive prompt from Streamlit on its first run. This was resolved by running the application in headless mode.
- **Code Errors**: An `IndentationError` in the application code was identified and fixed.

## 4. Final Status

The Streamlit application is fully functional and meets all the specified requirements. The application is running and accessible at http://localhost:8501.

## 5. How to Run the Application

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Streamlit app:**
    ```bash
    python -m streamlit run app.py
    ```

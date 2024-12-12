import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import plot_importance


# Load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Remove extra space from column name
    df.columns = df.columns.str.strip()
    # Shape of the dataset
    print("\nShape of the dataset:", df.shape)
    
    # Column names and data types
    print("\nData Types and Non-Null Counts:")
    df.info()
    
    # Checking for missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    print("\nColumns with missing values:")
    print(missing_values)
    
    # Drop rows where 'n_unique_tokens' > 1
    df = df[df['n_unique_tokens'] <= 1]
    # drop the URL column
    df = df.drop(columns=['url'])

    return df


def drop_features(df, feature_list):
    """
    Drop redundant or unnecessary features from a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame from which features are to be dropped.
        feature_list (list): List of column names to drop.
    
    Returns:
        pd.DataFrame: Updated DataFrame with specified columns removed.
    
    Raises:
        ValueError: If the DataFrame or feature list is invalid, or if any feature is missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input `df` must be a pandas DataFrame.")
    if not isinstance(feature_list, list):
        raise ValueError("Input `feature_list` must be a list of column names.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not feature_list:
        raise ValueError("Feature list is empty.")
    
    # Check for missing columns in the DataFrame
    missing_features = [feature for feature in feature_list if feature not in df.columns]
    if missing_features:
        raise ValueError(f"The following columns are not in the DataFrame: {missing_features}")
    
    # Drop the specified features
    df = df.drop(columns=feature_list)
    return df


def split_data(df, target_column='shares', scale=False):
    """
    Splits a dataset into training and testing sets, with an optional scaling transformation.

    Parameters:
    df (pd.DataFrame): The input dataset.
    target_column (str): The name of the target column. Default is 'shares'.
    scale (bool): Whether to scale the feature set using StandardScaler. Default is False.

    Returns:
    tuple: X_train, X_test, y_train, y_test
        - X_train, X_test: Feature sets for training and testing.
        - y_train, y_test: Target sets for training and testing.
    """
    # Separate features (X) and target variable (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply scaling transformation if required
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)  # Fit and transform on training data
        X_test = scaler.transform(X_test)       # Transform test data using the same scaler

    return X_train, X_test, y_train, y_test


def log_transform_column(df, column_name):
    """
    Applies a log transformation to a specified column in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to transform.
        column_name (str): The name of the column to log-transform.
    
    Returns:
        pd.DataFrame: A DataFrame with the transformed column.
    
    Raises:
        ValueError: If the input DataFrame or column is invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input `df` must be a pandas DataFrame.")
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' is not in the DataFrame.")
    if not np.issubdtype(df[column_name].dtype, np.number):
        raise ValueError(f"Column '{column_name}' must contain numeric values.")
    if (df[column_name] <= 0).any():
        raise ValueError(f"Column '{column_name}' contains non-positive values. Log transformation requires positive values.")
    
    # Apply log transformation
    df[column_name] = df[column_name].apply(np.log)
    return df


def reverse_log_transform_column(df, column_name):
    """
    Applies a reverse log transformation (exponentiation) to a specified column in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the column to transform.
        column_name (str): The name of the column to reverse log-transform.
    
    Returns:
        pd.DataFrame: A DataFrame with the reverse log-transformed column.
    
    Raises:
        ValueError: If the input DataFrame or column is invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input `df` must be a pandas DataFrame.")
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' is not in the DataFrame.")
    if not np.issubdtype(df[column_name].dtype, np.number):
        raise ValueError(f"Column '{column_name}' must contain numeric values.")
    
    # Apply reverse log transformation (exponentiation)
    df[column_name] = df[column_name].apply(np.exp)
    return df


def evaluate_model(X_test, y_test, y_pred, plot_results=False):
    """
    Evaluates the performance of a regression model on test data.

    Parameters:
    model: object
        The regression model to evaluate (must implement a `predict` method).
    X_test: array-like
        Feature set for testing the model.
    y_test: array-like
        Ground truth target values for the test set.
    plot_results: bool, optional (default=False)
        Whether to plot predicted vs. actual values.

    Returns:
    dict
        A dictionary containing the calculated evaluation metrics.
    """

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)                       # Root Mean Squared Error
    mape = mean_absolute_percentage_error(y_test, y_pred)  # Mean Absolute Percentage Error
    r2 = r2_score(y_test, y_pred)             # Coefficient of Determination (R-squared)

    # Print evaluation metrics
    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"R-Squared (R²): {r2:.4f}")

    # Optional: Plot predicted vs actual values
    if plot_results:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k', label='Predictions')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Return metrics as a dictionary
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
    }

    return metrics


def create_model(df):

    print("\n\n\n*************** Training Extreme Gradient Boosting Regressor ***************\n")
   
    # df is a DataFrame with features and target column 'shares'
    X = df.drop(columns=['shares'])  # Features
    y = df['shares']  # Target
    
    # Log-transform the target variable to stabilize variance
    y = np.log1p(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost DMatrices (optional for better performance in XGBoost)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # XGBoost model parameters
    params = {
        "objective": "reg:squarederror",  # Use squared error for regression
        "learning_rate": 0.1,            # Step size shrinkage
        "max_depth": 6,                  # Maximum tree depth
        "n_estimators": 100,             # Number of boosting rounds
        "subsample": 0.8,                # Fraction of samples for each tree
        "colsample_bytree": 0.8,         # Fraction of features for each tree
        "seed": 42                       # Random seed for reproducibility
    }
    
    # Initialize and train the XGBoost regressor
    xgb_model = xgb.XGBRegressor(
        objective=params["objective"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=params["seed"]
    )
    
    # Train the model
    xgb_model.fit(
        X_train,
        y_train
    )
    
    # Predict on test data
    y_pred_log = xgb_model.predict(X_test)
    
    # Back-transform predictions to original scale
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)
    
    # Evaluate the model
    print("*********** Extreme Gradient Boosting Regressor Evaluation Metrics ***********")
    xgb_model_metrics = evaluate_model(X_test, y_test_actual, y_pred, plot_results=False)

    return xgb_model, xgb_model_metrics
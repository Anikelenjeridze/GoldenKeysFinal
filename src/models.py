import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict


"""

This module contains functions for training and evaluating ML models.
"""

def prepare_data(df: pd.DataFrame, target_col: str, 
                feature_cols: list, test_size: float = 0.2,
                random_state: int = 42) -> Tuple:
    """
    Prepare data for machine learning by splitting into train and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    feature_cols : list
        List of feature column names
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple
        X_train, X_test, y_train, y_test
    """
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train) -> LinearRegression:



    """
    Train a Linear Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
        
    Returns:
    --------
    LinearRegression
        Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model trained successfully!")
    return model


def train_decision_tree(X_train, y_train, max_depth: int = 10,
                       min_samples_split: int = 20,
                       min_samples_leaf: int = 10,
                       random_state: int = 42) -> DecisionTreeRegressor:
    """
    Train a Decision Tree Regressor model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    max_depth : int
        Maximum depth of the tree
    min_samples_split : int
        Minimum samples required to split
    min_samples_leaf : int
        Minimum samples required in a leaf
    random_state : int
        Random seed
        
    Returns:
    --------
    DecisionTreeRegressor
        Trained model
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("Decision Tree model trained successfully!")
    return model


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> Dict:
    """
    Evaluate a trained model and return metrics.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    Dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }
    
    # Print results
    print(f"\n{model_name} Performance:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return metrics


def get_feature_importance(model, feature_names: list, model_type: str = 'linear'):
    """
    Get feature importance from a trained model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    feature_names : list
        List of feature names
    model_type : str
        Type of model ('linear' or 'tree')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with features and their importance
    """
    if model_type == 'linear':
        importance = model.coef_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': importance
        }).sort_values('Coefficient', key=abs, ascending=False)
    
    elif model_type == 'tree':
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
    
    return importance_df


def compare_models(metrics_list: list) -> pd.DataFrame:
    """
    Compare multiple models based on their metrics.
    
    Parameters:
    -----------
    metrics_list : list
        List of dictionaries containing model metrics
        
    Returns:
    --------
    pd.DataFrame
        Comparison dataframe
    """
    comparison_data = []
    
    for metrics in metrics_list:
        comparison_data.append({
            'Model': metrics['model_name'],
            'MSE': metrics['mse'],
            'RMSE': metrics['rmse'],
            'MAE': metrics['mae'],
            'R²': metrics['r2']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_idx = comparison_df['R²'].idxmax()
    best_model = comparison_df.loc[best_model_idx, 'Model']
    print(f"\nBest Model: {best_model}")
    
    return comparison_df


def make_single_prediction(model, features: dict, feature_names: list):
    """
    Make a prediction for a single instance.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    features : dict
        Dictionary of feature values
    feature_names : list
        List of expected feature names in order
        
    Returns:
    --------
    float
        Predicted value
    """

    
    # Create feature array in correct order
    feature_array = np.array([[features[name] for name in feature_names]])
    
    prediction = model.predict(feature_array)[0]
    
    return prediction


def calculate_residuals(y_true, y_pred):
    """
    Calculate residuals and residual statistics.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    Dict
        Dictionary containing residual statistics
    """
    residuals = y_true - y_pred
    
    residual_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'median': np.median(residuals)
    }
    
    print("\nResidual Statistics:")
    for key, value in residual_stats.items():
        print(f"  {key.capitalize()}: {value:.4f}")
    
    return residual_stats







if __name__ == "__main__":
    print("Machine Learning Models Module")
    print("Import this module to use ML model functions")
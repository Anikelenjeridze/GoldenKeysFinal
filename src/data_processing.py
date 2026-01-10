import pandas as pd
import numpy as np
from typing import Tuple, List

"""
This module contains functions for cleaning and preprocessing the housing data.
"""
def load_data(filepath: str) -> pd.DataFrame:



    
    """
    Load housing data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.Series
        Count of missing values per column
    """
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing)
    return missing


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> Tuple[int, float, float]:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to check for outliers
        
    Returns:
    --------
    Tuple[int, float, float]
        Count of outliers, lower bound, upper bound
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return len(outliers), lower_bound, upper_bound


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with new features added
    """
    df_new = df.copy()
    
    # Avoid division by zero
    df_new['AveOccup'] = df_new['AveOccup'].replace(0, np.nan)
    df_new['HouseAge'] = df_new['HouseAge'].replace(0, np.nan)
    
    # Create new features
    df_new['RoomsPerHousehold'] = df_new['AveRooms'] / df_new['AveOccup']
    df_new['BedroomsPerHousehold'] = df_new['AveBedrms'] / df_new['AveOccup']
    df_new['PopulationPerHousehold'] = df_new['Population'] / df_new['HouseAge']
    
    # Replace infinite values with median
    for col in ['RoomsPerHousehold', 'BedroomsPerHousehold', 'PopulationPerHousehold']:
        if np.isinf(df_new[col]).any():
            median_val = df_new[col].replace([np.inf, -np.inf], np.nan).median()
            df_new[col].replace([np.inf, -np.inf], median_val, inplace=True)
    
    # Fill NaN values with median
    df_new.fillna(df_new.median(), inplace=True)
    
    print(f"Created {df_new.shape[1] - df.shape[1]} new features")
    
    return df_new


def remove_outliers(df: pd.DataFrame, columns: List[str], method='iqr') -> pd.DataFrame:
    """
    Remove outliers from specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of column names to check for outliers
    method : str, optional
        Method to use ('iqr' or 'zscore')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers removed
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    if method == 'iqr':
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    removed_rows = initial_rows - len(df_clean)
    print(f"Removed {removed_rows} rows ({removed_rows/initial_rows*100:.2f}%)")
    
    return df_clean


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get comprehensive summary of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    return summary


def save_cleaned_data(df: pd.DataFrame, filepath: str) -> bool:
    """
    Save cleaned dataframe to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    filepath : str
        Output file path
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        df.to_csv(filepath, index=False)
        print(f"Data saved successfully to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("Data Processing Module")
    print("Import this module to use data processing functions")
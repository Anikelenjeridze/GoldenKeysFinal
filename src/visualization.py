import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
"""

This module contains functions for creating various data visualizations.
"""

def set_plot_style(style='whitegrid'):
    """
    Set the default plotting style.
    
    Parameters:
    -----------
    style : str
        Seaborn style name
    """
    sns.set_style(style)
    plt.rcParams['figure.figsize'] = (12, 6)


def plot_distribution(df: pd.DataFrame, column: str, bins: int = 50, 
                     color: str = 'steelblue', save_path: Optional[str] = None):
    """
    Plot distribution of a single variable with histogram and box plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to plot
    bins : int
        Number of bins for histogram
    color : str
        Color for the plot
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df[column], bins=bins, color=color, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of {column}')
    axes[0].axvline(df[column].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df[column].mean():.2f}')
    axes[0].axvline(df[column].median(), color='green', linestyle='--', 
                   label=f'Median: {df[column].median():.2f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df[column], vert=True)
    axes[1].set_ylabel(column)
    axes[1].set_title(f'Box Plot of {column}')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None,
                             save_path: Optional[str] = None):
    """
    Plot correlation heatmap for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str], optional
        List of columns to include
    save_path : str, optional
        Path to save the figure
    """
    if columns:
        correlation_matrix = df[columns].corr()
    else:
        correlation_matrix = df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_scatter_with_trend(df: pd.DataFrame, x_col: str, y_col: str,
                           title: Optional[str] = None, save_path: Optional[str] = None):
    """
    Create scatter plot with trend line.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 6))
    
    plt.scatter(df[x_col], df[y_col], alpha=0.3, c=df[y_col], 
               cmap='viridis', s=10)
    plt.colorbar(label=y_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'{y_col} vs {x_col}')
    
    # Add trend line
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    plt.plot(df[x_col].sort_values(), p(df[x_col].sort_values()), 
            "r--", linewidth=2, label='Trend line')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_geographic_distribution(df: pd.DataFrame, lat_col: str, lon_col: str,
                                value_col: str, size_col: Optional[str] = None,
                                save_path: Optional[str] = None):
    """
    Create geographic scatter plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name
    value_col : str
        Column to color-code points
    size_col : str, optional
        Column to size-code points
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    if size_col:
        sizes = df[size_col] / df[size_col].max() * 100
    else:
        sizes = 20
    
    scatter = plt.scatter(df[lon_col], df[lat_col], 
                         c=df[value_col], cmap='YlOrRd', 
                         alpha=0.4, s=sizes)
    plt.colorbar(scatter, label=value_col)
    plt.xlabel(lon_col)
    plt.ylabel(lat_col)
    plt.title(f'Geographic Distribution of {value_col}')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_multiple_distributions(df: pd.DataFrame, columns: List[str],
                                rows: int = 2, cols: int = 3,
                                save_path: Optional[str] = None):
    """
    Plot distributions for multiple columns in a grid.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        List of column names to plot
    rows : int
        Number of rows in the grid
    cols : int
        Number of columns in the grid
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        if idx < len(axes):
            axes[idx].hist(df[col], bins=50, color='teal', 
                          edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].axvline(df[col].median(), color='red', 
                            linestyle='--', label=f'Median: {df[col].median():.2f}')
            axes[idx].legend()
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prediction_comparison(y_true, y_pred, model_name: str,
                              save_path: Optional[str] = None):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    plt.plot([y_true.min(), y_true.max()], 
            [y_true.min(), y_true.max()], 
            'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals(y_true, y_pred, model_name: str,
                  save_path: Optional[str] = None):
    """
    Plot residual analysis.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual histogram
    axes[0].hist(residuals, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Residuals')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{model_name} - Residual Distribution')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].grid(alpha=0.3)
    
    # Residual scatter plot
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name} - Residual Plot')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    print("Visualization Module")
    print("Import this module to use visualization functions")
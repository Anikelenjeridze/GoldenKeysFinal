__version__ = "1.0.0"
__author__ = "[Your Name]"
"""


This package contains modules for data processing, visualization, and modeling.
"""
# Import main functions for easy access
from .data_processing import (
    load_data,
    check_missing_values,
    detect_outliers_iqr,
    create_derived_features,
    save_cleaned_data
)

from .visualization import (
    set_plot_style,
    plot_distribution,
    plot_correlation_heatmap,
    plot_scatter_with_trend
)

from .models import (
    prepare_data,
    train_linear_regression,
    train_decision_tree,
    evaluate_model,
    compare_models
)

__all__ = [
    # Data processing
    'load_data',
    'check_missing_values',
    'detect_outliers_iqr',
    'create_derived_features',
    'save_cleaned_data',
    # Visualization
    'set_plot_style',
    'plot_distribution',
    'plot_correlation_heatmap',
    'plot_scatter_with_trend',
    # Models
    'prepare_data',
    'train_linear_regression',
    'train_decision_tree',
    'evaluate_model',
    'compare_models'
]
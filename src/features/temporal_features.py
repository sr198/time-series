"""
Temporal feature engineering for bandwidth prediction.
Extracted from the original notebook to ensure consistency.
"""

import numpy as np
import pandas as pd


def create_temporal_features(df):
    """
    Create temporal features from datetime index.

    Args:
        df (pd.DataFrame): DataFrame with datetime index

    Returns:
        pd.DataFrame: DataFrame with additional temporal features
    """
    df = df.copy()

    # Basic temporal features
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofmonth"] = df.index.day

    # Binary features
    df["is_weekend"] = df.index.dayofweek == 5  # Saturday only (Nepal weekend)
    df["is_friday"] = df.index.dayofweek == 4
    df["is_month_start"] = df.index.is_month_start
    df["is_month_end"] = df.index.is_month_end
    df["is_year_start"] = df.index.is_year_start
    df["is_year_end"] = df.index.is_year_end

    # Cyclical features for seasonality
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def get_temporal_feature_names():
    """
    Get list of temporal feature column names.

    Returns:
        list: List of temporal feature names
    """
    return [
        "dayofweek", "month", "dayofmonth",
        "is_weekend", "is_friday", "is_month_start", "is_month_end",
        "is_year_start", "is_year_end",
        "sin_dayofweek", "cos_dayofweek", "sin_month", "cos_month"
    ]
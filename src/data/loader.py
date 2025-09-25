"""
Data loading and preprocessing for bandwidth prediction.
Handles loading bandwidth data and combining with event features.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import os

from ..features.temporal_features import create_temporal_features, get_temporal_feature_names
from ..features.event_features import (
    load_and_process_events, create_event_features, get_event_feature_names
)


class BandwidthDataLoader:
    """Data loader for bandwidth prediction with event integration."""

    def __init__(self, bandwidth_file: str, events_file: str):
        """
        Initialize the data loader.

        Args:
            bandwidth_file (str): Path to bandwidth CSV file
            events_file (str): Path to events CSV file
        """
        self.bandwidth_file = bandwidth_file
        self.events_file = events_file
        self.events_df = None
        self._load_events()

    def _load_events(self):
        """Load and process events data."""
        if os.path.exists(self.events_file):
            self.events_df = load_and_process_events(self.events_file)
            print(f"Loaded {len(self.events_df)} events from {self.events_file}")
        else:
            print(f"Events file not found: {self.events_file}")
            self.events_df = pd.DataFrame()

    def load_bandwidth_data(self) -> pd.DataFrame:
        """
        Load raw bandwidth data.

        Returns:
            pd.DataFrame: Raw bandwidth data with datetime index
        """
        df = pd.read_csv(self.bandwidth_file)

        # Keep only required columns
        required_cols = ['startdate', 'item', 'service_type', 'bandwidth_in_gbps', 'peak_bandwidth_utilization']
        df = df[required_cols]

        # Set datetime index
        df = df.set_index("startdate")
        df.index = pd.to_datetime(df.index)

        return df

    def get_available_combinations(self) -> List[Tuple[str, str]]:
        """
        Get all available item/service_type combinations.

        Returns:
            List[Tuple[str, str]]: List of (item, service_type) combinations
        """
        df = self.load_bandwidth_data()
        combinations = df.groupby(['item', 'service_type']).size().reset_index(name='count')
        return [(row['item'], row['service_type']) for _, row in combinations.iterrows()]

    def filter_combination(self, df: pd.DataFrame, item: str, service_type: str) -> pd.DataFrame:
        """
        Filter data for specific item/service_type combination.

        Args:
            df (pd.DataFrame): Bandwidth data
            item (str): Item name (e.g., 'Google')
            service_type (str): Service type (e.g., 'cache')

        Returns:
            pd.DataFrame: Filtered data sorted by date
        """
        filtered_df = df[(df['item'] == item) & (df['service_type'] == service_type)]
        return filtered_df.sort_index()

    def create_features(self, df: pd.DataFrame, include_events: bool = True,
                       lookback_days: int = 7) -> pd.DataFrame:
        """
        Create all features for the dataset.

        Args:
            df (pd.DataFrame): Input data with datetime index
            include_events (bool): Whether to include event features
            lookback_days (int): Lookback window for event features

        Returns:
            pd.DataFrame: Data with all features added
        """
        # Create temporal features
        df_features = create_temporal_features(df)

        # Add event features if available and requested
        if include_events and not self.events_df.empty:
            df_features = create_event_features(df_features, self.events_df, lookback_days)

        return df_features

    def prepare_training_data(self, item: str, service_type: str,
                            train_end_date: str = '2025-08-22',
                            include_events: bool = True,
                            lookback_days: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and testing data for a specific combination.

        Args:
            item (str): Item name
            service_type (str): Service type
            train_end_date (str): Date to split train/test
            include_events (bool): Whether to include event features
            lookback_days (int): Lookback window for event features

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames
        """
        # Load and filter data
        df = self.load_bandwidth_data()
        df_filtered = self.filter_combination(df, item, service_type)

        if df_filtered.empty:
            raise ValueError(f"No data found for combination: {item}/{service_type}")

        # Create features
        df_features = self.create_features(df_filtered, include_events, lookback_days)

        # Split train/test
        train = df_features.loc[df_features.index < train_end_date]
        test = df_features.loc[df_features.index >= train_end_date]

        return train, test

    def get_feature_columns(self, include_events: bool = True, lookback_days: int = 7) -> List[str]:
        """
        Get list of all feature column names.

        Args:
            include_events (bool): Whether to include event features
            lookback_days (int): Lookback window for event features

        Returns:
            List[str]: List of feature column names
        """
        features = get_temporal_feature_names()

        if include_events and not self.events_df.empty:
            features.extend(get_event_feature_names(lookback_days))

        return features

    def create_future_features(self, last_date: pd.Timestamp, n_days: int,
                             include_events: bool = True, lookback_days: int = 7) -> pd.DataFrame:
        """
        Create features for future prediction dates.

        Args:
            last_date (pd.Timestamp): Last date in the training data
            n_days (int): Number of future days to create features for
            include_events (bool): Whether to include event features
            lookback_days (int): Lookback window for event features

        Returns:
            pd.DataFrame: Future features DataFrame
        """
        # Create future date range
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=n_days,
            freq="D"
        )

        # Create empty DataFrame with future dates
        future_df = pd.DataFrame(index=future_index)

        # Generate features
        future_features = self.create_features(future_df, include_events, lookback_days)

        return future_features
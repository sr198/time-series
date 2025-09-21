"""
Exogenous event feature engineering for bandwidth prediction.
Processes events from internet_event_details.csv into categorical and numeric features.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple


class EventCategorizer:
    """Categorizes network events into predefined categories for feature engineering."""

    def __init__(self):
        self.categories = {
            'infrastructure_down': [
                'down', 'bgp down', 'link down', 'iplc.*down', 'ipt.*down'
            ],
            'infrastructure_up': [
                'up', 'restored', 'link up'
            ],
            'capacity_increase': [
                r'\+\d+g', r'\+\d+gbps', 'additional', 'upgrade'
            ],
            'capacity_decrease': [
                r'-\d+g', r'-\d+gbps', 'expiry', 'removal'
            ],
            'traffic_issue': [
                'traffic issue', 'fluctuate', 'brust traffic', 'no traffic'
            ],
            'external_event': [
                'protest', 'anniversary', 'jatra', 'constitution day', 'ipl', 'loadshedding'
            ],
            'power_issue': [
                'power issue', 'loadshedding', 'power'
            ],
            'node_issue': [
                'node issue', 'node problem', 'akamai.*issue'
            ]
        }

    def categorize_event(self, event_text: str) -> List[str]:
        """
        Categorize an event based on its description.

        Args:
            event_text (str): Event description text

        Returns:
            List[str]: List of applicable categories
        """
        event_lower = event_text.lower()
        applicable_categories = []

        for category, patterns in self.categories.items():
            for pattern in patterns:
                if re.search(pattern, event_lower):
                    applicable_categories.append(category)
                    break

        return applicable_categories if applicable_categories else ['other']

    def extract_capacity_change(self, event_text: str) -> float:
        """
        Extract capacity change amount from event text.

        Args:
            event_text (str): Event description

        Returns:
            float: Capacity change in Gbps (positive for increase, negative for decrease)
        """
        # Look for patterns like +18Gbps, +10G, -18Gbps
        increase_pattern = r'\+(\d+)g(?:bps)?'
        decrease_pattern = r'-(\d+)g(?:bps)?'

        increase_match = re.search(increase_pattern, event_text.lower())
        decrease_match = re.search(decrease_pattern, event_text.lower())

        if increase_match:
            return float(increase_match.group(1))
        elif decrease_match:
            return -float(decrease_match.group(1))

        return 0.0


def load_and_process_events(events_file: str) -> pd.DataFrame:
    """
    Load and process event data into features.

    Args:
        events_file (str): Path to internet_event_details.csv

    Returns:
        pd.DataFrame: Processed events with datetime index and feature columns
    """
    # Load events data
    events_df = pd.read_csv(events_file)

    # Parse datetime with error handling
    try:
        events_df['Datetime'] = pd.to_datetime(events_df['Datetime'], format='%m/%d/%y')
    except ValueError:
        # Try alternative formats if the first one fails
        events_df['Datetime'] = pd.to_datetime(events_df['Datetime'], infer_datetime_format=True)

    events_df = events_df.set_index('Datetime')

    # Ensure the index is properly sorted
    events_df = events_df.sort_index()

    categorizer = EventCategorizer()
    processed_events = []

    for date, event_text in zip(events_df.index, events_df['Event']):
        categories = categorizer.categorize_event(event_text)
        capacity_change = categorizer.extract_capacity_change(event_text)

        event_record = {
            'date': date,
            'capacity_change': capacity_change
        }

        # Add binary features for each category
        for category in categorizer.categories.keys():
            event_record[f'event_{category}'] = 1 if category in categories else 0

        # Special case for 'other' category
        event_record['event_other'] = 1 if 'other' in categories else 0

        processed_events.append(event_record)

    result_df = pd.DataFrame(processed_events).set_index('date')

    # Ensure all event columns are numeric
    for col in result_df.columns:
        if col.startswith('event_') or col == 'capacity_change':
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0)

    return result_df


def create_event_features(df: pd.DataFrame, events_df: pd.DataFrame,
                         lookback_days: int = 7) -> pd.DataFrame:
    """
    Create event-based features for bandwidth prediction.

    Args:
        df (pd.DataFrame): Main DataFrame with datetime index
        events_df (pd.DataFrame): Processed events DataFrame
        lookback_days (int): Number of days to look back for event impact

    Returns:
        pd.DataFrame: DataFrame with event features added
    """
    df = df.copy()

    # Return early if events_df is empty
    if events_df.empty:
        print("Warning: No events data available")
        return df

    # Ensure both DataFrames have datetime indices
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if not isinstance(events_df.index, pd.DatetimeIndex):
        events_df.index = pd.to_datetime(events_df.index)

    # Get all event feature columns
    event_columns = [col for col in events_df.columns if col.startswith('event_')]
    event_columns.append('capacity_change')

    # Initialize event features (both current and lookback)
    for col in event_columns:
        df[col] = 0.0  # Current day features
        df[f'{col}_recent_{lookback_days}d'] = 0.0  # Lookback features

    # For each date in the main DataFrame
    for original_date in df.index:
        # Ensure date is a Timestamp
        date = pd.Timestamp(original_date) if not isinstance(original_date, pd.Timestamp) else original_date

        # Look for events on this date
        if date in events_df.index:
            day_events = events_df.loc[events_df.index == date]

            # Aggregate events for the same day
            for col in event_columns:
                if col == 'capacity_change':
                    df.loc[original_date, col] = float(day_events[col].sum())
                else:
                    # Ensure numeric comparison
                    col_sum = pd.to_numeric(day_events[col], errors='coerce').fillna(0).sum()
                    df.loc[original_date, col] = 1.0 if col_sum > 0 else 0.0

        # Add features for recent events (lookback window)
        try:
            lookback_start = date - pd.Timedelta(days=lookback_days)
            recent_events = events_df.loc[
                (events_df.index > lookback_start) & (events_df.index < date)
            ]
        except TypeError as e:
            print(f"Date comparison error: {e}")
            print(f"Date type: {type(date)}, Events index type: {type(events_df.index)}")
            # Skip this date if there's a type error
            continue

        if not recent_events.empty:
            for col in event_columns:
                lookback_col = f'{col}_recent_{lookback_days}d'
                if col == 'capacity_change':
                    df.loc[original_date, lookback_col] = float(recent_events[col].sum())
                else:
                    # Ensure numeric comparison
                    col_sum = pd.to_numeric(recent_events[col], errors='coerce').fillna(0).sum()
                    df.loc[original_date, lookback_col] = 1.0 if col_sum > 0 else 0.0

    return df


def get_event_feature_names(lookback_days: int = 7) -> List[str]:
    """
    Get list of event feature column names.

    Args:
        lookback_days (int): Lookback window for recent events

    Returns:
        List[str]: List of event feature names
    """
    categorizer = EventCategorizer()
    base_features = []

    # Current day event features
    for category in categorizer.categories.keys():
        base_features.append(f'event_{category}')
    base_features.extend(['event_other', 'capacity_change'])

    # Recent event features
    recent_features = []
    for feature in base_features:
        recent_features.append(f'{feature}_recent_{lookback_days}d')

    return base_features + recent_features
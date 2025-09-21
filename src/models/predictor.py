"""
Model prediction module for bandwidth forecasting.
Handles loading trained models and making predictions.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from ..data.loader import BandwidthDataLoader


class BandwidthPredictor:
    """Model predictor for bandwidth forecasting."""

    def __init__(self, data_loader: BandwidthDataLoader, models_dir: str = "models"):
        """
        Initialize the predictor.

        Args:
            data_loader (BandwidthDataLoader): Data loader instance
            models_dir (str): Directory containing trained models
        """
        self.data_loader = data_loader
        self.models_dir = models_dir
        self._loaded_models = {}  # Cache for loaded models

    def load_model(self, model_path: str) -> Dict:
        """
        Load a trained model from file.

        Args:
            model_path (str): Path to the model file

        Returns:
            Dict: Loaded model data including model, metadata, and features
        """
        if model_path in self._loaded_models:
            return self._loaded_models[model_path]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self._loaded_models[model_path] = model_data
        return model_data

    def find_model_for_combination(self, item: str, service_type: str) -> Optional[str]:
        """
        Find the most recent model for a specific item/service_type combination.

        Args:
            item (str): Item name
            service_type (str): Service type

        Returns:
            Optional[str]: Path to the model file, or None if not found
        """
        models = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith('_metadata.json'):
                filepath = os.path.join(self.models_dir, filename)
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                    if (metadata.get('item') == item and
                        metadata.get('service_type') == service_type):
                        model_file = metadata.get('model_file')
                        if model_file:
                            model_path = os.path.join(self.models_dir, model_file)
                            if os.path.exists(model_path):
                                models.append((metadata.get('training_date', ''), model_path))

        if not models:
            return None

        # Return the most recently trained model
        models.sort(key=lambda x: x[0], reverse=True)
        return models[0][1]

    def predict_historical(self, model_path: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Make predictions for historical date range.

        Args:
            model_path (str): Path to the trained model
            start_date (str): Start date for predictions
            end_date (str): End date for predictions

        Returns:
            pd.DataFrame: Predictions with dates and values
        """
        model_data = self.load_model(model_path)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        metadata = model_data['metadata']

        # Get the combination info
        item = metadata['item']
        service_type = metadata['service_type']
        include_events = metadata['include_events']
        lookback_days = metadata['lookback_days']

        # Load and filter data for the combination
        df = self.data_loader.load_bandwidth_data()
        df_filtered = self.data_loader.filter_combination(df, item, service_type)

        # Filter for the requested date range
        df_period = df_filtered.loc[start_date:end_date]

        if df_period.empty:
            raise ValueError(f"No data available for {item}/{service_type} in date range {start_date} to {end_date}")

        # Create features
        df_features = self.data_loader.create_features(df_period, include_events, lookback_days)

        # Align features with model expectations
        X_pred = df_features[feature_columns]

        # Make predictions
        predictions = model.predict(X_pred)

        # Create results DataFrame
        results = pd.DataFrame({
            'date': df_features.index,
            'actual': df_features['peak_bandwidth_utilization'],
            'predicted': predictions,
            'error': df_features['peak_bandwidth_utilization'] - predictions,
            'abs_error': np.abs(df_features['peak_bandwidth_utilization'] - predictions)
        })

        results['item'] = item
        results['service_type'] = service_type

        return results

    def predict_future(self, model_path: str, n_days: int = 14) -> pd.DataFrame:
        """
        Make future predictions.

        Args:
            model_path (str): Path to the trained model
            n_days (int): Number of days to predict

        Returns:
            pd.DataFrame: Future predictions with dates and values
        """
        model_data = self.load_model(model_path)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        metadata = model_data['metadata']

        # Get the combination info
        item = metadata['item']
        service_type = metadata['service_type']
        include_events = metadata['include_events']
        lookback_days = metadata['lookback_days']

        # Get the last date from the original training data
        df = self.data_loader.load_bandwidth_data()
        df_filtered = self.data_loader.filter_combination(df, item, service_type)
        last_date = df_filtered.index.max()

        # Create future features
        future_features = self.data_loader.create_future_features(
            last_date, n_days, include_events, lookback_days
        )

        # Align features with model expectations
        X_future = future_features[feature_columns]

        # Make predictions
        predictions = model.predict(X_future)

        # Create results DataFrame
        results = pd.DataFrame({
            'date': future_features.index,
            'predicted': predictions
        })

        results['item'] = item
        results['service_type'] = service_type

        return results

    def predict_single_date(self, model_path: str, target_date: str) -> Dict:
        """
        Make prediction for a single date.

        Args:
            model_path (str): Path to the trained model
            target_date (str): Date to predict for

        Returns:
            Dict: Prediction result with metadata
        """
        target_datetime = pd.to_datetime(target_date)
        model_data = self.load_model(model_path)
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        metadata = model_data['metadata']

        # Get the combination info
        item = metadata['item']
        service_type = metadata['service_type']
        include_events = metadata['include_events']
        lookback_days = metadata['lookback_days']

        # Create features for the target date
        single_date_df = pd.DataFrame(index=[target_datetime])
        features_df = self.data_loader.create_features(single_date_df, include_events, lookback_days)

        # Align features with model expectations
        X_pred = features_df[feature_columns]

        # Make prediction
        prediction = model.predict(X_pred)[0]

        return {
            'date': target_date,
            'predicted_bandwidth': prediction,
            'item': item,
            'service_type': service_type,
            'model_metrics': model_data['metrics'],
            'confidence': self._calculate_confidence(model_data, prediction)
        }

    def _calculate_confidence(self, model_data: Dict, prediction: float) -> str:
        """
        Calculate confidence level for prediction based on model performance.

        Args:
            model_data (Dict): Model data including metrics
            prediction (float): Predicted value

        Returns:
            str: Confidence level
        """
        test_mape = model_data['metrics'].get('test_mape', 100)

        if test_mape < 5:
            return "High"
        elif test_mape < 15:
            return "Medium"
        else:
            return "Low"

    def get_model_performance(self, model_path: str) -> Dict:
        """
        Get performance metrics for a trained model.

        Args:
            model_path (str): Path to the model file

        Returns:
            Dict: Model performance metrics and metadata
        """
        model_data = self.load_model(model_path)

        return {
            'metadata': model_data['metadata'],
            'metrics': model_data['metrics'],
            'feature_count': len(model_data['feature_columns']),
            'features': model_data['feature_columns']
        }

    def compare_models(self, model_paths: List[str]) -> pd.DataFrame:
        """
        Compare performance metrics of multiple models.

        Args:
            model_paths (List[str]): List of model file paths

        Returns:
            pd.DataFrame: Comparison table with metrics
        """
        comparisons = []

        for model_path in model_paths:
            try:
                perf = self.get_model_performance(model_path)
                comparison = {
                    'model_path': os.path.basename(model_path),
                    'item': perf['metadata']['item'],
                    'service_type': perf['metadata']['service_type'],
                    'test_rmse': perf['metrics']['test_rmse'],
                    'test_mae': perf['metrics']['test_mae'],
                    'test_mape': perf['metrics']['test_mape'],
                    'train_samples': perf['metadata']['train_samples'],
                    'test_samples': perf['metadata']['test_samples'],
                    'n_features': perf['feature_count'],
                    'include_events': perf['metadata']['include_events'],
                    'training_date': perf['metadata']['training_date']
                }
                comparisons.append(comparison)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")

        return pd.DataFrame(comparisons)
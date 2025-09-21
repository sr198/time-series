"""
Model training module with hyperparameter optimization for bandwidth prediction.
Supports training models for different item/service_type combinations.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import logging
import pickle
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ..data.loader import BandwidthDataLoader


class BandwidthModelTrainer:
    """Model trainer for bandwidth prediction with hyperparameter optimization."""

    def __init__(self, data_loader: BandwidthDataLoader, models_dir: str = "models"):
        """
        Initialize the model trainer.

        Args:
            data_loader (BandwidthDataLoader): Data loader instance
            models_dir (str): Directory to save trained models
        """
        self.data_loader = data_loader
        self.models_dir = models_dir
        self.target_column = "peak_bandwidth_utilization"

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Suppress Optuna logging
        optuna.logging.set_verbosity(logging.WARNING)

    def _create_objective_function(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series):
        """
        Create Optuna objective function for hyperparameter optimization.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Callable: Objective function for Optuna
        """
        def objective(trial):
            # Define hyperparameter search space
            param = {
                "n_estimators": 1000,
                "early_stopping_rounds": 50,
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
                "eval_metric": "rmse"
            }

            # Train model with suggested parameters
            model = xgb.XGBRegressor(**param)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            # Calculate RMSE on test set
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            return rmse

        return objective

    def train_model(self, item: str, service_type: str,
                   train_end_date: str = '2025-08-22',
                   include_events: bool = True,
                   lookback_days: int = 7,
                   n_trials: int = 50) -> Dict:
        """
        Train a model for specific item/service_type combination.

        Args:
            item (str): Item name
            service_type (str): Service type
            train_end_date (str): Date to split train/test
            include_events (bool): Whether to include event features
            lookback_days (int): Lookback window for event features
            n_trials (int): Number of Optuna trials

        Returns:
            Dict: Training results including model, metrics, and metadata
        """
        print(f"Training model for {item}/{service_type}...")

        # Prepare data
        train_df, test_df = self.data_loader.prepare_training_data(
            item, service_type, train_end_date, include_events, lookback_days
        )

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(f"Insufficient data for {item}/{service_type}")

        # Get feature columns
        feature_columns = self.data_loader.get_feature_columns(include_events, lookback_days)

        # Prepare training data
        X_train = train_df[feature_columns]
        y_train = train_df[self.target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[self.target_column]

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(feature_columns)}")

        # Hyperparameter optimization
        print(f"Running hyperparameter optimization with {n_trials} trials...")
        study = optuna.create_study(direction="minimize")
        objective_func = self._create_objective_function(X_train, y_train, X_test, y_test)
        study.optimize(objective_func, n_trials=n_trials)

        # Train final model with best parameters
        best_params = study.best_params
        best_params.update({
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "eval_metric": "rmse"
        })

        print(f"Best RMSE: {study.best_value:.4f}")
        print("Training final model with best parameters...")

        model = xgb.XGBRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Calculate metrics
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_preds)),
            'train_mae': mean_absolute_error(y_train, train_preds),
            'test_mae': mean_absolute_error(y_test, test_preds),
            'best_optuna_rmse': study.best_value
        }

        # Calculate MAPE
        def mape(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        metrics['train_mape'] = mape(y_train, train_preds)
        metrics['test_mape'] = mape(y_test, test_preds)

        # Create training results
        results = {
            'model': model,
            'metrics': metrics,
            'best_params': best_params,
            'feature_columns': feature_columns,
            'metadata': {
                'item': item,
                'service_type': service_type,
                'train_end_date': train_end_date,
                'include_events': include_events,
                'lookback_days': lookback_days,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': len(feature_columns),
                'training_date': datetime.now().isoformat()
            }
        }

        print(f"Training completed!")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
        print(f"Test MAPE: {metrics['test_mape']:.2f}%")

        return results

    def save_model(self, results: Dict, filename: Optional[str] = None) -> str:
        """
        Save trained model and metadata.

        Args:
            results (Dict): Training results from train_model
            filename (str, optional): Custom filename

        Returns:
            str: Path to saved model file
        """
        metadata = results['metadata']

        if filename is None:
            # Create filename from metadata
            item_clean = metadata['item'].replace(' ', '_').replace('/', '_')
            service_clean = metadata['service_type'].replace(' ', '_').replace('/', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{item_clean}_{service_clean}_{timestamp}.pkl"

        filepath = os.path.join(self.models_dir, filename)

        # Save model and metadata
        save_data = {
            'model': results['model'],
            'metrics': results['metrics'],
            'best_params': results['best_params'],
            'feature_columns': results['feature_columns'],
            'metadata': metadata
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        # Save metadata as JSON for easy inspection
        json_filepath = filepath.replace('.pkl', '_metadata.json')
        with open(json_filepath, 'w') as f:
            json_metadata = {k: v for k, v in metadata.items()}
            json_metadata['metrics'] = results['metrics']
            json_metadata['model_file'] = filename
            json.dump(json_metadata, f, indent=2)

        print(f"Model saved to: {filepath}")
        print(f"Metadata saved to: {json_filepath}")

        return filepath

    def train_and_save_model(self, item: str, service_type: str, **kwargs) -> str:
        """
        Train and save a model in one step.

        Args:
            item (str): Item name
            service_type (str): Service type
            **kwargs: Additional arguments for train_model

        Returns:
            str: Path to saved model file
        """
        results = self.train_model(item, service_type, **kwargs)
        return self.save_model(results)

    def list_saved_models(self) -> List[Dict]:
        """
        List all saved models with their metadata.

        Returns:
            List[Dict]: List of model metadata
        """
        models = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith('_metadata.json'):
                filepath = os.path.join(self.models_dir, filename)
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                    models.append(metadata)

        return sorted(models, key=lambda x: x.get('training_date', ''), reverse=True)
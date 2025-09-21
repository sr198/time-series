"""
Command-line interface for bandwidth prediction application.
Provides commands for training, prediction, and model management.
"""

import argparse
import sys
import os
import pandas as pd
from typing import List

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.loader import BandwidthDataLoader
from src.models.trainer import BandwidthModelTrainer
from src.models.predictor import BandwidthPredictor


class BandwidthCLI:
    """Command-line interface for bandwidth prediction."""

    def __init__(self, bandwidth_file: str = "sample_data/internet_details.csv",
                 events_file: str = "sample_data/internet_event_details.csv"):
        """
        Initialize CLI with data files.

        Args:
            bandwidth_file (str): Path to bandwidth data CSV
            events_file (str): Path to events data CSV
        """
        self.data_loader = BandwidthDataLoader(bandwidth_file, events_file)
        self.trainer = BandwidthModelTrainer(self.data_loader)
        self.predictor = BandwidthPredictor(self.data_loader)

    def list_combinations(self):
        """List all available item/service_type combinations."""
        print("Available item/service_type combinations:")
        print("-" * 50)

        combinations = self.data_loader.get_available_combinations()
        for i, (item, service_type) in enumerate(combinations, 1):
            print(f"{i:2d}. {item} / {service_type}")

        print(f"\nTotal combinations: {len(combinations)}")

    def train_model(self, item: str, service_type: str, include_events: bool = True,
                   lookback_days: int = 7, n_trials: int = 50):
        """
        Train a model for specific combination.

        Args:
            item (str): Item name
            service_type (str): Service type
            include_events (bool): Include event features
            lookback_days (int): Event lookback window
            n_trials (int): Optuna optimization trials
        """
        try:
            print(f"Training model for: {item} / {service_type}")
            print(f"Include events: {include_events}")
            print(f"Lookback days: {lookback_days}")
            print(f"Optimization trials: {n_trials}")
            print("-" * 50)

            model_path = self.trainer.train_and_save_model(
                item=item,
                service_type=service_type,
                include_events=include_events,
                lookback_days=lookback_days,
                n_trials=n_trials
            )

            print(f"\nModel training completed successfully!")
            print(f"Model saved to: {model_path}")

        except Exception as e:
            print(f"Error training model: {e}")
            return False

        return True

    def predict_future(self, item: str, service_type: str, n_days: int = 14):
        """
        Make future predictions for a combination.

        Args:
            item (str): Item name
            service_type (str): Service type
            n_days (int): Number of days to predict
        """
        try:
            # Find model for combination
            model_path = self.predictor.find_model_for_combination(item, service_type)
            if not model_path:
                print(f"No trained model found for {item}/{service_type}")
                print("Please train a model first using the train command.")
                return False

            print(f"Making {n_days}-day forecast for: {item} / {service_type}")
            print(f"Using model: {os.path.basename(model_path)}")
            print("-" * 50)

            # Make predictions
            predictions = self.predictor.predict_future(model_path, n_days)

            # Display results
            print("\nFuture Bandwidth Predictions:")
            print(predictions.to_string(index=False, float_format='%.2f'))

            # Calculate some statistics
            mean_pred = predictions['predicted'].mean()
            max_pred = predictions['predicted'].max()
            min_pred = predictions['predicted'].min()

            print(f"\nSummary Statistics:")
            print(f"Mean predicted bandwidth: {mean_pred:.2f} Gbps")
            print(f"Maximum predicted bandwidth: {max_pred:.2f} Gbps")
            print(f"Minimum predicted bandwidth: {min_pred:.2f} Gbps")

        except Exception as e:
            print(f"Error making predictions: {e}")
            return False

        return True

    def predict_date(self, item: str, service_type: str, date: str):
        """
        Make prediction for a specific date.

        Args:
            item (str): Item name
            service_type (str): Service type
            date (str): Date to predict for (YYYY-MM-DD)
        """
        try:
            # Find model for combination
            model_path = self.predictor.find_model_for_combination(item, service_type)
            if not model_path:
                print(f"No trained model found for {item}/{service_type}")
                print("Please train a model first using the train command.")
                return False

            print(f"Predicting bandwidth for {date}: {item} / {service_type}")
            print(f"Using model: {os.path.basename(model_path)}")
            print("-" * 50)

            # Make prediction
            result = self.predictor.predict_single_date(model_path, date)

            # Display result
            print(f"\nPrediction Results:")
            print(f"Date: {result['date']}")
            print(f"Predicted Bandwidth: {result['predicted_bandwidth']:.2f} Gbps")
            print(f"Confidence: {result['confidence']}")
            print(f"Model Test MAPE: {result['model_metrics']['test_mape']:.2f}%")

        except Exception as e:
            print(f"Error making prediction: {e}")
            return False

        return True

    def list_models(self):
        """List all saved models."""
        print("Saved Models:")
        print("-" * 80)

        models = self.trainer.list_saved_models()
        if not models:
            print("No saved models found.")
            return

        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model['item']} / {model['service_type']}")
            print(f"    File: {model.get('model_file', 'N/A')}")
            print(f"    Test RMSE: {model['metrics']['test_rmse']:.4f}")
            print(f"    Test MAPE: {model['metrics']['test_mape']:.2f}%")
            print(f"    Events: {'Yes' if model['include_events'] else 'No'}")
            print(f"    Trained: {model['training_date']}")
            print()

    def compare_models(self):
        """Compare performance of all saved models."""
        print("Model Performance Comparison:")
        print("-" * 100)

        # Get all model files
        model_files = []
        for filename in os.listdir(self.trainer.models_dir):
            if filename.endswith('.pkl'):
                model_files.append(os.path.join(self.trainer.models_dir, filename))

        if not model_files:
            print("No saved models found.")
            return

        # Compare models
        comparison = self.predictor.compare_models(model_files)
        if not comparison.empty:
            # Sort by test RMSE
            comparison = comparison.sort_values('test_rmse')

            # Display comparison table
            display_cols = ['item', 'service_type', 'test_rmse', 'test_mape',
                          'include_events', 'n_features', 'training_date']
            print(comparison[display_cols].to_string(index=False, float_format='%.4f'))
        else:
            print("No valid models found for comparison.")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Bandwidth Prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available combinations
  python -m src.app.cli list-combinations

  # Train a model
  python -m src.app.cli train --item "Google" --service-type "cache"

  # Make future predictions
  python -m src.app.cli predict-future --item "Google" --service-type "cache" --days 14

  # Predict for specific date
  python -m src.app.cli predict-date --item "Google" --service-type "cache" --date "2025-09-01"

  # List saved models
  python -m src.app.cli list-models

  # Compare models
  python -m src.app.cli compare-models
        """
    )

    # Add data file arguments
    parser.add_argument('--bandwidth-file', default='sample_data/internet_details.csv',
                       help='Path to bandwidth CSV file')
    parser.add_argument('--events-file', default='sample_data/internet_event_details.csv',
                       help='Path to events CSV file')

    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List combinations command
    subparsers.add_parser('list-combinations', help='List available item/service_type combinations')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--item', required=True, help='Item name')
    train_parser.add_argument('--service-type', required=True, help='Service type')
    train_parser.add_argument('--no-events', action='store_true', help='Exclude event features')
    train_parser.add_argument('--lookback-days', type=int, default=7, help='Event lookback window')
    train_parser.add_argument('--trials', type=int, default=50, help='Optuna optimization trials')

    # Predict future command
    predict_future_parser = subparsers.add_parser('predict-future', help='Make future predictions')
    predict_future_parser.add_argument('--item', required=True, help='Item name')
    predict_future_parser.add_argument('--service-type', required=True, help='Service type')
    predict_future_parser.add_argument('--days', type=int, default=14, help='Number of days to predict')

    # Predict date command
    predict_date_parser = subparsers.add_parser('predict-date', help='Predict for specific date')
    predict_date_parser.add_argument('--item', required=True, help='Item name')
    predict_date_parser.add_argument('--service-type', required=True, help='Service type')
    predict_date_parser.add_argument('--date', required=True, help='Date to predict (YYYY-MM-DD)')

    # Model management commands
    subparsers.add_parser('list-models', help='List all saved models')
    subparsers.add_parser('compare-models', help='Compare model performance')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    try:
        cli = BandwidthCLI(args.bandwidth_file, args.events_file)
    except Exception as e:
        print(f"Error initializing CLI: {e}")
        return

    # Execute commands
    try:
        if args.command == 'list-combinations':
            cli.list_combinations()

        elif args.command == 'train':
            cli.train_model(
                item=args.item,
                service_type=args.service_type,
                include_events=not args.no_events,
                lookback_days=args.lookback_days,
                n_trials=args.trials
            )

        elif args.command == 'predict-future':
            cli.predict_future(
                item=args.item,
                service_type=args.service_type,
                n_days=args.days
            )

        elif args.command == 'predict-date':
            cli.predict_date(
                item=args.item,
                service_type=args.service_type,
                date=args.date
            )

        elif args.command == 'list-models':
            cli.list_models()

        elif args.command == 'compare-models':
            cli.compare_models()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error executing command: {e}")


if __name__ == '__main__':
    main()
"""
Interactive interface for bandwidth prediction.
Provides a user-friendly way to interact with the prediction system.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.loader import BandwidthDataLoader
from src.models.trainer import BandwidthModelTrainer
from src.models.predictor import BandwidthPredictor


class InteractiveBandwidthPredictor:
    """Interactive interface for bandwidth prediction."""

    def __init__(self, bandwidth_file: str = "sample_data/internet_details.csv",
                 events_file: str = "sample_data/internet_event_details.csv"):
        """
        Initialize interactive predictor.

        Args:
            bandwidth_file (str): Path to bandwidth data CSV
            events_file (str): Path to events data CSV
        """
        print("🌐 Bandwidth Prediction System")
        print("=" * 50)

        try:
            self.data_loader = BandwidthDataLoader(bandwidth_file, events_file)
            self.trainer = BandwidthModelTrainer(self.data_loader)
            self.predictor = BandwidthPredictor(self.data_loader)
            print("✅ System initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            raise

    def show_main_menu(self):
        """Display main menu options."""
        print("\n" + "=" * 50)
        print("📊 BANDWIDTH PREDICTION MAIN MENU")
        print("=" * 50)
        print("1. 📋 View available combinations")
        print("2. 🔧 Train a new model")
        print("3. 🔮 Make predictions")
        print("4. 📈 Manage models")
        print("5. ❌ Exit")
        print("-" * 50)

    def show_prediction_menu(self):
        """Display prediction options menu."""
        print("\n" + "-" * 40)
        print("🔮 PREDICTION OPTIONS")
        print("-" * 40)
        print("1. 📅 Predict for specific date")
        print("2. 🔮 Future forecast (multiple days)")
        print("3. 📊 Historical analysis")
        print("4. ⬅️  Back to main menu")
        print("-" * 40)

    def show_model_menu(self):
        """Display model management menu."""
        print("\n" + "-" * 40)
        print("📈 MODEL MANAGEMENT")
        print("-" * 40)
        print("1. 📋 List saved models")
        print("2. 🔍 Compare model performance")
        print("3. 📊 View model details")
        print("4. ⬅️  Back to main menu")
        print("-" * 40)

    def view_combinations(self):
        """Show available item/service_type combinations."""
        print("\n📋 Available Item/Service Type Combinations:")
        print("-" * 60)

        try:
            combinations = self.data_loader.get_available_combinations()
            for i, (item, service_type) in enumerate(combinations, 1):
                print(f"{i:2d}. {item:<25} | {service_type}")

            print(f"\n📊 Total combinations: {len(combinations)}")

        except Exception as e:
            print(f"❌ Error loading combinations: {e}")

    def select_combination(self):
        """Interactive combination selection."""
        combinations = self.data_loader.get_available_combinations()

        print("\n🎯 Select a combination:")
        for i, (item, service_type) in enumerate(combinations, 1):
            print(f"{i:2d}. {item} / {service_type}")

        while True:
            try:
                choice = input(f"\nEnter choice (1-{len(combinations)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(combinations):
                    return combinations[idx]
                else:
                    print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a valid number.")

    def train_model_interactive(self):
        """Interactive model training."""
        print("\n🔧 TRAIN NEW MODEL")
        print("-" * 30)

        try:
            # Select combination
            item, service_type = self.select_combination()
            print(f"\n✅ Selected: {item} / {service_type}")

            # Get training parameters
            include_events = self._get_yes_no("Include event features?", default=True)
            lookback_days = self._get_integer("Event lookback days", default=7, min_val=1, max_val=30)
            n_trials = self._get_integer("Optimization trials", default=50, min_val=10, max_val=200)

            print(f"\n⚙️  Training Configuration:")
            print(f"   Item/Service: {item} / {service_type}")
            print(f"   Include events: {'Yes' if include_events else 'No'}")
            print(f"   Lookback days: {lookback_days}")
            print(f"   Trials: {n_trials}")

            confirm = self._get_yes_no("\nProceed with training?", default=True)
            if not confirm:
                print("❌ Training cancelled.")
                return

            # Train model
            print("\n🚀 Starting model training...")
            model_path = self.trainer.train_and_save_model(
                item=item,
                service_type=service_type,
                include_events=include_events,
                lookback_days=lookback_days,
                n_trials=n_trials
            )

            print(f"\n✅ Model training completed!")
            print(f"📁 Model saved to: {os.path.basename(model_path)}")

        except Exception as e:
            print(f"❌ Error during training: {e}")

    def prediction_interface(self):
        """Interactive prediction interface."""
        while True:
            self.show_prediction_menu()
            choice = input("Enter your choice (1-4): ").strip()

            if choice == '1':
                self.predict_single_date()
            elif choice == '2':
                self.predict_future()
            elif choice == '3':
                self.predict_historical()
            elif choice == '4':
                break
            else:
                print("❌ Invalid choice. Please try again.")

    def predict_single_date(self):
        """Predict for a single date."""
        print("\n📅 PREDICT FOR SPECIFIC DATE")
        print("-" * 35)

        try:
            # Select combination
            item, service_type = self.select_combination()

            # Check if model exists
            model_path = self.predictor.find_model_for_combination(item, service_type)
            if not model_path:
                print(f"❌ No trained model found for {item}/{service_type}")
                train_now = self._get_yes_no("Would you like to train a model now?", default=True)
                if train_now:
                    print("🔄 Redirecting to training...")
                    # Could implement training here
                return

            print(f"✅ Found model: {os.path.basename(model_path)}")

            # Get date
            date_str = input("Enter date (YYYY-MM-DD): ").strip()
            try:
                pd.to_datetime(date_str)  # Validate date format
            except:
                print("❌ Invalid date format. Please use YYYY-MM-DD.")
                return

            # Make prediction
            print(f"\n🔮 Making prediction for {date_str}...")
            result = self.predictor.predict_single_date(model_path, date_str)

            # Display result
            print(f"\n📊 PREDICTION RESULT")
            print(f"{'='*30}")
            print(f"📅 Date: {result['date']}")
            print(f"🌐 Item/Service: {result['item']} / {result['service_type']}")
            print(f"📈 Predicted Bandwidth: {result['predicted_bandwidth']:.2f} Gbps")
            print(f"🎯 Confidence: {result['confidence']}")
            print(f"📊 Model MAPE: {result['model_metrics']['test_mape']:.2f}%")

        except Exception as e:
            print(f"❌ Error making prediction: {e}")

    def predict_future(self):
        """Make future predictions."""
        print("\n🔮 FUTURE FORECAST")
        print("-" * 25)

        try:
            # Select combination
            item, service_type = self.select_combination()

            # Check if model exists
            model_path = self.predictor.find_model_for_combination(item, service_type)
            if not model_path:
                print(f"❌ No trained model found for {item}/{service_type}")
                return

            print(f"✅ Found model: {os.path.basename(model_path)}")

            # Get forecast days
            n_days = self._get_integer("Number of days to forecast", default=14, min_val=1, max_val=90)

            # Make predictions
            print(f"\n🔮 Generating {n_days}-day forecast...")
            predictions = self.predictor.predict_future(model_path, n_days)

            # Display results
            print(f"\n📊 FUTURE PREDICTIONS")
            print(f"{'='*50}")
            print(f"🌐 Item/Service: {item} / {service_type}")
            print(f"📅 Forecast period: {n_days} days")
            print(f"\n{predictions[['date', 'predicted']].to_string(index=False, float_format='%.2f')}")

            # Summary statistics
            mean_pred = predictions['predicted'].mean()
            max_pred = predictions['predicted'].max()
            min_pred = predictions['predicted'].min()

            print(f"\n📈 SUMMARY STATISTICS")
            print(f"{'='*30}")
            print(f"📊 Average: {mean_pred:.2f} Gbps")
            print(f"⬆️  Maximum: {max_pred:.2f} Gbps")
            print(f"⬇️  Minimum: {min_pred:.2f} Gbps")

        except Exception as e:
            print(f"❌ Error making forecast: {e}")

    def predict_historical(self):
        """Historical prediction analysis."""
        print("\n📊 HISTORICAL ANALYSIS")
        print("-" * 30)

        try:
            # Select combination
            item, service_type = self.select_combination()

            # Check if model exists
            model_path = self.predictor.find_model_for_combination(item, service_type)
            if not model_path:
                print(f"❌ No trained model found for {item}/{service_type}")
                return

            print(f"✅ Found model: {os.path.basename(model_path)}")

            # Get date range
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD): ").strip()

            try:
                pd.to_datetime(start_date)
                pd.to_datetime(end_date)
            except:
                print("❌ Invalid date format. Please use YYYY-MM-DD.")
                return

            # Make predictions
            print(f"\n📊 Analyzing period: {start_date} to {end_date}")
            results = self.predictor.predict_historical(model_path, start_date, end_date)

            # Display results
            print(f"\n📈 HISTORICAL ANALYSIS RESULTS")
            print(f"{'='*50}")
            print(results[['date', 'actual', 'predicted', 'error']].to_string(
                index=False, float_format='%.2f'))

            # Summary statistics
            rmse = (results['error'] ** 2).mean() ** 0.5
            mae = results['abs_error'].mean()
            mape = (results['abs_error'] / results['actual']).mean() * 100

            print(f"\n📊 PERFORMANCE METRICS")
            print(f"{'='*30}")
            print(f"📏 RMSE: {rmse:.4f}")
            print(f"📐 MAE: {mae:.4f}")
            print(f"📊 MAPE: {mape:.2f}%")

        except Exception as e:
            print(f"❌ Error in historical analysis: {e}")

    def model_management(self):
        """Model management interface."""
        while True:
            self.show_model_menu()
            choice = input("Enter your choice (1-4): ").strip()

            if choice == '1':
                self.list_models()
            elif choice == '2':
                self.compare_models()
            elif choice == '3':
                self.view_model_details()
            elif choice == '4':
                break
            else:
                print("❌ Invalid choice. Please try again.")

    def list_models(self):
        """List all saved models."""
        print("\n📋 SAVED MODELS")
        print("-" * 25)

        try:
            models = self.trainer.list_saved_models()
            if not models:
                print("📭 No saved models found.")
                return

            for i, model in enumerate(models, 1):
                print(f"\n{i:2d}. {model['item']} / {model['service_type']}")
                print(f"    📁 File: {model.get('model_file', 'N/A')}")
                print(f"    📊 Test RMSE: {model['metrics']['test_rmse']:.4f}")
                print(f"    📈 Test MAPE: {model['metrics']['test_mape']:.2f}%")
                print(f"    🎯 Events: {'✅' if model['include_events'] else '❌'}")
                print(f"    📅 Trained: {model['training_date']}")

        except Exception as e:
            print(f"❌ Error listing models: {e}")

    def compare_models(self):
        """Compare model performance."""
        print("\n🔍 MODEL COMPARISON")
        print("-" * 25)

        try:
            # Get all model files
            model_files = []
            for filename in os.listdir(self.trainer.models_dir):
                if filename.endswith('.pkl'):
                    model_files.append(os.path.join(self.trainer.models_dir, filename))

            if not model_files:
                print("📭 No saved models found.")
                return

            # Compare models
            comparison = self.predictor.compare_models(model_files)
            if not comparison.empty:
                comparison = comparison.sort_values('test_rmse')

                print(f"\n📊 PERFORMANCE COMPARISON")
                print(f"{'='*70}")
                display_cols = ['item', 'service_type', 'test_rmse', 'test_mape', 'include_events']
                print(comparison[display_cols].to_string(index=False, float_format='%.4f'))
            else:
                print("❌ No valid models found for comparison.")

        except Exception as e:
            print(f"❌ Error comparing models: {e}")

    def view_model_details(self):
        """View detailed model information."""
        print("\n📊 MODEL DETAILS")
        print("-" * 20)

        try:
            # List available models
            models = self.trainer.list_saved_models()
            if not models:
                print("📭 No saved models found.")
                return

            print("Select a model to view details:")
            for i, model in enumerate(models, 1):
                print(f"{i:2d}. {model['item']} / {model['service_type']}")

            while True:
                try:
                    choice = input(f"\nEnter choice (1-{len(models)}): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        break
                    else:
                        print("❌ Invalid choice. Please try again.")
                except ValueError:
                    print("❌ Please enter a valid number.")

            # Load and display model details
            selected_model = models[idx]
            model_file = selected_model.get('model_file')
            if model_file:
                model_path = os.path.join(self.trainer.models_dir, model_file)
                performance = self.predictor.get_model_performance(model_path)

                print(f"\n📊 DETAILED MODEL INFORMATION")
                print(f"{'='*50}")
                print(f"🌐 Item/Service: {performance['metadata']['item']} / {performance['metadata']['service_type']}")
                print(f"📁 File: {model_file}")
                print(f"📅 Training Date: {performance['metadata']['training_date']}")
                print(f"📊 Train Samples: {performance['metadata']['train_samples']}")
                print(f"🧪 Test Samples: {performance['metadata']['test_samples']}")
                print(f"🔧 Features: {performance['feature_count']}")
                print(f"🎯 Include Events: {'✅' if performance['metadata']['include_events'] else '❌'}")

                print(f"\n📈 PERFORMANCE METRICS")
                print(f"{'='*30}")
                for metric, value in performance['metrics'].items():
                    print(f"📊 {metric.replace('_', ' ').title()}: {value:.4f}")

        except Exception as e:
            print(f"❌ Error viewing model details: {e}")

    def _get_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Get yes/no input from user."""
        default_str = "Y/n" if default else "y/N"
        while True:
            response = input(f"{prompt} ({default_str}): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            elif response == '':
                return default
            else:
                print("❌ Please enter 'y' for yes or 'n' for no.")

    def _get_integer(self, prompt: str, default: int, min_val: int = 1, max_val: int = 1000) -> int:
        """Get integer input from user."""
        while True:
            response = input(f"{prompt} (default {default}): ").strip()
            if response == '':
                return default
            try:
                value = int(response)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"❌ Please enter a value between {min_val} and {max_val}.")
            except ValueError:
                print("❌ Please enter a valid integer.")

    def run(self):
        """Run the interactive interface."""
        try:
            while True:
                self.show_main_menu()
                choice = input("Enter your choice (1-5): ").strip()

                if choice == '1':
                    self.view_combinations()
                elif choice == '2':
                    self.train_model_interactive()
                elif choice == '3':
                    self.prediction_interface()
                elif choice == '4':
                    self.model_management()
                elif choice == '5':
                    print("\n👋 Thank you for using Bandwidth Prediction System!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


def main():
    """Main function to run interactive interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Bandwidth Prediction Interface")
    parser.add_argument('--bandwidth-file', default='sample_data/internet_details.csv',
                       help='Path to bandwidth CSV file')
    parser.add_argument('--events-file', default='sample_data/internet_event_details.csv',
                       help='Path to events CSV file')

    args = parser.parse_args()

    try:
        app = InteractiveBandwidthPredictor(args.bandwidth_file, args.events_file)
        app.run()
    except Exception as e:
        print(f"❌ Failed to start interactive interface: {e}")


if __name__ == '__main__':
    main()
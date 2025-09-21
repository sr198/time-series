"""
Main entry point for the Bandwidth Prediction System.
Provides easy access to CLI and interactive interfaces.
"""

import sys
import argparse

def main():
    """Main entry point with mode selection."""
    parser = argparse.ArgumentParser(
        description="Bandwidth Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  interactive: Launch interactive GUI-like interface
  cli: Use command-line interface
  train: Quick training command
  predict: Quick prediction command

Examples:
  # Interactive mode
  python main.py interactive

  # CLI commands
  python main.py cli list-combinations
  python main.py cli list-models
  python main.py cli compare-models
  python main.py cli train --item "Google" --service-type "cache"
  python main.py cli predict-future --item "Google" --service-type "cache" --days 14
  python main.py cli predict-date --item "Google" --service-type "cache" --date "2025-09-15"

  # Quick commands (shortcuts)
  python main.py train --item "Google" --service-type "cache"
  python main.py predict --item "Google" --service-type "cache" --days 14
        """
    )

    parser.add_argument('mode', choices=['interactive', 'cli', 'train', 'predict'],
                       help='Interface mode to use')
    parser.add_argument('--bandwidth-file', default='sample_data/internet_details.csv',
                       help='Path to bandwidth CSV file')
    parser.add_argument('--events-file', default='sample_data/internet_event_details.csv',
                       help='Path to events CSV file')

    # Training arguments
    parser.add_argument('--item', help='Item name for training/prediction')
    parser.add_argument('--service-type', help='Service type for training/prediction')
    parser.add_argument('--no-events', action='store_true', help='Exclude event features')
    parser.add_argument('--lookback-days', type=int, default=7, help='Event lookback window')
    parser.add_argument('--trials', type=int, default=50, help='Optuna optimization trials')

    # Prediction arguments
    parser.add_argument('--days', type=int, default=14, help='Number of days to predict')
    parser.add_argument('--date', help='Specific date to predict (YYYY-MM-DD)')

    # Parse args
    args, remaining = parser.parse_known_args()

    if args.mode == 'interactive':
        # Launch interactive interface
        from src.app.interactive import InteractiveBandwidthPredictor
        try:
            app = InteractiveBandwidthPredictor(args.bandwidth_file, args.events_file)
            app.run()
        except Exception as e:
            print(f"❌ Error launching interactive mode: {e}")

    elif args.mode == 'cli':
        # Pass to CLI module
        from src.app.cli import main as cli_main
        # Reconstruct sys.argv for CLI
        sys.argv = ['cli.py', '--bandwidth-file', args.bandwidth_file,
                   '--events-file', args.events_file] + remaining
        cli_main()

    elif args.mode == 'train':
        # Quick training
        if not args.item or not args.service_type:
            print("❌ --item and --service-type are required for training mode")
            return

        from src.app.cli import BandwidthCLI
        try:
            cli = BandwidthCLI(args.bandwidth_file, args.events_file)
            cli.train_model(
                item=args.item,
                service_type=args.service_type,
                include_events=not args.no_events,
                lookback_days=args.lookback_days,
                n_trials=args.trials
            )
        except Exception as e:
            print(f"❌ Error in training mode: {e}")

    elif args.mode == 'predict':
        # Quick prediction
        if not args.item or not args.service_type:
            print("❌ --item and --service-type are required for prediction mode")
            return

        from src.app.cli import BandwidthCLI
        try:
            cli = BandwidthCLI(args.bandwidth_file, args.events_file)
            if args.date:
                cli.predict_date(args.item, args.service_type, args.date)
            else:
                cli.predict_future(args.item, args.service_type, args.days)
        except Exception as e:
            print(f"❌ Error in prediction mode: {e}")


if __name__ == '__main__':
    main()

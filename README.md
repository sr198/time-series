# Bandwidth Prediction System

A comprehensive solution for bandwidth utilization prediction with exogenous event features, supporting multiple item/service_type combinations with automated hyperparameter optimization using XGBoost and Optuna.

## Table of Contents

1. [Overview](#overview)
2. [Project Architecture](#project-architecture)
3. [Installation](#installation)
4. [Data Requirements](#data-requirements)
5. [Basic Usage](#basic-usage)
6. [Advanced Usage](#advanced-usage)
7. [Feature Engineering](#feature-engineering)
8. [Model Management](#model-management)
9. [Working with New Data](#working-with-new-data)
10. [Creating New Pipelines](#creating-new-pipelines)
11. [Performance Monitoring](#performance-monitoring)
12. [Troubleshooting](#troubleshooting)

## Overview

This system provides a production-ready solution for predicting bandwidth utilization across different network services and providers. It integrates temporal patterns with exogenous events (network outages, capacity changes, external events) to improve prediction accuracy.

### Key Features

- Multi-combination model support for different item/service_type pairs
- Automated hyperparameter optimization using Optuna
- Event-aware feature engineering from network incident logs
- Multiple user interfaces (CLI, interactive, notebook)
- Model persistence with metadata tracking
- Comprehensive evaluation metrics (RMSE, MAE, MAPE)

### Use Cases

- Network capacity planning and forecasting
- Bandwidth utilization monitoring
- Impact assessment of network events
- Predictive maintenance scheduling
- Resource allocation optimization

## Project Architecture

```
bandwidth_predictor/
├── src/
│   ├── features/                    # Feature engineering modules
│   │   ├── temporal_features.py     # Time-based feature extraction
│   │   └── event_features.py        # Event categorization and processing
│   ├── models/                      # Model training and prediction
│   │   ├── trainer.py               # XGBoost training with Optuna
│   │   └── predictor.py             # Model loading and inference
│   ├── data/                        # Data loading and preprocessing
│   │   └── loader.py                # Unified data pipeline
│   └── app/                         # User interfaces
│       ├── cli.py                   # Command-line interface
│       └── interactive.py           # Interactive menu system
├── models/                          # Saved model artifacts
├── notebooks/                       # Jupyter analysis notebooks
├── sample_data/                     # Example datasets
├── main.py                          # Main entry point
├── pyproject.toml                   # UV package configuration
└── README.md                        # This file
```

### Core Components

- **Data Loader**: Handles bandwidth data loading and event integration
- **Feature Engineering**: Creates temporal and event-based features
- **Model Trainer**: Trains XGBoost models with hyperparameter optimization
- **Model Predictor**: Loads models and generates predictions
- **User Interfaces**: CLI and interactive interfaces for end users

## Installation

### Prerequisites

- **Python 3.11 or higher** - Required for compatibility with all dependencies
- **Git** - For cloning the repository
- **uv** - Modern Python package manager (faster than pip)

### Setup Instructions

This project uses `uv` for package management. First, ensure you have `uv` installed:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and setup the project:

```bash
# Clone the repository
git clone https://github.com/sr198/time-series.git
cd time-series

# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Verify installation
python test_structure.py
```

### Development Installation

For development with additional tools:

```bash
# Install with development dependencies
uv sync --extra dev

# Install pre-commit hooks (optional)
pre-commit install
```

## Data Requirements

### Bandwidth Data File

**File**: `sample_data/internet_details.csv`

**Required Columns**:
- `startdate`: Date in YYYY-MM-DD format
- `item`: Item/provider name (e.g., "Google", "Meta", "Cloudflare")
- `service_type`: Service category (e.g., "cache", "IPLC", "IPT")
- `bandwidth_in_gbps`: Available bandwidth capacity
- `peak_bandwidth_utilization`: Target variable for prediction

**Example**:
```csv
startdate,item,service_type,bandwidth_in_gbps,peak_bandwidth_utilization
2025-06-15,Google,cache,274,182.04
2025-06-15,Meta,cache,400,196.48
2025-06-15,Cloudflare,cache,20,8.01
```

### Events Data File

**File**: `sample_data/internet_event_details.csv`

**Required Columns**:
- `Datetime`: Date in M/D/YY format
- `Event`: Free-text event description

**Example**:
```csv
Datetime,Event
1/6/25,Google IPLC 10G down at bhairahawa from 18:00
1/7/25,Google IPLC 10G UP at bhairahawa at 13:10
1/13/25,Tata IPT +18Gbps additional
```

### Event Categories

The system automatically categorizes events into:
- **infrastructure_down**: Network failures, link downs, BGP issues
- **infrastructure_up**: Service restorations
- **capacity_increase**: Bandwidth upgrades (+18Gbps, +10G)
- **capacity_decrease**: Bandwidth reductions, contract expirations
- **traffic_issue**: Traffic anomalies, routing problems
- **external_event**: Holidays, protests, special events
- **power_issue**: Power outages, load shedding
- **node_issue**: CDN node problems, equipment failures

## Basic Usage

### Quick Start Commands

```bash
# List available item/service_type combinations
python main.py cli list-combinations

# Train a model for Google cache service
python main.py train --item "Google" --service-type "cache"

# Generate 14-day forecast
python main.py predict --item "Google" --service-type "cache" --days 14

# Predict for specific date
python main.py predict --item "Google" --service-type "cache" --date "2025-09-15"
```

### Interactive Mode

For users preferring a menu-driven interface:

```bash
python main.py interactive
```

This provides guided workflows for:
- Exploring available data combinations
- Training models with custom parameters
- Making various types of predictions
- Managing saved models

### Command-Line Interface

For automation and scripting:

```bash
# List all CLI commands
python main.py cli --help

# Advanced training with custom parameters
python main.py cli train --item "Google" --service-type "cache" \
  --trials 100 --lookback-days 14 --no-events

# Model management
python main.py cli list-models
python main.py cli compare-models

# Make predictions
python main.py cli predict-future --item "Google" --service-type "cache" --days 14
python main.py cli predict-date --item "Google" --service-type "cache" --date "2025-09-15"
```

## Advanced Usage

### Custom Training Parameters

```bash
python main.py train \
  --item "Google" \
  --service-type "cache" \
  --trials 200 \              # Optuna optimization trials
  --lookback-days 10 \         # Event impact window
  --no-events                  # Exclude event features
```

### Batch Predictions

```bash
# Multiple day ranges
python main.py predict --item "Google" --service-type "cache" --days 7
python main.py predict --item "Google" --service-type "cache" --days 30
python main.py predict --item "Google" --service-type "cache" --days 90
```

### Historical Analysis

```bash
# For historical analysis, use the interactive mode
python main.py interactive
# Then select: Prediction Options -> Historical analysis

# Or use the predictor module programmatically (see examples below)
```

## Feature Engineering

### Temporal Features

The system automatically generates time-based features:

**Basic Temporal**:
- `dayofweek`: Day of week (0=Monday)
- `month`: Month (1-12)
- `dayofmonth`: Day of month (1-31)

**Binary Indicators**:
- `is_weekend`: Saturday flag (Nepal weekend)
- `is_friday`: Friday flag
- `is_month_start/end`: Month boundary flags
- `is_year_start/end`: Year boundary flags

**Cyclical Encoding**:
- `sin_dayofweek`, `cos_dayofweek`: Cyclical day representation
- `sin_month`, `cos_month`: Cyclical month representation

### Event Features

Events are processed into numerical features:

**Current Day Events**:
- `event_infrastructure_down`: Infrastructure failure flag
- `event_capacity_increase`: Capacity upgrade flag
- `capacity_change`: Numerical capacity change (Gbps)

**Recent Event Impact**:
- `event_*_recent_7d`: Event occurred in last 7 days
- `capacity_change_recent_7d`: Total capacity changes in last 7 days

## Model Management

### Model Storage

Trained models are saved in the `models/` directory with:
- **Model file**: Serialized XGBoost model (`.pkl`)
- **Metadata file**: Training parameters and metrics (`.json`)

### Model Naming Convention

```
{item}_{service_type}_{timestamp}.pkl
Google_cache_20250921_143022.pkl
```

### Model Metadata

Each model stores:
- Training parameters and hyperparameters
- Performance metrics (RMSE, MAE, MAPE)
- Feature list and engineering parameters
- Training/test dataset sizes
- Training timestamp

### Model Comparison

```bash
# Compare all saved models
python main.py cli compare-models

# Get detailed model information
python main.py interactive  # -> Model Management -> View Details
```

## Working with New Data

### Adding New Data Points

1. **Update Bandwidth Data**:
   ```bash
   # Append new rows to sample_data/internet_details.csv
   echo "2025-09-22,Google,cache,274,185.67" >> sample_data/internet_details.csv
   ```

2. **Update Events Data**:
   ```bash
   # Add new events to sample_data/internet_event_details.csv
   echo "9/22/25,Google cache maintenance scheduled" >> sample_data/internet_event_details.csv
   ```

3. **Retrain Models**:
   ```bash
   # Retrain with updated data
   python main.py train --item "Google" --service-type "cache"
   ```

### Adding New Item/Service Combinations

1. **Ensure Data Availability**:
   ```bash
   # Check available combinations
   python main.py cli list-combinations
   ```

2. **Train New Model**:
   ```bash
   # Train for new combination
   python main.py train --item "NewProvider" --service-type "newservice"
   ```

3. **Validate Performance**:
   ```bash
   # Compare with existing models
   python main.py cli compare-models
   ```

### Data Quality Checks

```python
# Programmatic data validation
from src.data.loader import BandwidthDataLoader

loader = BandwidthDataLoader('data/bandwidth.csv', 'data/events.csv')
combinations = loader.get_available_combinations()

for item, service_type in combinations:
    train_df, test_df = loader.prepare_training_data(item, service_type)
    print(f"{item}/{service_type}: {len(train_df)} train, {len(test_df)} test samples")
```

## Creating New Pipelines

### Custom Feature Engineering

1. **Extend Temporal Features**:
   ```python
   # In src/features/temporal_features.py
   def create_custom_temporal_features(df):
       df = create_temporal_features(df)  # Base features
       df['quarter'] = df.index.quarter
       df['week_of_year'] = df.index.isocalendar().week
       return df
   ```

2. **Custom Event Categories**:
   ```python
   # In src/features/event_features.py
   def add_custom_event_category(self):
       self.categories['maintenance'] = [
           'maintenance', 'planned outage', 'scheduled downtime'
       ]
   ```

### Custom Model Training

```python
from src.data.loader import BandwidthDataLoader
from src.models.trainer import BandwidthModelTrainer

# Initialize with custom parameters
data_loader = BandwidthDataLoader('data/bandwidth.csv', 'data/events.csv')
trainer = BandwidthModelTrainer(data_loader)

# Custom training configuration
results = trainer.train_model(
    item="Google",
    service_type="cache",
    include_events=True,
    lookback_days=14,  # Longer event impact window
    n_trials=200       # More optimization trials
)

# Save with custom naming
model_path = trainer.save_model(results, "google_cache_production_v2.pkl")
```

### Automated Retraining Pipeline

```bash
#!/bin/bash
# retrain_models.sh

# List of critical combinations to retrain
combinations=(
    "Google cache"
    "Meta cache"
    "Cloudflare cache"
)

for combo in "${combinations[@]}"; do
    IFS=' ' read -r item service_type <<< "$combo"
    echo "Retraining $item/$service_type..."

    python main.py train \
        --item "$item" \
        --service-type "$service_type" \
        --trials 100

    echo "Completed $item/$service_type"
done

echo "All models retrained successfully"
```

## Performance Monitoring

### Model Evaluation Metrics

- **RMSE (Root Mean Square Error)**: Overall prediction accuracy
- **MAE (Mean Absolute Error)**: Average prediction error
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error

### Confidence Levels

- **High Confidence**: MAPE < 5%
- **Medium Confidence**: 5% ≤ MAPE < 15%
- **Low Confidence**: MAPE ≥ 15%

### Monitoring Commands

```bash
# Check model performance
python main.py cli compare-models

# For historical analysis, use interactive mode or programmatic approach
python main.py interactive
# Select: Prediction Options -> Historical analysis
```

### Automated Performance Tracking

```python
# performance_monitor.py
from src.models.predictor import BandwidthPredictor
from src.data.loader import BandwidthDataLoader
import pandas as pd

def monitor_model_performance(model_path, days_back=30):
    loader = BandwidthDataLoader('data/bandwidth.csv', 'data/events.csv')
    predictor = BandwidthPredictor(loader)

    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')

    results = predictor.predict_historical(model_path, start_date, end_date)

    mape = (results['abs_error'] / results['actual']).mean() * 100
    rmse = (results['error'] ** 2).mean() ** 0.5

    print(f"Performance over last {days_back} days:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.4f}")

    return results
```

## Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure you're in the project root directory and virtual environment is activated
source .venv/bin/activate
python test_structure.py
```

**2. Data File Not Found**
```bash
# Check file paths
ls -la sample_data/
python main.py cli list-combinations  # Will show if data loads correctly
```

**3. No Trained Model Found**
```bash
# Train a model first
python main.py train --item "Google" --service-type "cache"

# List available models
python main.py cli list-models
```

**4. Poor Model Performance**
```bash
# Increase optimization trials
python main.py train --item "Google" --service-type "cache" --trials 200

# Include event features
python main.py train --item "Google" --service-type "cache" --lookback-days 14

# Check data quality
python main.py cli list-combinations  # Verify sufficient data points
```

**5. Memory Issues with Large Datasets**
```bash
# Reduce optimization trials
python main.py train --item "Google" --service-type "cache" --trials 25

# Train without events temporarily
python main.py train --item "Google" --service-type "cache" --no-events
```

### Debug Mode

```bash
# Run with verbose output
python -u main.py train --item "Google" --service-type "cache" --trials 10

# Test individual components
python test_structure.py
python -c "from src.data.loader import BandwidthDataLoader; print('Data loader works')"
```

### Getting Help

```bash
# Command help
python main.py --help
python main.py cli --help
python main.py train --help

# Interactive mode for guided usage
python main.py interactive
```

### Performance Optimization

**For Large Datasets**:
- Reduce Optuna trials for initial testing
- Use event feature engineering selectively
- Consider data sampling for development

**For Production**:
- Increase Optuna trials (100-500)
- Use longer event lookback windows (7-14 days)
- Implement automated retraining schedules

This comprehensive system provides a robust foundation for bandwidth prediction with the flexibility to adapt to changing data patterns and requirements.
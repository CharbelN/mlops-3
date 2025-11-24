#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Titanic Survival Prediction Pipeline"
echo "=========================================="

# Configuration
RAW_TRAIN="data/raw/train.csv"
RAW_TEST="data/raw/test.csv"
PROCESSED_TRAIN="data/processed/train_processed.csv"
PROCESSED_TEST="data/processed/test_processed.csv"
FEATURIZED_TRAIN="data/featurized/train_featurized.csv"
FEATURIZED_TEST="data/featurized/test_featurized.csv"
MODEL_PATH="models/model.pkl"
PREDICTIONS_PATH="predictions/predictions.csv"
METRICS_PATH="results/metrics.json"

# Step 1: Preprocess raw data
echo ""
echo "Step 1/5: Preprocessing data..."
echo "----------------------------------------"
uv run python scripts/preprocess.py \
  --input "$RAW_TRAIN" \
  --test_path "$RAW_TEST" \
  --output "$PROCESSED_TRAIN" \
  --output_test "$PROCESSED_TEST"

# Step 2: Feature engineering
echo ""
echo "Step 2/5: Engineering features..."
echo "----------------------------------------"
uv run python scripts/featurize.py \
  --input "$PROCESSED_TRAIN" \
  --output "$FEATURIZED_TRAIN"

uv run python scripts/featurize.py \
  --input "$PROCESSED_TEST" \
  --output "$FEATURIZED_TEST"

# Step 3: Train model
echo ""
echo "Step 3/5: Training model..."
echo "----------------------------------------"
uv run python scripts/train.py \
  --input "$FEATURIZED_TRAIN" \
  --output "$MODEL_PATH"

# Step 4: Evaluate model (if test labels available)
echo ""
echo "Step 4/5: Evaluating model..."
echo "----------------------------------------"
if grep -q "Survived" "$FEATURIZED_TEST" 2>/dev/null; then
  uv run python scripts/evaluate.py \
    --model "$MODEL_PATH" \
    --input "$FEATURIZED_TEST" \
    --output "$METRICS_PATH"
else
  echo "Skipping evaluation (no labels in test data)"
fi

# Step 5: Make predictions
echo ""
echo "Step 5/5: Making predictions..."
echo "----------------------------------------"
uv run python scripts/predict.py \
  --model "$MODEL_PATH" \
  --input "$FEATURIZED_TEST" \
  --output "$PREDICTIONS_PATH"

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Model saved to: $MODEL_PATH"
echo "Predictions saved to: $PREDICTIONS_PATH"
if [ -f "$METRICS_PATH" ]; then
  echo "Metrics saved to: $METRICS_PATH"
fi
echo "=========================================="

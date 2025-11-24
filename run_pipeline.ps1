$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Titanic Survival Prediction Pipeline" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Configuration
$RAW_TRAIN = "data/raw/train.csv"
$RAW_TEST = "data/raw/test.csv"
$PROCESSED_TRAIN = "data/processed/train_processed.csv"
$PROCESSED_TEST = "data/processed/test_processed.csv"
$FEATURIZED_TRAIN = "data/featurized/train_featurized.csv"
$FEATURIZED_TEST = "data/featurized/test_featurized.csv"
$MODEL_PATH = "models/model.pkl"
$PREDICTIONS_PATH = "predictions/predictions.csv"
$METRICS_PATH = "results/metrics.json"

# Step 1: Preprocess raw data
Write-Host ""
Write-Host "Step 1/5: Preprocessing data..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
uv run python scripts/preprocess.py `
  --input $RAW_TRAIN `
  --test_path $RAW_TEST `
  --output $PROCESSED_TRAIN `
  --output_test $PROCESSED_TEST

# Step 2: Feature engineering
Write-Host ""
Write-Host "Step 2/5: Engineering features..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
uv run python scripts/featurize.py `
  --input $PROCESSED_TRAIN `
  --output $FEATURIZED_TRAIN

uv run python scripts/featurize.py `
  --input $PROCESSED_TEST `
  --output $FEATURIZED_TEST

# Step 3: Train model
Write-Host ""
Write-Host "Step 3/5: Training model..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
uv run python scripts/train.py `
  --input $FEATURIZED_TRAIN `
  --output $MODEL_PATH

# Step 4: Evaluate model (if test labels available)
Write-Host ""
Write-Host "Step 4/5: Evaluating model..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
if (Test-Path $FEATURIZED_TEST) {
  $content = Get-Content $FEATURIZED_TEST -First 1
  if ($content -match "Survived") {
    uv run python scripts/evaluate.py `
      --model $MODEL_PATH `
      --input $FEATURIZED_TEST `
      --output $METRICS_PATH
  } else {
    Write-Host "Skipping evaluation (no labels in test data)" -ForegroundColor Gray
  }
} else {
  Write-Host "Test data not found, skipping evaluation" -ForegroundColor Gray
}

# Step 5: Make predictions
Write-Host ""
Write-Host "Step 5/5: Making predictions..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
uv run python scripts/predict.py `
  --model $MODEL_PATH `
  --input $FEATURIZED_TEST `
  --output $PREDICTIONS_PATH

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Pipeline completed successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Model saved to: $MODEL_PATH"
Write-Host "Predictions saved to: $PREDICTIONS_PATH"
if (Test-Path $METRICS_PATH) {
  Write-Host "Metrics saved to: $METRICS_PATH"
}
Write-Host "==========================================" -ForegroundColor Green

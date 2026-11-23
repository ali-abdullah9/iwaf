#!/bin/bash
# Run all baselines + ablations end-to-end.
# Stops at first failure so you don't waste GPU time on a broken script.

set -e

if [ ! -f train_split.csv ]; then
  echo "Splits not found. Running make_splits.py first..."
  python make_splits.py
fi

echo ""
echo "================================================================"
echo "1/7 LinearSVC + char TF-IDF (baseline: pre-BiLSTM)"
echo "================================================================"
python train_linearsvc.py

echo ""
echo "================================================================"
echo "2/7 CharCNN (baseline)"
echo "================================================================"
python train_neural.py --model cnn

echo ""
echo "================================================================"
echo "3/7 Plain LSTM (ablation: directionality + attention)"
echo "================================================================"
python train_neural.py --model lstm_plain

echo ""
echo "================================================================"
echo "4/7 BiLSTM mean-pool (ablation: does attention matter?)"
echo "================================================================"
python train_neural.py --model bilstm_meanpool

echo ""
echo "================================================================"
echo "5/7 BiLSTM-Attn no metadata (ablation: do meta features matter?)"
echo "================================================================"
python train_neural.py --model bilstm_nometa

echo ""
echo "================================================================"
echo "6/7 BiLSTM-Attn FULL (deployed architecture, on identical splits)"
echo "================================================================"
python train_neural.py --model bilstm_full

echo ""
echo "================================================================"
echo "7/7 DistilBERT (optional, GPU-heavy — comment out if no time)"
echo "================================================================"
python train_distilbert.py || echo "DistilBERT failed or skipped — continuing."

echo ""
echo "================================================================"
echo "ALL DONE"
echo "================================================================"
echo "Per-model results are in results/:"
ls -la results/

#!/bin/bash

# Bash script to setup VQAv2 dataset + COCO images and run GRPO training
# Automatically skips download if data already exists

set -e  # Exit on error

echo "========================================="
echo "VQAv2 GRPO Training Setup and Execution"
echo "========================================="
echo ""

# Step 1: Install dependencies
echo "Step 1: Installing Python dependencies..."
if ! pip show transformers > /dev/null 2>&1; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi
echo ""

# Step 2: Setup Grounding DINO (optional for training, but recommended)
echo "Step 2: Setting up Grounding DINO..."
if [ ! -d "GroundingDINO" ]; then
    python setup_grounding_dino.py
    cd GroundingDINO && pip install -e . && cd ..
    echo "✓ Grounding DINO setup complete"
else
    echo "✓ Grounding DINO already setup"
fi
echo ""

# Step 3: Download VQAv2 dataset and COCO images
echo "Step 3: Downloading VQAv2 dataset and COCO images..."
python download_vqav2.py
echo "✓ Dataset download complete"
echo ""

# Step 4: Run GRPO training
echo "Step 4: Starting GRPO training for sub-question generation..."
echo ""
echo "Training configuration:"
echo "  - Model: Qwen/Qwen2.5-3B-Instruct"
echo "  - Judge: Qwen/Qwen2.5-7B-Instruct"
echo "  - Checkpoints: Every 100 steps"
echo "  - Logs: training.log"
echo ""
echo "Training will begin shortly..."
echo "You can monitor progress with: tail -f training.log"
echo ""

# Run training
python train_subquestion_grpo.py

echo ""
echo "========================================="
echo "✓ Training complete!"
echo "========================================="
echo ""
echo "Output files:"
echo "  - Checkpoints: checkpoints/subquestion_grpo/"
echo "  - Final model: checkpoints/subquestion_grpo/final_model/"
echo "  - Logs: training.log"
echo "  - Metrics: logs/subquestion_grpo/"
echo ""

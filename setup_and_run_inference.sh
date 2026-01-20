#!/bin/bash

# Bash script to setup VQAv2 dataset + COCO images and run inference
# Automatically skips download if data already exists

set -e  # Exit on error

echo "========================================="
echo "VQAv2 Inference Setup and Execution"
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

# Step 2: Setup Grounding DINO
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

# Step 4: Run inference
echo "Step 4: Running VQA inference..."
echo ""
echo "Usage examples:"
echo "  1. Single image inference:"
echo "     python run_inference.py --image path/to/image.jpg --question \"What is in this image?\""
echo ""
echo "  2. With HTML report:"
echo "     python generate_report_example.py"
echo ""
echo "  3. Run demo:"
echo "     python demo.py"
echo ""

# Check if user provided image and question
if [ "$#" -eq 0 ]; then
    echo "Running demo to verify setup..."
    python demo.py
else
    # Run with user-provided arguments
    python run_inference.py "$@"
fi

echo ""
echo "========================================="
echo "✓ Setup and inference complete!"
echo "========================================="

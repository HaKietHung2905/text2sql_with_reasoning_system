#!/bin/bash
# Quick test script for Vertex AI setup

# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate text2sql

# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/gcp-keys/vertex-ai-key.json"
export USE_VERTEX_AI=true
export GOOGLE_CLOUD_PROJECT=useful-patrol-477810-s1

echo "🔧 Environment Variables:"
echo "  GOOGLE_APPLICATION_CREDENTIALS: $GOOGLE_APPLICATION_CREDENTIALS"
echo "  USE_VERTEX_AI: $USE_VERTEX_AI"
echo "  GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT"
echo ""

echo "📁 Checking credentials file..."
if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "  ✅ File exists: $GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "  ❌ File NOT found: $GOOGLE_APPLICATION_CREDENTIALS"
    exit 1
fi
echo ""

echo "🧪 Testing Vertex AI..."
python -c "
from src.models.google_genai import GoogleGenAI
model = GoogleGenAI()
print(f'✅ Backend: {model.backend}')
print(f'✅ Project: {model.project_id}')
print(f'✅ Model: {model.model_name}')
print()
print('Testing generation...')
response = model.generate('Say hello in one word')
print(f'Response: {response}')
print()
print('🎉 SUCCESS! Vertex AI is working!')
"

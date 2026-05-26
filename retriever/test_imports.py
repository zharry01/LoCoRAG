#!/usr/bin/env python3
"""Test script to check if all imports work correctly"""
import sys

print("Python version:", sys.version)
print("\n--- Testing imports ---\n")

try:
    import faiss
    print("✓ faiss imported successfully")
    print(f"  faiss version: {faiss.__version__}")
except Exception as e:
    print(f"✗ faiss import failed: {e}")
    sys.exit(1)

try:
    import torch
    print("✓ torch imported successfully")
    print(f"  torch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
except Exception as e:
    print(f"✗ torch import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy imported successfully")
    print(f"  numpy version: {np.__version__}")
except Exception as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

try:
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    print("✓ transformers imported successfully")
except Exception as e:
    print(f"✗ transformers import failed: {e}")
    sys.exit(1)

try:
    import datasets
    print("✓ datasets imported successfully")
    print(f"  datasets version: {datasets.__version__}")
except Exception as e:
    print(f"✗ datasets import failed: {e}")
    sys.exit(1)

try:
    import uvicorn
    from fastapi import FastAPI
    print("✓ fastapi and uvicorn imported successfully")
except Exception as e:
    print(f"✗ fastapi/uvicorn import failed: {e}")
    sys.exit(1)

print("\n--- All imports successful! ---\n")

#!/usr/bin/env python3
"""Test script to debug the retrieval server"""
import sys
import os

print("Testing retrieval server components...")

# Test 1: Check if data files exist
print("\n--- Test 1: Check data files ---")
file_path = os.getenv("RETRIEVER_DATA_DIR", "retriever/indexes/musique")
index_file = os.getenv("INDEX_FILE", f"{file_path}/e5-large_Flat.index")
corpus_file = os.getenv("CORPUS_FILE", "retriever/Corpus/musique_corpus.jsonl")

print(f"Index file: {index_file}")
print(f"  Exists: {os.path.exists(index_file)}")

print(f"\nCorpus file: {corpus_file}")
print(f"  Exists: {os.path.exists(corpus_file)}")

if not os.path.exists(index_file):
    print("\n✗ Index file not found!")
    sys.exit(1)

if not os.path.exists(corpus_file):
    print("\n✗ Corpus file not found!")
    sys.exit(1)

print("✓ All files exist")

# Test 2: Test faiss index loading
print("\n--- Test 2: Load faiss index ---")
try:
    import faiss
    index = faiss.read_index(index_file)
    print(f"✓ Faiss index loaded successfully")
    print(f"  Index size: {index.ntotal} vectors")
except Exception as e:
    print(f"✗ Failed to load faiss index: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test corpus loading with single process
print("\n--- Test 3: Load corpus (single process) ---")
try:
    import datasets
    corpus = datasets.load_dataset(
        'json',
        data_files=corpus_file,
        split="train",
        num_proc=1  # Changed from 4 to 1 to avoid multiprocessing issues
    )
    print(f"✓ Corpus loaded successfully")
    print(f"  Corpus size: {len(corpus)}")
except Exception as e:
    print(f"✗ Failed to load corpus: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test model loading
print("\n--- Test 4: Load model ---")
try:
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    import torch

    model_path = "intfloat/e5-large-v2"
    print(f"  Loading model from: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    print("  Loading config...")
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    print("  Loading model...")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model = model.to(device)

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n--- All tests passed! ---")

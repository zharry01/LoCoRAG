# LoCoRAG

This repository (LoCoRAG) is the implementation of Self-Correcting Agentic RAG via Memory-Grounded Failure Localization.

## 1. Environment Setup

Create the Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your machine needs GPU FAISS, install the FAISS package that matches your CUDA environment instead of `faiss-cpu`.

Copy the environment template:

```bash
cp .env.example .env
```

Configure the LLM provider in `.env`.

For OpenAI-compatible APIs:

```env
AI_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_LLM_MODEL=gpt-4.1-mini
```

For vLLM:

```env
AI_PROVIDER=vllm
VLLM_HOST=http://127.0.0.1
VLLM_LLM_PORT=30023
VLLM_LLM_MODEL=your_served_model_name
```

The RAG pipeline expects a retriever service at:

```env
SEARCH_SERVICE_HOST=127.0.0.1
SEARCH_SERVICE_PORT=8091
SEARCH_SERVICE_ENDPOINT=/retrieve
TOP_K=5
FACT_EXTRACTION_MODE=two_stage
```

## 2. Configure and Start Retriever

The included local retriever uses an E5 encoder plus a FAISS index. Small corpus files are provided in `retriever/Corpus/`.

Build an index:

```bash
CORPUS_FILE=retriever/Corpus/hotpotqa_corpus.jsonl \
INDEX_SAVE_DIR=retriever/indexes/hotpotqa \
RETRIEVER_MODEL=intfloat/e5-large-v2 \
bash retriever/build_index.sh
```

Start the retriever:

```bash
INDEX_FILE=retriever/indexes/hotpotqa/e5-large_Flat.index \
CORPUS_FILE=retriever/Corpus/hotpotqa_corpus.jsonl \
RETRIEVER_MODEL=intfloat/e5-large-v2 \
bash retriever/retrieval.sh
```

The retriever serves `POST /retrieve` on port `8091`.

Use the matching dataset and corpus/index pair for other tasks, for example:

```bash
CORPUS_FILE=retriever/Corpus/musique_corpus.jsonl \
INDEX_SAVE_DIR=retriever/indexes/musique \
bash retriever/build_index.sh
```

Then start it with:

```bash
INDEX_FILE=retriever/indexes/musique/e5-large_Flat.index \
CORPUS_FILE=retriever/Corpus/musique_corpus.jsonl \
bash retriever/retrieval.sh
```

## 3. Run One RAG Query

Make sure the LLM service and retriever are both running, then test one query:

```bash
python -c "from rag_pipeline_lib import llm_adapter; from rag_pipeline_lib.pipeline import run_multistep_pipeline; llm_adapter.configure_llm_provider(); print(run_multistep_pipeline('When did Lothair II mother die?', verbose=True))"
```

## 4. Batch Prediction

Input JSONL format:

```json
{"id": "case_id", "input": "question", "output": [{"answer": "gold answer"}]}
```

Run batch prediction:

```bash
python evaluation/generate_predictions_from_multistep.py \
  data/hotpotqa.jsonl \
  Result/hotpotqa_predictions.jsonl \
  --sample_size 50 \
  --sequential_sampling \
  --max_workers 4
```

Resume an interrupted run:

```bash
python evaluation/generate_predictions_from_multistep.py \
  data/hotpotqa.jsonl \
  Result/hotpotqa_predictions.jsonl \
  --resume \
  --max_workers 4
```

Prediction output format:

```json
{"id": "case_id", "question": "question", "output": [{"answer": "predicted answer", "reasoning": "optional reasoning"}]}
```

## 5. Evaluation

Compute exact match, F1, cover EM, and ROUGE-L:

```bash
python evaluation/evaluation_script.py \
  data/hotpotqa.jsonl \
  --guess_file Result/hotpotqa_predictions.jsonl
```

Write per-example metrics:

```bash
python evaluation/evaluation_with_metrics.py \
  data/hotpotqa.jsonl \
  --guess_file Result/hotpotqa_predictions.jsonl \
  --output_file Result/hotpotqa_predictions_with_metrics.jsonl
```

Run LLM-based answer judging:

```bash
python evaluation/llm_evaluate_predictions.py \
  --prediction_file Result/hotpotqa_predictions.jsonl \
  --dataset_file data/hotpotqa.jsonl \
  --output_file Result/hotpotqa_llm_eval.jsonl \
  --max_workers 8
```

## 6. Included Data

| File | Description |
| --- | --- |
| `data/hotpotqa.jsonl` | HotpotQA-style QA subset |
| `data/2wikimultihopqa.jsonl` | 2WikiMultihopQA-style QA subset |
| `data/musique.jsonl` | MuSiQue-style QA subset |
| `data/ragtracer.jsonl` | RAG tracing / QA subset |
| `retriever/Corpus/hotpotqa_corpus.jsonl` | HotpotQA retriever corpus subset |
| `retriever/Corpus/2wikimultihopqa_corpus.jsonl` | 2Wiki retriever corpus subset |
| `retriever/Corpus/musique_corpus.jsonl` | MuSiQue retriever corpus subset |

Runtime outputs are ignored by Git, including `Result/`, `data/evidence_memory/`, `data/log.txt`, and `retriever/indexes/`.

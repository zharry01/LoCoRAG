SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

file_path=${RETRIEVER_DATA_DIR:-"$PROJECT_ROOT/retriever/indexes/hotpotqa"}
index_file=${INDEX_FILE:-$file_path/e5-large_Flat.index}
corpus_file=${CORPUS_FILE:-"$PROJECT_ROOT/retriever/Corpus/hotpotqa_corpus.jsonl"}
retriever_name=${RETRIEVER_NAME:-e5-large}
retriever_path=${RETRIEVER_MODEL:-intfloat/e5-large-v2}

# Set environment variables to prevent memory issues
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python "$SCRIPT_DIR/retrieval_server.py" --index_path "$index_file" \
                          --corpus_path "$corpus_file" \
                          --topk 5 \
                          --retriever_name "$retriever_name" \
                          --retriever_model "$retriever_path"

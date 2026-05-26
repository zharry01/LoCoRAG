"""
Configuration settings for the RAG pipeline.
Loads sensitive keys from .env file.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

# --- AI Provider Selection (for LLM Generation) ---
# Supported: 'vllm', 'openai'
AI_PROVIDER = os.getenv('AI_PROVIDER', 'vllm').lower()

# --- API Keys (Loaded from .env file or environment variables) ---

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # For OpenAI
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.siliconflow.cn/v1/') # Base URL for OpenAI API


# --- VLLM Settings ---
VLLM_HOST = os.getenv('VLLM_HOST', "http://127.0.0.1")

# Default model and port for general RAG response generation.
# Use either a served model name or a local model path exposed by vLLM.
VLLM_LLM_MODEL = os.getenv('VLLM_LLM_MODEL', "Qwen3-Next-80B-A3B")
VLLM_LLM_PORT = int(os.getenv('VLLM_LLM_PORT', 30023))

# --- VLLM Dedicated Models Configuration ---
# Set to True to use a different model/port for each VLLM task.
# If False, all tasks will use the default VLLM_LLM_MODEL and VLLM_LLM_PORT.
VLLM_USE_DEDICATED_MODELS = _get_bool('VLLM_USE_DEDICATED_MODELS', True)

# Configuration for 'analyze_query' task
VLLM_ANALYZE_QUERY_MODEL = os.getenv('VLLM_ANALYZE_QUERY_MODEL', VLLM_LLM_MODEL)
VLLM_ANALYZE_QUERY_PORT = int(os.getenv('VLLM_ANALYZE_QUERY_PORT', VLLM_LLM_PORT))

# Configuration for 'extract_facts' task
VLLM_EXTRACT_FACTS_MODEL = os.getenv('VLLM_EXTRACT_FACTS_MODEL', VLLM_LLM_MODEL)
VLLM_EXTRACT_FACTS_PORT = int(os.getenv('VLLM_EXTRACT_FACTS_PORT', VLLM_LLM_PORT))

# Configuration for 'update_plan' task
VLLM_UPDATE_PLAN_MODEL = os.getenv('VLLM_UPDATE_PLAN_MODEL', VLLM_LLM_MODEL)
VLLM_UPDATE_PLAN_PORT = int(os.getenv('VLLM_UPDATE_PLAN_PORT', VLLM_LLM_PORT))

# Configuration for 'replan_conditions' task
VLLM_REPLAN_CONDITIONS_MODEL = os.getenv('VLLM_REPLAN_CONDITIONS_MODEL', VLLM_LLM_MODEL)
VLLM_REPLAN_CONDITIONS_PORT = int(os.getenv('VLLM_REPLAN_CONDITIONS_PORT', VLLM_LLM_PORT))

# Configuration for 'generate_final_answer' task    
VLLM_GENERATE_FINAL_ANSWER_MODEL = os.getenv('VLLM_GENERATE_FINAL_ANSWER_MODEL', VLLM_LLM_MODEL)
VLLM_GENERATE_FINAL_ANSWER_PORT = int(os.getenv('VLLM_GENERATE_FINAL_ANSWER_PORT', VLLM_LLM_PORT))


# Deprecated, use VLLM_HOST and specific ports instead
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL', f"{VLLM_HOST}:{VLLM_LLM_PORT}/v1")

OPENAI_LLM_MODEL = os.getenv('OPENAI_LLM_MODEL', "deepseek-chat")


# --- LLM for evaluation ---
#LLM_MODEL_EVAL = "deepseek-chat"
#LLM_MODEL_EVAL = OPENAI_LLM_MODEL
LLM_MODEL_EVAL = os.getenv('LLM_MODEL_EVAL', "deepseek-v3.2")

# --- Search Service Configuration ---
SEARCH_SERVICE_HOST = os.getenv('SEARCH_SERVICE_HOST', '127.0.0.1')
SEARCH_SERVICE_PORT = int(os.getenv('SEARCH_SERVICE_PORT', 8091))
SEARCH_SERVICE_ENDPOINT = os.getenv('SEARCH_SERVICE_ENDPOINT', '/retrieve')

# --- Proxy Configuration (Optional) ---
# Set PROXY_ENABLED to 'True' or 'true' in .env or environment to enable
PROXY_ENABLED = _get_bool('PROXY_ENABLED', False)
PROXY_URL = os.getenv('PROXY_URL', None) 


# --- RAG Parameters ---
TOP_K = int(os.getenv('TOP_K', 5)) # Number of documents to retrieve

# --- Control Flags ---
SHOW_RETRIEVED_CONTEXT = _get_bool('SHOW_RETRIEVED_CONTEXT', True)

# --- Fact Extraction Mode ---
# Set to 'two_stage' to use the new two-stage extraction (evidence + reasoned facts)
# Set to 'single_stage' to use the legacy single-stage extraction
FACT_EXTRACTION_MODE = os.getenv('FACT_EXTRACTION_MODE', 'two_stage').lower()

# --- Enable Thinking Mode ---
# Set to 'False' to disable thinking mode for Qwen models
ENABLE_THINKING = _get_bool('ENABLE_THINKING', False)

# --- Final Answer Mode ---
# True: final answer prompt returns {"reasoning": ..., "answer": ...}
# False: final answer prompt returns plain answer text
USE_FINAL_ANSWER_REASONING = _get_bool('USE_FINAL_ANSWER_REASONING', True)

# --- LLM Model Name (Set based on provider for generation) ---
LLM_MODEL_NAME = None # Initialize
if  AI_PROVIDER == 'vllm':
    LLM_MODEL_NAME = VLLM_LLM_MODEL
elif AI_PROVIDER == 'openai':
    LLM_MODEL_NAME = OPENAI_LLM_MODEL
else:
    raise ValueError(f"Unsupported AI_PROVIDER for LLM: {AI_PROVIDER}.")

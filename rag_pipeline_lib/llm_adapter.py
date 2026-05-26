import sys
import config
from rag_pipeline_lib.llm_providers import vllm_utils, openai_utils
from rag_pipeline_lib.pipeline import tracer_context
from functools import wraps

def traceable_llm_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracer = tracer_context.get()
        result = func(*args, **kwargs)
        
        if tracer:
            input_args = {key: value for key, value in kwargs.items()}
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            for i, arg in enumerate(args):
                input_args[arg_names[i]] = arg
            tracer.record_llm_call(
                adapter_function_name=func.__name__,
                inputs=input_args,
                output=result
            )
        return result
    return wrapper

def configure_llm_provider():
    print(f"Configuring LLM provider: {config.AI_PROVIDER}")
    if config.AI_PROVIDER == 'vllm':
        try:
            vllm_utils.configure_vllm_client()
        except Exception as e:
            print(f"Error: Failed to configure vLLM: {e}", file=sys.stderr)
            raise
    elif config.AI_PROVIDER == 'openai':
        try:
            openai_utils.configure_openai_client()
        except Exception as e:
            print(f"Error: Failed to configure OpenAI: {e}", file=sys.stderr)
            raise
    else:
        error_msg = f"Unsupported AI_PROVIDER: {config.AI_PROVIDER}. Please choose 'gemini', 'ollama', 'grok', 'deepseek', 'vllm', or 'openai'."
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg)

@traceable_llm_call
def generate_rag_response(query: str, context: str) -> str:
    """
    Generates a RAG response using the configured LLM provider.
    The default model from config.py for the provider will be used.
    """
    if config.AI_PROVIDER == 'vllm':
        return vllm_utils.vllm_generate_rag_response(query, context)
    elif config.AI_PROVIDER == 'openai':
        return openai_utils.openai_generate_rag_response(query, context)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {config.AI_PROVIDER}. Cannot generate RAG response.")

@traceable_llm_call
def analyze_query(query: str) -> str:
    """
    Analyzes the input query using the configured graph LLM provider.
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling analyze_query (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_analyze_query(query)
    elif provider == 'openai':
        return openai_utils.openai_analyze_query(query)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot analyze query.")

@traceable_llm_call
def extract_facts(query: str, active_requirement: str, known_facts: str, retrieved_documents: str) -> str:
    """
    Extracts facts using the configured LLM provider (single-stage, legacy method).
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling extract_facts (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_extract_facts(query, active_requirement, known_facts, retrieved_documents)
    elif provider == 'openai':
        return openai_utils.openai_extract_facts(query, active_requirement, known_facts, retrieved_documents)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot extract facts.")

@traceable_llm_call
def extract_potential_evidence(current_sub_query: str, original_query: str, retrieved_documents: str) -> str:
    """
    Stage 1: Extracts potential evidence using the configured LLM provider (maximal recall).
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling extract_potential_evidence (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_extract_potential_evidence(current_sub_query, original_query, retrieved_documents)
    elif provider == 'openai':
        return openai_utils.openai_extract_potential_evidence(current_sub_query, original_query, retrieved_documents)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot extract potential evidence.")

@traceable_llm_call
def generate_reasoned_facts(original_query: str, current_sub_query: str, extracted_evidence: str, requirement_id: str) -> str:
    """
    Stage 2: Generates reasoned facts from potential evidence using the configured LLM provider.
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling generate_reasoned_facts (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_generate_reasoned_facts(original_query, current_sub_query, extracted_evidence, requirement_id)
    elif provider == 'openai':
        return openai_utils.openai_generate_reasoned_facts(original_query, current_sub_query, extracted_evidence, requirement_id)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot generate reasoned facts.")

@traceable_llm_call
def update_plan(query: str, collected_facts: str, pending_requirements: str) -> str:
    """
    Updates the plan using the configured LLM provider.
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling update_plan (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_update_plan(query, collected_facts, pending_requirements)
    elif provider == 'openai':
        return openai_utils.openai_update_plan(query, collected_facts, pending_requirements)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot update plan.")

@traceable_llm_call
def replan_conditions(query: str, collected_facts: str, pending_requirements: str) -> str:
    """
    Replans conditions using the configured LLM provider.
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling replan_conditions (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_replan_conditions(query, collected_facts, pending_requirements)
    elif provider == 'openai':
        return openai_utils.openai_replan_conditions(query, collected_facts, pending_requirements)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot replan conditions.")

@traceable_llm_call
def generate_final_answer(query: str, facts: str) -> str:
    """
    Generates the final answer using the configured LLM provider.
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling generate_final_answer (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_generate_final_answer(query, facts)
    elif provider == 'openai':
        return openai_utils.openai_generate_final_answer(query, facts)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot generate final answer.")

@traceable_llm_call
def generate_next_actions(query: str, collected_facts: str, pending_requirements: str) -> str:
    """
    Generates next actions when the planner fails to provide them.
    This is a fallback mechanism to ensure the pipeline can continue.
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling generate_next_actions (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_generate_next_actions(query, collected_facts, pending_requirements)
    elif provider == 'openai':
        return openai_utils.openai_generate_next_actions(query, collected_facts, pending_requirements)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot generate next actions.")

@traceable_llm_call
def evaluate_answer(question: str, golden_answer: str, predicted_answer: str) -> str:
    """
    Evaluates if the predicted answer is correct using the configured LLM provider.
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling evaluate_answer (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_evaluate_answer(question, golden_answer, predicted_answer)
    elif provider == 'openai':
        return openai_utils.openai_evaluate_answer(question, golden_answer, predicted_answer)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot evaluate answer.")

@traceable_llm_call
def enough_to_update_plan(query: str, collected_facts: str, pending_requirements: str) -> str:
    """
    Updates the plan when evidence is sufficient (ENOUGH_TO_UPDATE scenario).
    This is a lightweight alternative to full replanning.

    Args:
        query: The original user query
        collected_facts: JSON string of collected facts (including PARTIAL_EVIDENCE)
        pending_requirements: JSON string of pending requirements

    Returns:
        LLM response with updated plan and next actions
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling enough_to_update_plan (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_enough_to_update_plan(query, collected_facts, pending_requirements)
    elif provider == 'openai':
        return openai_utils.openai_enough_to_update_plan(query, collected_facts, pending_requirements)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot update plan with sufficient evidence.")

@traceable_llm_call
def failure_attribution(original_query: str, completed_query: str, failed_requirement_query: str) -> str:
    """
    Step 1: Analyzes the failure and determines the root cause using Failure Attribution Prompt.

    Args:
        original_query: The original user query
        completed_query: JSON string of completed requirements with answers/evidence
        failed_requirement_query: JSON string of the failed requirement

    Returns:
        LLM response with failure classification (ENOUGH_TO_UPDATE, UPSTREAM_ISSUE, etc.)
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling failure_attribution (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_failure_attribution(original_query, completed_query, failed_requirement_query)
    elif provider == 'openai':
        return openai_utils.openai_failure_attribution(original_query, completed_query, failed_requirement_query)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot perform failure attribution.")

@traceable_llm_call
def replan_with_analysis(original_query: str, original_plan: str, failed_requirement: str, problem_analysis: str) -> str:
    """
    Step 2: Generates a new recovery plan based on the failure analysis using Replan Prompt.

    Args:
        original_query: The original user query
        original_plan: JSON string of the original plan
        failed_requirement: JSON string of the failed requirement
        problem_analysis: JSON string of the failure analysis from Step 1

    Returns:
        LLM response with updated plan and next actions
    """
    provider = config.AI_PROVIDER
    print(f"LLM Adapter: Calling replan_with_analysis (AI Provider: {provider})")
    if provider == 'vllm':
        return vllm_utils.vllm_replan_with_analysis(original_query, original_plan, failed_requirement, problem_analysis)
    elif provider == 'openai':
        return openai_utils.openai_replan_with_analysis(original_query, original_plan, failed_requirement, problem_analysis)
    else:
        raise ValueError(f"Unsupported AI_PROVIDER: {provider}. Cannot replan with analysis.")

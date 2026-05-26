import config
from openai import OpenAI
import httpx
from .. import prompts

# Global variable to cache the OpenAI client and avoid "Too many open files" error
_openai_client_cache = None

def configure_openai_client():
    """
    Configures and returns the OpenAI API client.
    Checks for OPENAI_API_KEY in config.py and applies proxy if configured.
    Uses a global cache to reuse the client across calls.
    """
    global _openai_client_cache
    
    if _openai_client_cache is not None:
        return _openai_client_cache

    if not hasattr(config, 'OPENAI_API_KEY') or not config.OPENAI_API_KEY:
        raise ValueError("Error: OPENAI_API_KEY is not set in config.py.")

    client_params = {
        "api_key": config.OPENAI_API_KEY,
        "base_url": config.OPENAI_BASE_URL
    }

    if config.PROXY_ENABLED and config.PROXY_URL:
        proxy = config.PROXY_URL
        _http_client = httpx.Client(proxy=proxy)
        client_params["http_client"] = _http_client
        print(f"OpenAI API client will use proxy: {config.PROXY_URL}")
    # Removed noisy print when no proxy is used

    try:
        _openai_client_cache = OpenAI(**client_params)
        print("OpenAI API client configured and cached.")
        return _openai_client_cache
    except Exception as e:
        print(f"Error configuring OpenAI API client: {e}")
        raise

def openai_generate_rag_response(query: str, context: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Generates a response using the OpenAI LLM based on the query and context."""
    user_prompt = prompts.USER_PROMPT_TEMPLATE_RAG.format(context=context, query=query)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to generate RAG response...")

        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_RAG},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                 "enable_thinking": False  #"chat_template_kwargs":"enable_thinking": False}
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty RAG response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to generate RAG response: {e}")
        return f"Sorry, encountered an error when calling LLM to generate RAG response: {e}"

def openai_analyze_query(query: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Analyzes the input query using openai."""
    combined_instruction = prompts.USER_PROMPT_QUERY_ANALYSIS.format(query=query)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to analyze query...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_QUERY_ANALYSIS},
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0.0,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty query analysis response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI ({llm_model_name}) to analyze query: {e}")
        return f"Sorry, encountered an error when calling LLM to analyze query: {e}"


def openai_extract_potential_evidence(
    current_sub_query: str,
    original_query: str,
    retrieved_documents: str,
    llm_model_name: str = config.LLM_MODEL_NAME
) -> str:
    """Stage 1: Extracts potential evidence from retrieved documents using OpenAI LLM."""
    user_prompt = prompts.USER_PROMPT_EXTRACT_POTENTIAL_EVIDENCE.format(
        current_sub_query=current_sub_query,
        original_query=original_query,
        retrieved_documents=retrieved_documents
    )

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to extract potential evidence...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_EXTRACT_POTENTIAL_EVIDENCE},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty potential evidence response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to extract potential evidence: {e}")
        return f"Sorry, encountered an error when calling LLM to extract potential evidence: {e}"

def openai_generate_reasoned_facts(original_query: str, current_sub_query: str, extracted_evidence: str, requirement_id: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Stage 2: Generates reasoned facts from potential evidence using OpenAI LLM."""
    user_prompt = prompts.USER_PROMPT_GENERATE_REASONED_FACTS.format(
        original_query=original_query,
        current_sub_query=current_sub_query,
        extracted_evidence=extracted_evidence,
        requirement_id=requirement_id
    )

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to generate reasoned facts...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_GENERATE_REASONED_FACTS},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty reasoned facts response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to generate reasoned facts: {e}")
        return f"Sorry, encountered an error when calling LLM to generate reasoned facts: {e}"

def openai_extract_facts(query: str, active_requirement: str, retrieved_documents: str,  known_facts: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Extracts facts from the context that satisfy the given condition based on the query using OpenAI LLM."""
    user_prompt = prompts.USER_PROMPT_FACT_EXTRACTION.format(query=query, active_requirement=active_requirement, known_facts=known_facts, retrieved_documents=retrieved_documents)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to extract facts from context...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_FACT_EXTRACTION},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty fact extraction response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to extract facts from context: {e}")
        return f"Sorry, encountered an error when calling LLM to extract facts: {e}"

def openai_update_plan(query: str, collected_facts: str, pending_requirements: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Updates the plan based on collected facts and pending requirements using OpenAI LLM."""
    user_prompt = prompts.USER_PROMPT_PLAN_UPDATER.format(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to update plan...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_PLAN_UPDATER},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty plan update response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to update plan: {e}")
        return f"Sorry, encountered an error when calling LLM to update plan: {e}"

def openai_replan_conditions(query: str, collected_facts: str, pending_requirements: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Replans the required conditions based on the user query, initial condition, and extracted facts using OpenAI LLM."""
    user_prompt = prompts.USER_PROMPT_CONDITION_REPLAN.format(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to replan conditions...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_CONDITION_REPLAN},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty condition replan response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to replan conditions: {e}")
        return f"Sorry, encountered an error when calling LLM to replan conditions: {e}"

def openai_generate_final_answer(query: str, facts: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Generates the final answer based on the user query and extracted facts using OpenAI LLM."""
    system_prompt = prompts.get_system_prompt_final_answer()
    user_prompt = prompts.USER_PROMPT_FINAL_ANSWER.format(query=query, facts=facts)

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to generate final answer...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty final answer response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to generate final answer: {e}")
        return f"Sorry, encountered an error when calling LLM to generate final answer: {e}"

def openai_evaluate_answer(question: str, golden_answer: str, predicted_answer: str, llm_model_name: str = config.LLM_MODEL_EVAL) -> str:
    """
    Uses OpenAI LLM to evaluate if the predicted answer is correct.
    Returns "True", "False", or an error string.
    """
    user_prompt = prompts.USER_PROMPT_EVALUATION.format(
        question=question,
        golden_answer=golden_answer,
        predicted_answer=predicted_answer
    )

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to evaluate answer...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_EVALUATION},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=5,
            temperature=0.0,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Evaluation Error: Empty response from OpenAI."

        cleaned_response = response_text.strip().lower()
        if "true" in cleaned_response:
            return "True"
        elif "false" in cleaned_response:
            return "False"
        else:
            return f"Evaluation Error: Unexpected response '{response_text.strip()}'"
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to evaluate answer: {e}")
        return f"Evaluation Error: {e}"

def openai_generate_next_actions(query: str, collected_facts: str, pending_requirements: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """
    Generates next actions using OpenAI LLM when the planner fails to provide them.
    This is a fallback mechanism to ensure the pipeline can continue.
    """
    user_prompt = prompts.USER_PROMPT_GENERATE_NEXT_ACTIONS.format(
        query=query,
        collected_facts=collected_facts,
        pending_requirements=pending_requirements
    )

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) to generate next actions (fallback)...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_GENERATE_NEXT_ACTIONS},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Error: Empty next actions generation response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) to generate next actions: {e}")
        return f"Error calling LLM to generate next actions: {e}"


def openai_failure_attribution(original_query: str, completed_query: str, failed_requirement_query: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """
    Step 1 of new replan: Analyze failure and determine root cause using OpenAI.

    Args:
        original_query: The original user query
        completed_query: JSON string of completed requirements with answers/evidence
        failed_requirement_query: JSON string of the failed requirement
        llm_model_name: Model name to use

    Returns:
        LLM response with failure classification
    """
    user_prompt = prompts.USER_PROMPT_FAILURE_ATTRIBUTION.format(
        original_query=original_query,
        completed_query=completed_query,
        failed_requirement_query=failed_requirement_query
    )

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) for failure attribution...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_FAILURE_ATTRIBUTION},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Error: Empty failure attribution response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) for failure attribution: {e}")
        return f"Error calling LLM for failure attribution: {e}"


def openai_replan_with_analysis(original_query: str, original_plan: str, failed_requirement: str, problem_analysis: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """
    Step 2 of new replan: Generate recovery plan based on failure analysis using OpenAI.

    Args:
        original_query: The original user query
        original_plan: JSON string of the original plan
        failed_requirement: JSON string of the failed requirement
        problem_analysis: JSON string of the failure analysis from Step 1
        llm_model_name: Model name to use

    Returns:
        LLM response with updated plan and next actions
    """
    user_prompt = prompts.USER_PROMPT_REPLAN_WITH_ANALYSIS.format(
        original_query=original_query,
        original_plan=original_plan,
        failed_requirement=failed_requirement,
        problem_analysis=problem_analysis
    )

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) for replan with analysis...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_REPLAN_WITH_ANALYSIS},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            extra_body={
                 "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Error: Empty replan response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) for replan with analysis: {e}")
        return f"Error calling LLM for replan with analysis: {e}"


def openai_enough_to_update_plan(
    query: str,
    collected_facts: str,
    pending_requirements: str,
    llm_model_name: str = config.LLM_MODEL_NAME
) -> str:
    """
    OpenAI implementation for ENOUGH_TO_UPDATE scenario.
    Updates plan using available evidence without full replanning.
    """
    from rag_pipeline_lib.prompts import (
        SYSTEM_PROMPT_ENOUGH_TO_UPDATE,
        USER_PROMPT_ENOUGH_TO_UPDATE
    )

    system_prompt = SYSTEM_PROMPT_ENOUGH_TO_UPDATE
    user_prompt = USER_PROMPT_ENOUGH_TO_UPDATE.format(
        original_query=query,
        collected_facts=collected_facts,
        pending_requirements=pending_requirements
    )

    try:
        client = configure_openai_client()
        print(f"Calling OpenAI LLM ({llm_model_name}) for enough_to_update_plan...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                "enable_thinking": False
            }
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Error: Empty enough_to_update response received from OpenAI API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling OpenAI LLM ({llm_model_name}) for enough_to_update_plan: {e}")
        return f"Error calling LLM for enough_to_update_plan: {e}"

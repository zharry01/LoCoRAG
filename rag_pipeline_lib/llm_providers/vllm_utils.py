import config
from openai import OpenAI
import httpx  # Required for proxy support
from .. import prompts

# Global cache for VLLM clients to avoid "Too many open files" error
# Key: base_url, Value: OpenAI client instance
_vllm_client_cache = {}

def configure_vllm_client(task_type: str = "rag_response"):
    """
    Configures and returns the OpenAI-compatible client for vLLM.
    Selects the base URL and model based on the task type and config settings.
    Uses a global cache to reuse clients for the same base URL.

    Args:
        task_type (str): The type of task being performed.
                         Examples: 'rag_response', 'analyze_query', 'extract_facts',
                                   'extract_evidence', 'generate_facts', etc.
    """
    global _vllm_client_cache

    model_name = config.VLLM_LLM_MODEL
    port = config.VLLM_LLM_PORT

    if config.VLLM_USE_DEDICATED_MODELS and task_type not in ["rag_response", "evaluate_answer"]:
        model_map = {
            "analyze_query": (config.VLLM_ANALYZE_QUERY_MODEL, config.VLLM_ANALYZE_QUERY_PORT),
            "extract_facts": (config.VLLM_EXTRACT_FACTS_MODEL, config.VLLM_EXTRACT_FACTS_PORT),
            "extract_evidence": (config.VLLM_EXTRACT_FACTS_MODEL, config.VLLM_EXTRACT_FACTS_PORT),  # Use same model as extract_facts
            "generate_facts": (config.VLLM_EXTRACT_FACTS_MODEL, config.VLLM_EXTRACT_FACTS_PORT),  # Use same model as extract_facts
            "update_plan": (config.VLLM_UPDATE_PLAN_MODEL, config.VLLM_UPDATE_PLAN_PORT),
            "replan_conditions": (config.VLLM_REPLAN_CONDITIONS_MODEL, config.VLLM_REPLAN_CONDITIONS_PORT),
            "failure_attribution": (config.VLLM_REPLAN_CONDITIONS_MODEL, config.VLLM_REPLAN_CONDITIONS_PORT),  # Use same model as replan
            "replan_with_analysis": (config.VLLM_REPLAN_CONDITIONS_MODEL, config.VLLM_REPLAN_CONDITIONS_PORT),  # Use same model as replan
            "generate_final_answer": (config.VLLM_GENERATE_FINAL_ANSWER_MODEL, config.VLLM_GENERATE_FINAL_ANSWER_PORT),
        }
        if task_type in model_map:
            model_name, port = model_map[task_type]
            # print(f"Using dedicated VLLM model for '{task_type}': {model_name} on port {port}")
        else:
            print(f"Warning: No dedicated model found for task '{task_type}'. Falling back to default.")

    base_url_to_use = f"{config.VLLM_HOST}:{port}/v1"
    
    if base_url_to_use in _vllm_client_cache:
        return _vllm_client_cache[base_url_to_use], model_name

    print(f"Configuring new vLLM client for '{task_type}': URL='{base_url_to_use}'")

    client_params = {
        "api_key": "empty",
        "base_url": base_url_to_use,
    }

    try:
        client = OpenAI(**client_params)
        _vllm_client_cache[base_url_to_use] = client
        print(f"VLLM API client for {base_url_to_use} configured successfully and cached.")
        return client, model_name
    except Exception as e:
        print(f"Error configuring VLLM API client: {e}")
        raise

def vllm_generate_rag_response(query: str, context: str, llm_model_name: str = config.LLM_MODEL_NAME) -> str:
    """Generates a response using VLLM based on the query and context."""
    # Combine system prompt and user prompt into a single instruction
    combined_instruction = f"{prompts.SYSTEM_PROMPT_RAG}\n{prompts.USER_PROMPT_TEMPLATE_RAG.format(context=context, query=query)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="rag_response")
        print(f"Calling VLLM ({model_to_use}) to generate RAG response...")
        
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty RAG response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to generate RAG response: {e}")
        return f"Sorry, encountered an error when calling LLM to generate RAG response: {e}"

def vllm_analyze_query(query: str, llm_model_name: str = None) -> str:
    """Analyzes the input query using VLLM."""
    combined_instruction = f"{prompts.SYSTEM_PROMPT_QUERY_ANALYSIS}\n{prompts.USER_PROMPT_QUERY_ANALYSIS.format(query=query)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="analyze_query")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to analyze query...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty query analysis response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to analyze query: {e}")
        return f"Sorry, encountered an error when calling LLM to analyze query: {e}"

def vllm_extract_potential_evidence(
    current_sub_query: str,
    original_query: str,
    retrieved_documents: str,
    llm_model_name: str = None
) -> str:
    """Stage 1: Extracts potential evidence from retrieved documents using VLLM."""
    combined_instruction = (
        f"{prompts.SYSTEM_PROMPT_EXTRACT_POTENTIAL_EVIDENCE}\n"
        f"{prompts.USER_PROMPT_EXTRACT_POTENTIAL_EVIDENCE.format(current_sub_query=current_sub_query, original_query=original_query, retrieved_documents=retrieved_documents)}"
    )

    try:
        client, model_to_use = configure_vllm_client(task_type="extract_evidence")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to extract potential evidence...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty potential evidence response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to extract potential evidence: {e}")
        return f"Sorry, encountered an error when calling LLM to extract potential evidence: {e}"

def vllm_generate_reasoned_facts(original_query: str, current_sub_query: str, extracted_evidence: str, requirement_id: str, llm_model_name: str = None) -> str:
    """Stage 2: Generates reasoned facts from potential evidence using VLLM."""
    combined_instruction = f"{prompts.SYSTEM_PROMPT_GENERATE_REASONED_FACTS}\n{prompts.USER_PROMPT_GENERATE_REASONED_FACTS.format(original_query=original_query, current_sub_query=current_sub_query, extracted_evidence=extracted_evidence, requirement_id=requirement_id)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="generate_facts")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to generate reasoned facts...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty reasoned facts response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to generate reasoned facts: {e}")
        return f"Sorry, encountered an error when calling LLM to generate reasoned facts: {e}"

def vllm_extract_facts(query: str, active_requirement: str, retrieved_documents: str, known_facts: str, llm_model_name: str = None) -> str:
    """Extracts facts from the context that satisfy the given condition based on the query using VLLM."""
    combined_instruction = f"{prompts.SYSTEM_PROMPT_FACT_EXTRACTION}\n{prompts.USER_PROMPT_FACT_EXTRACTION.format(query=query, active_requirement=active_requirement, known_facts=known_facts, retrieved_documents=retrieved_documents)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="extract_facts")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to extract facts from context...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty fact extraction response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to extract facts from context: {e}")
        return f"Sorry, encountered an error when calling LLM to extract facts: {e}"

def vllm_update_plan(query: str, collected_facts: str, pending_requirements: str, llm_model_name: str = None) -> str:
    """Updates the plan based on collected facts and pending requirements using VLLM."""
    combined_instruction = f"{prompts.SYSTEM_PROMPT_PLAN_UPDATER}\n{prompts.USER_PROMPT_PLAN_UPDATER.format(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="update_plan")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to update plan...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty plan update response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to update plan: {e}")
        return f"Sorry, encountered an error when calling LLM to update plan: {e}"

def vllm_replan_conditions(query: str, collected_facts: str, pending_requirements: str, llm_model_name: str = None) -> str:
    """Replans the required conditions based on the user query, initial condition, and extracted facts using VLLM."""
    combined_instruction = f"{prompts.SYSTEM_PROMPT_CONDITION_REPLAN}\n{prompts.USER_PROMPT_CONDITION_REPLAN.format(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="replan_conditions")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to replan conditions...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty condition replan response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to replan conditions: {e}")
        return f"Sorry, encountered an error when calling LLM to replan conditions: {e}"

def vllm_generate_final_answer(query: str, facts: str, llm_model_name: str = None) -> str:
    """Generates the final answer based on the user query and extracted facts using VLLM."""
    combined_instruction = f"{prompts.get_system_prompt_final_answer()}\n{prompts.USER_PROMPT_FINAL_ANSWER.format(query=query, facts=facts)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="generate_final_answer")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to generate final answer...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Empty final answer response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to generate final answer: {e}")
        return f"Sorry, encountered an error when calling LLM to generate final answer: {e}"

def vllm_evaluate_answer(question: str, golden_answer: str, predicted_answer: str, llm_model_name: str = None) -> str:
    """
    Uses VLLM to evaluate if the predicted answer is correct.
    Returns "True", "False", or an error string.
    """
    # VLLM often works best with a single combined prompt
    instruction = f"{prompts.SYSTEM_PROMPT_EVALUATION}\n\n{prompts.USER_PROMPT_EVALUATION.format(question=question, golden_answer=golden_answer, predicted_answer=predicted_answer)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="evaluate_answer")
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to evaluate answer...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": instruction}
            ],
            max_tokens=5,
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Evaluation Error: Empty response from VLLM."

        cleaned_response = response_text.strip().lower()
        if "true" in cleaned_response:
            return "True"
        elif "false" in cleaned_response:
            return "False"
        else:
            return f"Evaluation Error: Unexpected response '{response_text.strip()}'"
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to evaluate answer: {e}")
        return f"Evaluation Error: {e}"

def vllm_generate_next_actions(query: str, collected_facts: str, pending_requirements: str, llm_model_name: str = None) -> str:
    """
    Generates next actions using VLLM when the planner fails to provide them.
    This is a fallback mechanism to ensure the pipeline can continue.
    """
    combined_instruction = f"{prompts.SYSTEM_PROMPT_GENERATE_NEXT_ACTIONS}\n{prompts.USER_PROMPT_GENERATE_NEXT_ACTIONS.format(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)}"

    try:
        client, model_to_use = configure_vllm_client(task_type="update_plan")  # Reuse update_plan model
        if llm_model_name is None: llm_model_name = model_to_use
        print(f"Calling VLLM ({llm_model_name}) to generate next actions (fallback)...")
        completion = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "user", "content": combined_instruction}
            ],
            temperature=0
        )
        response_text = completion.choices[0].message.content
        if not response_text:
            return "Error: Empty next actions generation response received from VLLM API."
        return response_text.strip()
    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) to generate next actions: {e}")
        return f"Error calling LLM to generate next actions: {e}"


def vllm_failure_attribution(original_query: str, completed_query: str, failed_requirement_query: str, llm_model_name: str = None) -> str:
    """
    Step 1 of new replan: Analyze failure and determine root cause.

    Args:
        original_query: The original user query
        completed_query: JSON string of completed requirements with answers/evidence
        failed_requirement_query: JSON string of the failed requirement
        llm_model_name: Optional model name override

    Returns:
        LLM response with failure classification
    """
    try:
        if llm_model_name is None:
            client, llm_model_name = configure_vllm_client(task_type="failure_attribution")
        else:
            client, _ = configure_vllm_client(task_type="failure_attribution")

        user_prompt = prompts.USER_PROMPT_FAILURE_ATTRIBUTION.format(
            original_query=original_query,
            completed_query=completed_query,
            failed_requirement_query=failed_requirement_query
        )

        response = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_FAILURE_ATTRIBUTION},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        result = response.choices[0].message.content
        print(f"✅ VLLM ({llm_model_name}) failure attribution completed")
        return result

    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) for failure attribution: {e}")
        return f"Error calling LLM for failure attribution: {e}"


def vllm_replan_with_analysis(original_query: str, original_plan: str, failed_requirement: str, problem_analysis: str, llm_model_name: str = None) -> str:
    """
    Step 2 of new replan: Generate recovery plan based on failure analysis.

    Args:
        original_query: The original user query
        original_plan: JSON string of the original plan
        failed_requirement: JSON string of the failed requirement
        problem_analysis: JSON string of the failure analysis from Step 1
        llm_model_name: Optional model name override

    Returns:
        LLM response with updated plan and next actions
    """
    try:
        if llm_model_name is None:
            client, llm_model_name = configure_vllm_client(task_type="replan_with_analysis")
        else:
            client, _ = configure_vllm_client(task_type="replan_with_analysis")

        user_prompt = prompts.USER_PROMPT_REPLAN_WITH_ANALYSIS.format(
            original_query=original_query,
            original_plan=original_plan,
            failed_requirement=failed_requirement,
            problem_analysis=problem_analysis
        )

        response = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_REPLAN_WITH_ANALYSIS},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        result = response.choices[0].message.content
        print(f"✅ VLLM ({llm_model_name}) replan with analysis completed")
        return result

    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) for replan with analysis: {e}")
        return f"Error calling LLM for replan with analysis: {e}"


def vllm_enough_to_update_plan(
    query: str,
    collected_facts: str,
    pending_requirements: str,
    llm_model_name: str = None
) -> str:
    """
    vLLM implementation for ENOUGH_TO_UPDATE scenario.
    Updates plan using available evidence without full replanning.
    """
    from rag_pipeline_lib import prompts

    try:
        if llm_model_name is None:
            client, llm_model_name = configure_vllm_client(task_type="replan_conditions")
        else:
            client, _ = configure_vllm_client(task_type="replan_conditions")

        user_prompt = prompts.USER_PROMPT_ENOUGH_TO_UPDATE.format(
            original_query=query,
            collected_facts=collected_facts,
            pending_requirements=pending_requirements
        )

        response = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": prompts.SYSTEM_PROMPT_ENOUGH_TO_UPDATE},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        result = response.choices[0].message.content
        print(f"✅ VLLM ({llm_model_name}) enough_to_update_plan completed")
        return result

    except Exception as e:
        print(f"Error calling VLLM ({llm_model_name}) for enough_to_update_plan: {e}")
        return f"Error calling LLM for enough_to_update_plan: {e}"


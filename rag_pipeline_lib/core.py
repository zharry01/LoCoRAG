import config
from search import search_utils
from rag_pipeline_lib import llm_adapter
from rag_pipeline_lib import generator
from rag_pipeline_lib import thinker  # Thinker owns Stage-1 evidence extraction.
from rag_pipeline_lib.pipeline_logger import get_pipeline_logger
import re
import json

def retrieve_context(query: str) -> list[dict]:
    """Retrieves relevant context from the external search service."""
    print(f"\n--- Stage: Retrieving context for query: '{query}' via HTTP search ---")
    try:
        search_hits = search_utils.search_by_http(
            query=query,
            k=config.TOP_K,
            host=config.SEARCH_SERVICE_HOST,
            port=config.SEARCH_SERVICE_PORT
        )
        return search_hits
    except Exception as e:
        print(f"Error during context retrieval via HTTP: {e}")
        return []

def perform_rag(retrieval_query: str, generation_query: str = None) -> str:
    actual_generation_query = generation_query if generation_query is not None else retrieval_query

    print(f"\n--- Performing Single-Stage RAG (with HTTP Search) ---")
    print(f"--- Retrieval Query: '{retrieval_query}' ---")
    if generation_query is not None and retrieval_query != generation_query:
        print(f"--- Generation Query: '{generation_query}' ---")

    # 1. Retrieve context using the new HTTP-based retrieval
    search_hits = retrieve_context(retrieval_query)

    # 2. Prepare context string
    context = "No relevant context found."
    if search_hits:
        retrieved_texts_with_tags = []
        for i, hit in enumerate(search_hits):
            context_content = ""
            title = ""
            if isinstance(hit, dict):
                if 'contents' in hit:
                    context_content = str(hit['contents'])
                else:
                    print(f"Warning: Search hit missing 'contents', using full hit: {str(hit)[:100]}...")
                    context_content = str(hit)
                
                if 'title' in hit:
                    title = str(hit['title'])
            else:
                context_content = str(hit)
            
            if title:
                retrieved_texts_with_tags.append(f'<document id={i+1} title="{title}">\n{context_content}\n</document>')
            else:
                retrieved_texts_with_tags.append(f'<document id={i+1}>\n{context_content}\n</document>')
        
        if retrieved_texts_with_tags:
            context = "\n\n".join(retrieved_texts_with_tags)

        print(f"--- Retrieved Context (for retrieval query: '{retrieval_query}') ---")
        if config.SHOW_RETRIEVED_CONTEXT:
            print(context)
        else:
            print("[Context display disabled]")
        print("---------------------------------------")
    else:
        print(f"No relevant context found for retrieval query: '{retrieval_query}'.")

    # 3. Generate response
    try:
        answer = llm_adapter.generate_rag_response(
            query=actual_generation_query,
            context=context
        )
    except Exception as e:
        print(f"Error generating response for generation query '{actual_generation_query}': {e}")
        answer = f"Error generating response for generation query '{actual_generation_query}'."

    print(f"\nGeneration Query: {actual_generation_query}")
    print(f"Generated Answer: {answer}")
    return answer

def analyze_and_decompose_query(query: str) -> dict:
    """
    Analyzes the user query to understand the goal and decomposes it into sub-questions.
    It calls the LLM adapter, which returns a JSON string, and then parses it.

    Args:
        query: The user's input query.

    Returns:
        A dictionary containing the user's goal and a list of sub-questions (requirements),
        or an empty dictionary if the analysis or parsing fails.
    """
    print(f"\n--- Stage: Analyzing and decomposing query: '{query}' ---")
    try:
        # 1. Call the LLM adapter to get the analysis in JSON format
        analysis_json_str = llm_adapter.analyze_query(query)

        match = re.search(r'\{.*\}', analysis_json_str, re.DOTALL)
        if not match:
            print(f"Error: Could not find a JSON object in the LLM response.")
            print(f"Raw Response:\n{analysis_json_str}")
            return {}
        
        json_str = match.group(0)

        # 2. Parse the JSON string into a Python dictionary
        analysis_result = json.loads(json_str)
        
        # 3. Validate the structure
        if "user_goal" not in analysis_result or "requirements" not in analysis_result:
            print("Error: Parsed JSON is missing 'user_goal' or 'requirements' key.")
            print(f"Parsed Data: {analysis_result}")
            return {}

        print("--- Query Analysis Successful ---")
        print(f"User Goal: {analysis_result.get('user_goal')}")
        print(f"Decomposed into {len(analysis_result.get('requirements', []))} sub-questions.")
        print("---------------------------------")
        
        return analysis_result

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from LLM response. {e}")
        print(f"Raw Response that caused error:\n{analysis_json_str}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during query analysis: {e}")
        return {}

def retrieve_and_extract_facts_two_stage(search_query: str, requirement: dict, collected_facts: dict, original_query: str) -> dict:
    """
    Retrieves context based on the search query and performs two-stage fact extraction:
    1. Extract potential evidence (maximal recall)
    2. Generate reasoned facts from the evidence

    Args:
        search_query: The search query to find relevant documents.
        requirement: A single requirement dictionary to be fulfilled.
        collected_facts: A dictionary containing facts collected in previous steps.
        original_query: The original user query.

    Returns:
        A dictionary containing the list of reasoned facts, or an empty dictionary on failure.
    """
    print(f"\n--- Stage: Retrieving Context and Two-Stage Fact Extraction for search query: '{search_query}' ---")

    # 1. Retrieve context using the search query
    search_hits = retrieve_context(search_query)

    # 2. Prepare context string from search hits
    retrieved_documents_str = "No relevant context found."
    if search_hits:
        docs_list = []
        for i, hit in enumerate(search_hits):
            content = hit.get('contents', str(hit))
            title = hit.get('title', '')
            if title:
                docs_list.append(f'<document id={i+1} title="{title}">\n{content}\n</document>')
            else:
                docs_list.append(f'<document id={i+1}>\n{content}\n</document>')
        retrieved_documents_str = "\n\n".join(docs_list)
        if config.SHOW_RETRIEVED_CONTEXT:
            print("--- Retrieved Context ---")
            print(retrieved_documents_str)
            print("-------------------------")
        else:
            print("--- [Context display disabled] --- ")
        _logger = get_pipeline_logger()
        if _logger:
            _logger.log(f"Retrieved Documents for '{search_query}'", {"search_query": search_query, "document_count": len(search_hits), "documents": retrieved_documents_str})
        
    else:
        print("No relevant context found.")

    # === Stage 1: Extract Potential Evidence ===
    # Call the thinker module to extract potential evidence (maximal recall)
    potential_evidence = thinker.extract_potential_evidence_stage1(
        search_query=search_query,
        retrieved_documents=retrieved_documents_str,
        search_hits=search_hits,
        requirement=requirement,
        original_query=original_query,
        save_to_memory=True
    )

    # If Stage 1 returned empty evidence, we cannot proceed
    if not potential_evidence:
        print("❌ Stage 1: No evidence extracted. Cannot proceed to Stage 2. Fallback to using original documents.")
        potential_evidence = retrieved_documents_str
        # return {}

    # === Stage 2: Generate Sub-Answers from Evidence ===
    return generator.generate_sub_answers_stage2(
        search_query=search_query,
        requirement=requirement,
        potential_evidence=potential_evidence,
        original_query=original_query
    )


def retrieve_and_extract_facts(search_query: str, requirement: dict, collected_facts: dict) -> dict:
    """
    Retrieves context based on the search query and then calls the LLM to extract facts
    for a single requirement.

    Args:
        search_query: The search query to find relevant documents.
        requirement: A single requirement dictionary to be fulfilled.
        collected_facts: A dictionary containing facts collected in previous steps.

    Returns:
        A dictionary containing the list of reasoned facts, or an empty dictionary on failure.
    """
    print(f"\n--- Stage: Retrieving Context and Extracting Facts for search query: '{search_query}' ---")

    # 1. Retrieve context using the search query
    search_hits = retrieve_context(search_query)

    # 2. Prepare context string from search hits, similar to perform_rag
    retrieved_documents_str = "No relevant context found."
    if search_hits:
        docs_list = []
        for i, hit in enumerate(search_hits):
            content = hit.get('contents', str(hit))
            title = hit.get('title', '')
            if title:
                docs_list.append(f'<document id={i+1} title="{title}">\n{content}\n</document>')
            else:
                docs_list.append(f'<document id={i+1}>\n{content}\n</document>')
        retrieved_documents_str = "\n\n".join(docs_list)
        print("--- Retrieved Context ---")
        if config.SHOW_RETRIEVED_CONTEXT:
            print(retrieved_documents_str)
        else:
            print("[Context display disabled]")
        print("-------------------------")
    else:
        print("No relevant context found.")

    # 3. Format the single requirement and known facts into strings
    active_requirement_str = json.dumps(requirement, indent=2)
    facts_list_str = json.dumps(collected_facts, indent=2)

    # 4. Call the LLM adapter to extract facts
    try:
        facts_json_str = llm_adapter.extract_facts(
            query=search_query,
            active_requirement=active_requirement_str,
            known_facts=facts_list_str,
            retrieved_documents=retrieved_documents_str
        )

        # 5. Parse the JSON response from the LLM
        match = re.search(r'\{.*\}', facts_json_str, re.DOTALL)
        if not match:
            print("Error: Could not find a JSON object in the fact extraction response.")
            print(f"Raw Response:\n{facts_json_str}")
            return {}

        json_str = match.group(0)
        extracted_data = json.loads(json_str)

        if isinstance(extracted_data, dict) and "reasoned_facts" not in extracted_data:
            if "statement" in extracted_data:
                print("Info: LLM returned a single fact object. Wrapping it in the correct structure.")
                extracted_data = {"reasoned_facts": [extracted_data]}

        if "reasoned_facts" not in extracted_data:
            print("Error: Parsed JSON from fact extraction is missing 'reasoned_facts' key.")
            print(f"Parsed Data: {extracted_data}")
            return {}

        print("--- Fact Extraction Successful ---")
        num_facts = len(extracted_data.get('reasoned_facts', []))
        print(f"Extracted {num_facts} facts.")
        print("--------------------------------")

        return extracted_data

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from fact extraction response. {e}")
        print(f"Raw Response that caused error:\n{facts_json_str}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during fact extraction: {e}")
        return {}

def update_plan(query: str, collected_facts: dict, pending_requirements: list) -> dict:
    """
    Updates the plan based on the current state (collected facts vs. pending requirements).
    It calls the LLM to get a decision, which is either to continue searching or to synthesize an answer.
    This function expects a response containing only a 'decision' object.

    Args:
        query: The original user query.
        collected_facts: A dictionary containing the list of facts collected so far.
        pending_requirements: A list of requirement dictionaries that are not yet fulfilled.

    Returns:
        A dictionary containing the decision for the next step, or an empty dictionary on failure.
    """
    print(f"\n--- Stage: Updating Plan for query: '{query}' ---")
    
    # 1. Format the inputs into JSON strings for the LLM prompt
    collected_facts_str = json.dumps(collected_facts, indent=2)
    pending_requirements_str = json.dumps(pending_requirements, indent=2)

    # 2. Call the LLM adapter to get the update decision
    try:
        update_json_str = llm_adapter.update_plan(
            query=query,
            collected_facts=collected_facts_str,
            pending_requirements=pending_requirements_str
        )

        # 3. Parse the JSON response from the LLM
        match = re.search(r'\{.*\}', update_json_str, re.DOTALL)
        if not match:
            print("Error: Could not find a JSON object in the plan update response.")
            print(f"Raw Response:\n{update_json_str}")
            return {}
        
        json_str = match.group(0)
        update_result = json.loads(json_str)

        # 4. Validate the structure
        if "decision" not in update_result:
            print("Error: Parsed JSON from plan update is missing 'decision' key.")
            print(f"Parsed Data: {update_result}")
            return {}

        print("--- Plan Update Successful ---")
        decision = update_result.get("decision", {})
        next_step = decision.get("next_step", "UNKNOWN")
        print(f"Decision: {next_step}")
        if next_step == "CONTINUE_SEARCH":
            next_actions = decision.get("next_actions", [])
            print(f"Next Actions: {len(next_actions)} new search(es) planned.")
        print("----------------------------")

        return update_result

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from plan update response. {e}")
        print(f"Raw Response that caused error:\n{update_json_str}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during plan update: {e}")
        return {}

def replan_questions(query: str, collected_facts: dict, pending_requirements: list) -> dict:
    """
    Analyzes the current state (collected facts vs. pending requirements) to decide the next action.
    It calls the LLM to get a decision, which is either to continue searching or to synthesize an answer.

    Args:
        query: The original user query.
        collected_facts: A dictionary containing the list of facts collected so far.
        pending_requirements: A list of requirement dictionaries that are not yet fulfilled.

    Returns:
        A dictionary containing the analysis and decision for the next step, or an empty dictionary on failure.
    """
    print(f"\n--- Stage: Replanning Next Actions for query: '{query}' ---")
    
    # 1. Format the inputs into JSON strings for the LLM prompt
    collected_facts_str = json.dumps(collected_facts, indent=2)
    pending_requirements_str = json.dumps(pending_requirements, indent=2)

    # 2. Call the LLM adapter to get the replan decision
    try:
        replan_json_str = llm_adapter.replan_conditions(
            query=query,
            collected_facts=collected_facts_str,
            pending_requirements=pending_requirements_str
        )

        # 3. Parse the JSON response from the LLM
        match = re.search(r'\{.*\}', replan_json_str, re.DOTALL)
        if not match:
            print("Error: Could not find a JSON object in the replanning response.")
            print(f"Raw Response:\n{replan_json_str}")
            return {}
        
        json_str = match.group(0)
        replan_result = json.loads(json_str)

        # 4. Validate the structure
        if "analysis" not in replan_result or "decision" not in replan_result:
            print("Error: Parsed JSON from replanning is missing 'analysis' or 'decision' key.")
            print(f"Parsed Data: {replan_result}")
            return {}

        print("--- Replanning Successful ---")
        decision = replan_result.get("decision", {})
        next_step = decision.get("next_step", "UNKNOWN")
        print(f"Decision: {next_step}")
        if next_step == "CONTINUE_SEARCH":
            next_actions = decision.get("next_actions", [])
            print(f"Next Actions: {len(next_actions)} new search(es) planned.")
        print("---------------------------")

        return replan_result

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from replanning response. {e}")
        print(f"Raw Response that caused error:\n{replan_json_str}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during replanning: {e}")
        return {}

def synthesize_final_answer(query: str, collected_facts: dict) -> dict:
    """
    Backward-compatible wrapper for final answer synthesis.

    The generator module owns final answer synthesis; this wrapper preserves the
    existing core API used by the pipeline and evaluation scripts.
    """
    return generator.synthesize_final_answer(query=query, collected_facts=collected_facts)

"""
Thinker module for Stage-1 evidence extraction in two-stage fact extraction.
This module handles the maximal recall phase of evidence extraction and saves
the extracted evidence to memory for future reuse.
"""
import json
import re
from typing import List, Dict, Any
from rag_pipeline_lib import llm_adapter
from rag_pipeline_lib.memory_manager import get_memory_manager
from rag_pipeline_lib.pipeline_logger import get_pipeline_logger


def extract_potential_evidence_stage1(
    search_query: str,
    retrieved_documents: str,
    search_hits: List[Dict[str, Any]],
    requirement: Dict[str, Any],
    original_query: str,
    save_to_memory: bool = True
) -> List[Dict[str, Any]]:
    """
    Stage 1: Extract potential evidence from retrieved documents (maximal recall).

    This function attempts to extract ALL potentially relevant evidence from
    the retrieved documents using the LLM. If the LLM extraction fails, it
    falls back to using the original documents as evidence.

    Args:
        search_query: The search query used to retrieve documents
        retrieved_documents: Formatted string containing the retrieved documents
        search_hits: List of search hit dictionaries from the retrieval service
        requirement: The requirement dictionary containing:
            - requirement_id: ID of the requirement (e.g., "req1")
            - question: The question for this requirement
        original_query: The original user query (for memory storage)
        save_to_memory: Whether to save evidence to memory (default: True)

    Returns:
        A list of potential evidence dictionaries. Each evidence dictionary contains:
        - evidence_quote: Exact string from text (from LLM) OR content (from fallback)
        - relevance_type: Category classification (from LLM) OR "Contextual" (from fallback)
        - latent_connection: Optional source note added only by the fallback path

        Returns empty list if both LLM extraction and fallback fail.
    """
    print("\n--- Stage 1: Extracting Potential Evidence (Maximal Recall) ---")
    potential_evidence = None
    stage1_failed = False

    try:
        current_sub_query = requirement.get('question', search_query)

        # Attempt LLM-based extraction
        evidence_json_str = llm_adapter.extract_potential_evidence(
            current_sub_query=current_sub_query,
            original_query=original_query,
            retrieved_documents=retrieved_documents
        )

        # Parse the JSON array response
        match = re.search(r'\[.*\]', evidence_json_str, re.DOTALL)
        if not match:
            print("Error: Could not find a JSON array in the potential evidence response.")
            print(f"Raw Response:\n{evidence_json_str}")
            stage1_failed = True
        else:
            json_str = match.group(0)
            potential_evidence = json.loads(json_str)
            print(potential_evidence)

            if not isinstance(potential_evidence, list):
                print("Error: Potential evidence is not a list.")
                stage1_failed = True
            else:
                print(f"✅ Extracted {len(potential_evidence)} potential evidence fragments.")
                _logger = get_pipeline_logger()
                if _logger:
                    _logger.log(f"Stage 1: Potential Evidence for '{current_sub_query}'", {"evidence_count": len(potential_evidence), "evidence": potential_evidence})

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from potential evidence response. {e}")
        print(f"Raw Response that caused error:\n{evidence_json_str}")
        stage1_failed = True
    except Exception as e:
        print(f"An unexpected error occurred during potential evidence extraction: {e}")
        stage1_failed = True

    # Fallback: If Stage 1 failed, use original documents as evidence
    if stage1_failed:
        print("⚠️  Stage 1 failed. Using original documents as evidence for Stage 2.")
        potential_evidence = _convert_search_hits_to_evidence(search_hits)

        if not potential_evidence:
            print("No search hits available to use as evidence.")
            return []

    # Save to memory if requested
    if save_to_memory and potential_evidence:
        _save_evidence_to_memory(
            original_query=original_query,
            requirement=requirement,
            search_query=search_query,
            evidences=potential_evidence
        )

    return potential_evidence


def _save_evidence_to_memory(
    original_query: str,
    requirement: Dict[str, Any],
    search_query: str,
    evidences: List[Dict[str, Any]]
):
    """
    Save extracted evidence to memory.

    Args:
        original_query: The original user query
        requirement: The requirement dictionary
        search_query: The search query used
        evidences: List of extracted evidence dictionaries
    """
    try:
        memory_manager = get_memory_manager()

        requirement_id = requirement.get('requirement_id', 'unknown')
        requirement_question = requirement.get('question', search_query)

        query_id = memory_manager.save_requirement_evidence(
            original_query=original_query,
            requirement_id=requirement_id,
            requirement_question=requirement_question,
            search_query=search_query,
            evidences=evidences
        )

        print(f"💾 Evidence saved to memory (query_id: {query_id}, requirement: {requirement_id})")

    except Exception as e:
        print(f"⚠️  Warning: Failed to save evidence to memory: {e}")


def _convert_search_hits_to_evidence(search_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fallback function: Convert search hits to evidence format when LLM extraction fails.

    Args:
        search_hits: List of search hit dictionaries from the retrieval service

    Returns:
        A list of evidence dictionaries in the standard format
    """
    potential_evidence = []

    if not search_hits:
        return potential_evidence

    for i, hit in enumerate(search_hits):
        content = hit.get('contents', str(hit))
        title = hit.get('title', '')

        # Create evidence item in standard format
        evidence_item = {
            "evidence_quote": content,  # Use full content as quote
            "latent_connection": f"Source: {title if title else f'Document {i + 1}'}",
            "relevance_type": "Contextual"  # Default to contextual for fallback
        }
        potential_evidence.append(evidence_item)

    print(f"📄 Converted {len(potential_evidence)} search hits to evidence format (fallback mode).")
    return potential_evidence

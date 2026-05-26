"""
Generator module for answer generation in the multi-agent RAG pipeline.

The generator owns:
- Stage 2 sub-answer generation from potential evidence.
- Final answer synthesis from collected facts.
"""
import json
import re
from typing import Any, Dict, Optional

import config
from rag_pipeline_lib import llm_adapter
from rag_pipeline_lib.pipeline_logger import get_pipeline_logger


def generate_sub_answers_stage2(
    search_query: str,
    requirement: Dict[str, Any],
    potential_evidence: Any,
    original_query: str,
) -> Dict[str, Any]:
    """
    Stage 2: Generate reasoned sub-answers from potential evidence.

    The pipeline still consumes the historical `reasoned_facts` schema, so this
    function preserves that output contract while making the generator the owner
    of the generation step.
    """
    print("\n--- Stage 2: Generator Creating Sub-Answers from Evidence ---")

    current_sub_query = requirement.get("question", search_query)
    requirement_id = requirement.get("requirement_id", "unknown")
    evidence_str = _format_evidence_for_prompt(potential_evidence)
    facts_json_str = ""

    try:
        facts_json_str = llm_adapter.generate_reasoned_facts(
            original_query=original_query,
            current_sub_query=current_sub_query,
            extracted_evidence=evidence_str,
            requirement_id=requirement_id,
        )

        extracted_data = _parse_reasoned_facts_response(facts_json_str)
        if not extracted_data:
            return {}

        num_facts = len(extracted_data.get("reasoned_facts", []))
        print(f"✅ Generated {num_facts} reasoned sub-answer(s).")
        print("--- Generator Stage 2 Successful ---")
        print("--------------------------------")
        _logger = get_pipeline_logger()
        if _logger:
            _logger.log(f"Stage 2: Sub-Answers for '{current_sub_query}'", extracted_data)

        return extracted_data

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from reasoned facts response. {e}")
        print(f"Raw Response that caused error:\n{facts_json_str}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during sub-answer generation: {e}")
        return {}


def synthesize_final_answer(query: str, collected_facts: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate the final answer from all collected facts.

    Args:
        query: The original user query.
        collected_facts: Facts gathered during the pipeline.

    Returns:
        A dictionary with `answer` and `reasoning` keys.
    """
    print(f"\n--- Stage: Generator Synthesizing Final Answer for query: '{query}' ---")

    facts_str = json.dumps(collected_facts, indent=2)

    try:
        final_answer_response = llm_adapter.generate_final_answer(
            query=query,
            facts=facts_str,
        )
        if config.USE_FINAL_ANSWER_REASONING:
            final_answer = _parse_final_answer_response(final_answer_response)
        else:
            final_answer = {
                "answer": final_answer_response.strip(),
                "reasoning": "",
            }

        print("--- Final Answer Generation Successful ---")
        print(json.dumps(final_answer, indent=2, ensure_ascii=False))
        _logger = get_pipeline_logger()
        if _logger:
            _logger.log("Final Answer Synthesis", final_answer)
        return final_answer

    except Exception as e:
        error_message = f"An unexpected error occurred during final answer synthesis: {e}"
        print(error_message)
        return {
            "answer": error_message,
            "reasoning": "",
        }


def _format_evidence_for_prompt(potential_evidence: Any) -> str:
    """Format potential evidence as prompt-ready JSON/text."""
    if isinstance(potential_evidence, str):
        return potential_evidence
    return json.dumps(potential_evidence, indent=2, ensure_ascii=False)


def _parse_reasoned_facts_response(facts_json_str: str) -> Dict[str, Any]:
    """Parse the Stage 2 LLM response into the `reasoned_facts` schema."""
    match = re.search(r"\{.*\}", facts_json_str, re.DOTALL)
    if not match:
        print("Error: Could not find a JSON object in the reasoned facts response.")
        print(f"Raw Response:\n{facts_json_str}")
        return {}

    extracted_data = json.loads(match.group(0))

    if isinstance(extracted_data, dict) and "reasoned_facts" not in extracted_data:
        if "statement" in extracted_data:
            print("Info: LLM returned a single fact object. Wrapping it in the correct structure.")
            extracted_data = {"reasoned_facts": [extracted_data]}

    if "reasoned_facts" not in extracted_data:
        print("Error: Parsed JSON from reasoned facts generation is missing 'reasoned_facts' key.")
        print(f"Parsed Data: {extracted_data}")
        return {}

    return extracted_data


def _parse_final_answer_response(raw_response: str) -> Dict[str, str]:
    """Parse the final answer LLM response into {'answer': ..., 'reasoning': ...}."""
    if not raw_response:
        return {"answer": "", "reasoning": ""}

    parsed_payload: Optional[Dict[str, Any]] = None
    response_text = raw_response.strip()

    try:
        loaded = json.loads(response_text)
        if isinstance(loaded, dict):
            parsed_payload = loaded
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                loaded = json.loads(match.group(0))
                if isinstance(loaded, dict):
                    parsed_payload = loaded
            except json.JSONDecodeError:
                parsed_payload = None

    if parsed_payload is None:
        return {"answer": response_text, "reasoning": ""}

    answer = parsed_payload.get("answer", "")
    reasoning = parsed_payload.get("reasoning", "")

    if answer is None:
        answer = ""
    if reasoning is None:
        reasoning = ""

    return {
        "answer": str(answer).strip(),
        "reasoning": str(reasoning).strip(),
    }

"""
Planner module for query planning and replanning in the RAG pipeline.
Provides different strategies for deciding next actions during iterative fact extraction.
"""
import json
import re
from rag_pipeline_lib import llm_adapter
from rag_pipeline_lib.memory_manager import get_memory_manager
from rag_pipeline_lib.pipeline_logger import get_pipeline_logger


def new_replan_questions(
    query: str,
    collected_facts: dict,
    pending_requirements: list,
    failed_requirements: list = None,
) -> dict:
    """
    New two-step replanning strategy with failure attribution and intelligent recovery.

    This method implements a two-step process:
    Step 1 (Failure Attribution): Analyze why requirements failed using execution history from memory
    Step 2 (Replan): Generate a recovery plan based on the failure analysis

    The two-step approach provides:
    - Deep diagnosis of root causes (ENOUGH_TO_UPDATE, UPSTREAM_ISSUE, REQUIREMENT_DEFICIENCY, DECOMPOSITION_DEFICIENCY)
    - Strategic recovery based on problem type (REWRITE, DECOMPOSE, REPLAN_FROM_ROOT_CAUSE)
    - Better handling of complex multi-hop failures

    Args:
        query: The original user query.
        collected_facts: A dictionary containing the list of facts collected so far.
        pending_requirements: A list of requirement dictionaries that are not yet fulfilled.
        failed_requirements: The requirement dictionaries from the previous execution
            round that triggered replanning. If omitted, falls back to inferring failed
            requirements from the current pending plan for backward compatibility.

    Returns:
        A dictionary containing:
        {
            "analysis": {
                "problem_diagnosis": "...",
                "recovery_strategy": "<REWRITE | DECOMPOSE | REPLAN_FROM_ROOT_CAUSE>",
                "strategy_rationale": "..."
            },
            "decision": {
                "next_step": "CONTINUE_SEARCH" or "SYNTHESIZE_ANSWER",
                "updated_plan": [...],
                "next_actions": [...]
            }
        }
        or an empty dictionary on failure.
    """
    print(f"\n--- Stage: NEW Two-Step Replanning for query: '{query}' ---")
    print("🆕 Using intelligent failure attribution and recovery")

    try:
        # === PREPARATION: Get Full Execution History from Memory ===
        memory_manager = get_memory_manager()
        execution_history = memory_manager.get_full_execution_history(
            original_query=query,
            collected_facts=collected_facts,
            pending_requirements=pending_requirements
        )

        print(f"📊 Execution History: {execution_history.get('execution_summary', 'N/A')}")

        if failed_requirements is None:
            # Backward-compatible inference for callers that do not provide the
            # exact requirements that failed in the previous execution round.
            failed_requirements = []
            for req in pending_requirements:
                req_id = req.get('requirement_id')
                # Check facts for this requirement
                req_facts = [
                    f for f in collected_facts.get("reasoned_facts", [])
                    if f.get("fulfills_requirement_id") == req_id
                ]
                # If there are facts but none are DIRECT_ANSWER, this is a failure
                if req_facts:
                    has_direct = any(f.get("fulfillment_level") == "DIRECT_ANSWER" for f in req_facts)
                    if not has_direct:
                        failed_requirements.append(req)
                else:
                    # No facts collected yet for this requirement
                    failed_requirements.append(req)
        else:
            failed_requirements = list(failed_requirements)

        if not failed_requirements:
            print("⚠️  No failed requirements detected. All requirements are either completed or haven't been attempted yet.")
            # Fall back to standard replan
            return _fallback_to_standard_replan(query, collected_facts, pending_requirements)

        failed_req_ids = [req.get('requirement_id') for req in failed_requirements]
        print(f"🔍 Analyzing failed requirement(s): {failed_req_ids}")

        # Format inputs for the two-step process

        # Step 1: Format completed requirements
        completed_reqs = execution_history.get('completed_requirements', [])
        completed_query_str = json.dumps(completed_reqs, indent=2, ensure_ascii=False)

        # Format failed requirement(s). Keep the legacy single-object payload for
        # single failures, but pass a JSON array when parallel actions failed.
        failed_requirement_payload = failed_requirements[0] if len(failed_requirements) == 1 else failed_requirements
        failed_requirement_str = json.dumps(failed_requirement_payload, indent=2, ensure_ascii=False)

        # Format original plan (all requirements)
        original_plan = {
            "original_query": query,
            "completed_requirements": completed_reqs,
            "pending_requirements": pending_requirements
        }
        original_plan_str = json.dumps(original_plan, indent=2, ensure_ascii=False)

        # === STEP 1: FAILURE ATTRIBUTION ===
        print("\n📋 Step 1: Failure Attribution - Diagnosing root cause...")

        attribution_response = llm_adapter.failure_attribution(
            original_query=query,
            completed_query=completed_query_str,
            failed_requirement_query=failed_requirement_str
        )

        # Parse attribution response
        match = re.search(r'\{.*\}', attribution_response, re.DOTALL)
        if not match:
            print("Error: Could not parse failure attribution response as JSON.")
            print(f"Raw Response:\n{attribution_response}")
            return _fallback_to_standard_replan(query, collected_facts, pending_requirements)

        attribution_json = match.group(0)
        attribution_result = json.loads(attribution_json)

        analysis_type = attribution_result.get("analysis_type")
        reasoning = attribution_result.get("reasoning", "")
        recommendation = attribution_result.get("recommendation", "")

        print(f"✅ Failure Attribution Complete:")
        print(f"   Analysis Type: {analysis_type}")
        print(f"   Reasoning: {reasoning}")
        print(f"   Recommendation: {recommendation}")
        _logger = get_pipeline_logger()
        if _logger:
            _logger.log("Replanning - Failure Attribution", attribution_result)

        # === BRANCH: Handle ENOUGH_TO_UPDATE separately ===
        if analysis_type == "ENOUGH_TO_UPDATE":
            print("\n🔄 Step 2: Using lightweight update with sufficient evidence...")

            # Format inputs for enough_to_update_plan
            collected_facts_str = json.dumps(collected_facts, indent=2, ensure_ascii=False)
            pending_requirements_str = json.dumps(pending_requirements, indent=2, ensure_ascii=False)

            try:
                update_response = llm_adapter.enough_to_update_plan(
                    query=query,
                    collected_facts=collected_facts_str,
                    pending_requirements=pending_requirements_str
                )

                # Parse the response
                match = re.search(r'\{.*\}', update_response, re.DOTALL)
                if not match:
                    print("Error: Could not parse enough_to_update response as JSON.")
                    print(f"Raw Response:\n{update_response}")
                    return _fallback_to_standard_replan(query, collected_facts, pending_requirements)

                update_json = match.group(0)
                update_result = json.loads(update_json)

                # Validate structure
                if "analysis" not in update_result or "decision" not in update_result:
                    print("Error: Update response missing 'analysis' or 'decision' key.")
                    return _fallback_to_standard_replan(query, collected_facts, pending_requirements)

                print("✅ Plan Update with Sufficient Evidence Successful")

                _logger = get_pipeline_logger()
                if _logger:
                    _logger.log("Replanning - ENOUGH_TO_UPDATE Plan", update_result)
                analysis = update_result.get("analysis", {})
                print("\n📊 Update Analysis:")
                if "evidence_summary" in analysis:
                    print(f"  📋 Evidence: {analysis['evidence_summary']}")
                if "update_strategy" in analysis:
                    print(f"  🔧 Strategy: {analysis['update_strategy']}")

                # Print decision
                decision = update_result.get("decision", {})
                next_step = decision.get("next_step", "UNKNOWN")
                print(f"\n🎯 Next Step: {next_step}")

                if next_step == "CONTINUE_SEARCH":
                    next_actions = decision.get("next_actions", [])
                    print(f"🔎 Next Actions: {len(next_actions)} action(s) planned")
                    for i, action in enumerate(next_actions, 1):
                        req_id = action.get("requirement_id", "unknown")
                        question = action.get("question", "")
                        reasoning = action.get("reasoning", "")
                        print(f"  {i}. [{req_id}] {question}")
                        if reasoning:
                            print(f"     → {reasoning}")

                print("---------------------------")
                return update_result

            except json.JSONDecodeError as e:
                print(f"Error: Failed to decode JSON from update response. {e}")
                return _fallback_to_standard_replan(query, collected_facts, pending_requirements)
            except Exception as e:
                print(f"Error: Unexpected error during plan update: {e}")
                import traceback
                traceback.print_exc()
                return _fallback_to_standard_replan(query, collected_facts, pending_requirements)

        # === STEP 2: REPLAN WITH ANALYSIS (for other 3 failure types) ===
        else:
            print(f"\n🔧 Step 2: Replan - Generating recovery strategy for {analysis_type}...")

            # Format the problem analysis for step 2
            problem_analysis_str = json.dumps(attribution_result, indent=2, ensure_ascii=False)

            replan_response = llm_adapter.replan_with_analysis(
                original_query=query,
                original_plan=original_plan_str,
                failed_requirement=failed_requirement_str,
                problem_analysis=problem_analysis_str
            )

            # Parse replan response
            match = re.search(r'\{.*\}', replan_response, re.DOTALL)
            if not match:
                print("Error: Could not parse replan response as JSON.")
                print(f"Raw Response:\n{replan_response}")
                return _fallback_to_standard_replan(query, collected_facts, pending_requirements)

            replan_json = match.group(0)
            replan_result = json.loads(replan_json)

            # Validate structure
            if "analysis" not in replan_result or "decision" not in replan_result:
                print("Error: Replan response missing 'analysis' or 'decision' key.")
                return _fallback_to_standard_replan(query, collected_facts, pending_requirements)

            print("✅ Two-Step Replanning Successful")
            _logger = get_pipeline_logger()
            if _logger:
                _logger.log("Replanning - Recovery Plan", replan_result)
            decision = replan_result.get("decision", {})

            # Print detailed analysis
            analysis = replan_result.get("analysis", {})
            print("\n📊 Recovery Plan Analysis:")
            if analysis:
                if "problem_diagnosis" in analysis:
                    print(f"  📋 Diagnosis: {analysis['problem_diagnosis']}")
                if "recovery_strategy" in analysis:
                    print(f"  🔧 Strategy: {analysis['recovery_strategy']}")
                if "strategy_rationale" in analysis:
                    print(f"  💡 Rationale: {analysis['strategy_rationale']}")

            next_step = decision.get("next_step", "UNKNOWN")
            print(f"\n🎯 Next Step: {next_step}")

            if next_step == "CONTINUE_SEARCH":
                next_actions = decision.get("next_actions", [])
                print(f"🔎 Next Actions: {len(next_actions)} action(s) planned")

                # Print detailed next actions
                for i, action in enumerate(next_actions, 1):
                    req_id = action.get("requirement_id", "unknown")
                    question = action.get("question", "")
                    reasoning = action.get("reasoning", "")
                    print(f"  {i}. [{req_id}] {question}")
                    if reasoning:
                        print(f"     → {reasoning}")

            print("---------------------------")

            return replan_result

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON during two-step replanning. {e}")
        return _fallback_to_standard_replan(query, collected_facts, pending_requirements)
    except Exception as e:
        print(f"Error: Unexpected error during two-step replanning: {e}")
        import traceback
        traceback.print_exc()
        return _fallback_to_standard_replan(query, collected_facts, pending_requirements)


def _fallback_to_standard_replan(query: str, collected_facts: dict, pending_requirements: list) -> dict:
    """
    Fallback to standard replan if two-step process fails.
    This maintains backward compatibility and ensures robustness.
    """
    print("\n⚠️  Falling back to standard replan method")

    from rag_pipeline_lib import core as rag_core

    try:
        # Use the existing replan_questions from core.py
        result = rag_core.replan_questions(
            query=query,
            collected_facts=collected_facts,
            pending_requirements=pending_requirements
        )

        if result:
            print("✅ Fallback to standard replan successful")
        return result

    except Exception as e:
        print(f"❌ Fallback replan also failed: {e}")
        return {}

from concurrent.futures import ThreadPoolExecutor, as_completed
from rag_pipeline_lib import core as rag_core
from rag_pipeline_lib import planner
import config
from rag_pipeline_lib.pipeline_logger import PipelineLogger, pipeline_logger_ctx
from contextvars import ContextVar, copy_context
from functools import partial
import json
import copy
import re
import time

tracer_context: ContextVar = ContextVar('tracer_context', default=None)

class Tracer:
    def __init__(self):
        self.log = []
        self.iteration_count = 0
        self._pending_log = []

    def record_llm_call(self, adapter_function_name, inputs, output):
        trace_entry = {
            "iteration": self.iteration_count,
            "adapter_function_name": adapter_function_name,
            "llm_inputs": copy.deepcopy(inputs),
            "llm_output": copy.deepcopy(output)
        }
        self._pending_log.append(trace_entry)

    def commit_pending(self):
        self.log.extend(self._pending_log)
        self._pending_log.clear()

    def discard_pending(self):
        self._pending_log.clear()

def run_multistep_pipeline(query: str, verbose: bool = True, trace_collector: Tracer = None, use_new_replan: bool = True, query_id: str | None = None) -> dict:
    current_tracer = tracer_context.get()
    if verbose:
        print(f"\n##############################################################")
        print(f"\n🚀 Starting new multi-step pipeline for query: '{query}'")

    # Initialize pipeline logger
    p_logger = PipelineLogger(query, query_id=query_id)
    pipeline_logger_ctx.set(p_logger)

    # Stage 1: Analyze and decompose query
    analysis_result = None
    max_analysis_retries = 3
    for attempt in range(max_analysis_retries):
        try:
            if verbose and attempt > 0:
                print(f"🔄 Retrying query analysis... (Attempt {attempt + 1}/{max_analysis_retries})")
            
            if current_tracer: current_tracer.discard_pending()
            analysis_result = rag_core.analyze_and_decompose_query(query=query)
            
            if not analysis_result or "requirements" not in analysis_result:
                raise ValueError("Analysis result is empty or missing 'requirements' key.")
            
            if current_tracer: current_tracer.commit_pending()
            if verbose: print("✅ Query analysis successful.")
            break 
        except Exception as e:
            if verbose:
                print(f"❌ Query analysis failed on attempt {attempt + 1}. Error: {e}")
            if attempt == max_analysis_retries - 1:
                if current_tracer: current_tracer.discard_pending()
                error_msg = f"Pipeline Error: Failed to analyze and decompose the query after {max_analysis_retries} attempts."
                if verbose: print(f"\n❌ {error_msg}")
                return {"answer": error_msg, "reasoning": ""}
    
    if not analysis_result:
        return {"answer": "Pipeline Error: Could not obtain a valid query analysis.", "reasoning": ""}

    if verbose:
        print("--- Initial Analysis and Requirements ---")
        print(json.dumps(analysis_result, indent=2, ensure_ascii=False))
        print("------------------------------------")
    p_logger.log("Stage 1: Query Analysis & Decomposition", analysis_result)

    # Initialize state
    all_requirements = analysis_result.get("requirements", [])
    pending_requirements = list(all_requirements)
    req_id_to_question = {req['requirement_id']: req['question'] for req in all_requirements}
    collected_facts = {"reasoned_facts": []}
    max_iterations = 5
    max_total_attempts = 10
    last_extraction_was_direct_only = True 
    last_failed_requirements_for_replan = []

    # Stage 2 & 3: Use while loop for iterative fact extraction and planning
    iteration_count = 0
    total_attempt_count = 0
    while iteration_count < max_iterations:
        total_attempt_count += 1 # Increment total attempt count for each loop
        if total_attempt_count > max_total_attempts:
            if verbose: print(f"\n⚠️ Warning: Reached maximum total attempts ({max_total_attempts}) due to repeated failures. Moving to synthesis.")
            break

        if not pending_requirements:
            if verbose: print("\n✅ All requirements have been fulfilled. Moving to final answer synthesis.")
            break
        
        # If tracer exists, we only update iteration count
        if current_tracer:
            current_tracer.iteration_count = iteration_count + 1
            current_tracer.discard_pending() # At the start of each iteration attempt, discard logs from failed previous round

        if verbose:
            print(f"\n--- Iteration {iteration_count + 1}/{max_iterations} ---")
            print(f"📝 Pending Requirements: {[req['question'] for req in pending_requirements]}")

        # Save all potentially modified state before iteration begins
        facts_before_iteration = [fact for fact in collected_facts["reasoned_facts"]]
        pending_reqs_before_iteration = list(pending_requirements)
        req_map_before_iteration = dict(req_id_to_question)
        last_direct_before_iteration = last_extraction_was_direct_only
        failed_reqs_before_iteration = list(last_failed_requirements_for_replan)
        
        try:
            # 1. Planning step
            decision_result = None
            if last_extraction_was_direct_only:
                if verbose: print("\n🔄 Last extraction was successful. Using lightweight 'update_plan'.")
                decision_result = rag_core.update_plan(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)
            else:
                if verbose: print("\n🤔 Last extraction had partial clues or failures. Using full 'replan_questions'.")
                iteration_count += 1
                if use_new_replan:
                    if verbose: print("🆕 Using NEW replan strategy from planner.py")
                    decision_result = planner.new_replan_questions(
                        query=query,
                        collected_facts=collected_facts,
                        pending_requirements=pending_requirements,
                        failed_requirements=last_failed_requirements_for_replan
                    )
                else:
                    if verbose: print("📋 Using original replan strategy from core.py")
                    decision_result = rag_core.replan_questions(query=query, collected_facts=collected_facts, pending_requirements=pending_requirements)

            if not decision_result or "decision" not in decision_result:
                raise ValueError("Planning step failed to return a valid decision.")
            if verbose:
                print("--- Planning Decision ---")
                print(json.dumps(decision_result, indent=2, ensure_ascii=False))
                print("-------------------------")
            p_logger.log(f"Iteration {iteration_count + 1} - Planning Decision", decision_result)
            decision = decision_result.get("decision", {})

            # Update pending requirements based on updated_plan in planning result
            if "updated_plan" in decision and isinstance(decision["updated_plan"], list):
                if verbose: print("🔄 Updating pending requirements based on the new plan.")
                pending_requirements = decision["updated_plan"]
                req_id_to_question = {req['requirement_id']: req['question'] for req in pending_requirements}

            next_step = decision.get("next_step")
            if next_step == "SYNTHESIZE_ANSWER":
                if verbose: print("✅ Planning module decided all necessary facts are collected. Moving to synthesis.")
                if current_tracer: current_tracer.commit_pending()
                break
            
            if next_step != "CONTINUE_SEARCH":
                raise ValueError(f"Received unexpected next step '{next_step}'.")

            next_actions = decision.get("next_actions") or decision.get("next_questions", [])
            p_logger.log(f"Iteration {iteration_count + 1} - Search Actions", next_actions)

            # Fallback mechanism: If next_actions is empty, use LLM to generate them
            if not next_actions:
                if verbose:
                    print("\n⚠️  Warning: Planner suggested to continue search but provided no actions.")
                    print("🔄 Activating fallback mechanism: Generating next actions via dedicated LLM call...")

                try:
                    # Format inputs for the fallback LLM call
                    collected_facts_str = json.dumps(collected_facts, indent=2)
                    pending_requirements_str = json.dumps(pending_requirements, indent=2)

                    # Call LLM to generate next_actions
                    fallback_response = rag_core.llm_adapter.generate_next_actions(
                        query=query,
                        collected_facts=collected_facts_str,
                        pending_requirements=pending_requirements_str
                    )

                    # Parse the JSON response
                    match = re.search(r'\{.*\}', fallback_response, re.DOTALL)
                    if not match:
                        raise ValueError("Fallback LLM response did not contain valid JSON.")

                    fallback_json = json.loads(match.group(0))
                    next_actions = fallback_json.get("next_actions", [])

                    if not next_actions:
                        raise ValueError("Fallback LLM also failed to generate next actions.")

                    if verbose:
                        print(f"✅ Fallback successful: Generated {len(next_actions)} next action(s).")
                        print("--- Fallback Generated Actions ---")
                        print(json.dumps(next_actions, indent=2, ensure_ascii=False))
                        print("----------------------------------")

                except Exception as fallback_error:
                    if verbose:
                        print(f"❌ Fallback mechanism failed: {fallback_error}")
                    raise ValueError("Planner suggested to continue search but provided no actions, and fallback mechanism also failed.") from fallback_error

            # 2. Execute fact extraction in parallel
            if verbose: print(f"\n🔎 Executing {len(next_actions)} search action(s) in parallel...")
            iteration_new_facts = []
            extraction_had_errors = False

            # Choose fact extraction method based on config
            use_two_stage = (config.FACT_EXTRACTION_MODE == 'two_stage')
            if verbose:
                mode_str = "two-stage" if use_two_stage else "single-stage"
                print(f"📊 Using {mode_str} fact extraction mode")

            with ThreadPoolExecutor() as executor:
                future_to_action = {}
                executed_requirements = []
                executed_req_ids = set()
                for action in next_actions:
                    # Find the matching requirement
                    matching_reqs = [req for req in pending_requirements if req['requirement_id'] == action.get("requirement_id")]
                    if not matching_reqs:
                        continue
                    requirement = matching_reqs[0]
                    req_id = requirement.get("requirement_id")
                    if req_id and req_id not in executed_req_ids:
                        executed_requirements.append(requirement)
                        executed_req_ids.add(req_id)

                    # Choose extraction function based on mode
                    if use_two_stage:
                        # Two-stage extraction: evidence + reasoned facts
                        future = executor.submit(
                            partial(copy_context().run, rag_core.retrieve_and_extract_facts_two_stage),
                            search_query=action.get("question"),
                            requirement=requirement,
                            collected_facts=collected_facts,
                            original_query=query
                        )
                    else:
                        # Single-stage extraction (legacy)
                        future = executor.submit(
                            partial(copy_context().run, rag_core.retrieve_and_extract_facts),
                            search_query=action.get("question"),
                            requirement=requirement,
                            collected_facts=collected_facts
                        )
                    future_to_action[future] = action

                for future in as_completed(future_to_action):
                    action = future_to_action[future]
                    try:
                        newly_extracted_data = future.result()
                        if verbose:
                            print(f"  - 📝 Result for '{action.get('question')}':")
                            try:
                                print(f"    {json.dumps(newly_extracted_data, indent=4, ensure_ascii=False)}")
                            except (TypeError, json.JSONDecodeError):
                                print(f"    (Could not format non-JSON or invalid JSON output: {newly_extracted_data})")

                        if not isinstance(newly_extracted_data, dict) or "reasoned_facts" not in newly_extracted_data:
                            raise ValueError(f"Invalid data structure received for '{action.get('question')}'.")

                        iteration_new_facts.extend(newly_extracted_data.get("reasoned_facts", []))
                    except Exception as exc:
                        if verbose: print(f"  - ❌ Error processing result for '{action.get('question')}': {exc}")
                        raise RuntimeError(f"Fact extraction failed for '{action.get('question')}'") from exc
            
            if verbose:
                print(f"  - ✅ Fact extraction phase completed. Found {len(iteration_new_facts)} new fact(s).")
            p_logger.log(f"Iteration {iteration_count + 1} - Extraction Results", {"new_facts_count": len(iteration_new_facts), "new_facts": iteration_new_facts})

            # 3. Update state

            if iteration_new_facts:
                processed_facts = []
                for fact in iteration_new_facts:
                    req_id = fact.get("fulfills_requirement_id")
                    if not req_id:
                        continue

                    question = req_id_to_question.get(req_id, "Unknown Question")
                    
                    processed_fact = {
                        "fulfills_requirement_id": req_id,
                        "requirement": question,
                        "reasoning": fact.get("reasoning"),
                        "statement": fact.get("statement"),
                        "fulfillment_level": fact.get("fulfillment_level")
                    }
                    processed_fact = {k: v for k, v in processed_fact.items() if v is not None}

                    processed_facts.append(processed_fact)

                collected_facts["reasoned_facts"].extend(processed_facts)
                fulfilled_req_ids = {fact['fulfills_requirement_id'] for fact in iteration_new_facts if fact.get("fulfillment_level") == "DIRECT_ANSWER"}
                pending_requirements = [req for req in pending_requirements if req['requirement_id'] not in fulfilled_req_ids]
                last_extraction_was_direct_only = all(fact.get("fulfillment_level") == "DIRECT_ANSWER" for fact in iteration_new_facts) and not extraction_had_errors
                facts_by_req_id = {}
                for fact in iteration_new_facts:
                    req_id = fact.get("fulfills_requirement_id")
                    if req_id:
                        facts_by_req_id.setdefault(req_id, []).append(fact)
                last_failed_requirements_for_replan = [
                    req for req in executed_requirements
                    if not any(
                        fact.get("fulfillment_level") == "DIRECT_ANSWER"
                        for fact in facts_by_req_id.get(req.get("requirement_id"), [])
                    )
                ]
            else:
                last_extraction_was_direct_only = False
                last_failed_requirements_for_replan = list(executed_requirements)

            # --- Successfully completed iteration, increment counter and commit logs ---
            if current_tracer: current_tracer.commit_pending()

        except (json.JSONDecodeError, ValueError, RuntimeError) as e:
            if verbose: 
                print(f"\n❌ Iteration {iteration_count + 1} failed: {e}")
                print("🔄 Rolling back state to the beginning of the iteration and retrying.")
            
            # --- State rollback ---
            collected_facts["reasoned_facts"] = facts_before_iteration
            pending_requirements = pending_reqs_before_iteration
            req_id_to_question = req_map_before_iteration
            last_extraction_was_direct_only = last_direct_before_iteration
            last_failed_requirements_for_replan = failed_reqs_before_iteration
            
            if current_tracer: current_tracer.discard_pending()
            
            continue
    else:
        if verbose: print("\n⚠️ Warning: Reached maximum iterations. Moving to synthesis with potentially incomplete facts.")

    # Stage 4: Synthesize final answer
    if verbose:
        print("\n--- Final Stage: Synthesizing Answer from Collected Facts ---")
        print("Collected Facts Summary:")
        print(json.dumps(collected_facts, indent=2, ensure_ascii=False))
    p_logger.log("Final Collected Facts Before Synthesis", collected_facts)
    
    if current_tracer: current_tracer.discard_pending()
    final_answer = rag_core.synthesize_final_answer(query=query, collected_facts=collected_facts)
    if current_tracer: current_tracer.commit_pending()
    p_logger.log("Final Answer", final_answer)
    p_logger.save()
    return final_answer

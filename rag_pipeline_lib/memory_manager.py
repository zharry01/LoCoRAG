"""
Memory Manager module for storing and retrieving potential evidence.
Saves evidence extracted by the thinker module for future reuse.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
from threading import Lock


class EvidenceMemory:
    """Manages storage and retrieval of extracted evidence."""

    def __init__(self, memory_dir: str = "data/evidence_memory", run_timestamp: Optional[str] = None):
        """
        Initialize the Evidence Memory manager.

        Args:
            memory_dir: Root directory to store memory files (default: data/evidence_memory)
            run_timestamp: Optional timestamp folder for this pipeline run
        """
        self.memory_root_dir = memory_dir
        self.run_timestamp = run_timestamp or self._generate_run_timestamp()
        self.memory_dir = os.path.join(memory_dir, self.run_timestamp)
        self._ensure_memory_dir_exists()

    def _generate_run_timestamp(self) -> str:
        """Generate a filesystem-safe timestamp for this pipeline run."""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _ensure_memory_dir_exists(self):
        """Create memory directory if it doesn't exist."""
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)
            print(f"📁 Created memory directory: {self.memory_dir}")

    def _generate_query_id(self, original_query: str) -> str:
        """
        Generate a unique ID for the query based on its content.

        Args:
            original_query: The original user query

        Returns:
            A unique ID string (hash of the query)
        """
        # Create a hash of the query for a unique but deterministic ID
        query_hash = hashlib.md5(original_query.encode('utf-8')).hexdigest()[:12]
        return f"query_{query_hash}"

    def _get_memory_filepath(self, query_id: str) -> str:
        """
        Get the full filepath for a query's memory file.

        Args:
            query_id: The unique query identifier

        Returns:
            Full path to the memory JSON file
        """
        return os.path.join(self.memory_dir, f"{query_id}.json")

    def save_requirement_evidence(
        self,
        original_query: str,
        requirement_id: str,
        requirement_question: str,
        search_query: str,
        evidences: List[Dict[str, Any]]
    ) -> str:
        """
        Save evidence for a single requirement.

        Args:
            original_query: The original user query
            requirement_id: ID of the requirement (e.g., "req1", "req2")
            requirement_question: The question for this requirement
            search_query: The search query used to retrieve documents
            evidences: List of evidence dictionaries

        Returns:
            The query_id of the saved memory
        """
        query_id = self._generate_query_id(original_query)
        filepath = self._get_memory_filepath(query_id)

        # Load existing memory or create new structure
        memory_data = self._load_or_create_memory(filepath, original_query, query_id)

        # Create evidence entry
        evidence_entry = {
            "requirement_id": requirement_id,
            "requirement_question": requirement_question,
            "search_query": search_query,
            "evidences": evidences,
            "evidence_count": len(evidences),
            "timestamp": datetime.now().isoformat()
        }

        # Find existing requirement or append new one
        existing_req_index = None
        for i, req in enumerate(memory_data.get("requirements", [])):
            if req.get("requirement_id") == requirement_id:
                existing_req_index = i
                break

        if existing_req_index is not None:
            # Update existing requirement
            memory_data["requirements"][existing_req_index] = evidence_entry
            print(f"🔄 Updating evidence for requirement '{requirement_id}' in memory: {filepath}")
        else:
            # Append new requirement
            memory_data["requirements"].append(evidence_entry)
            print(f"💾 Saving evidence for requirement '{requirement_id}' to memory: {filepath}")

        # Update metadata
        memory_data["last_updated"] = datetime.now().isoformat()
        memory_data["total_requirements"] = len(memory_data["requirements"])
        memory_data["total_evidences"] = sum(
            req.get("evidence_count", 0) for req in memory_data["requirements"]
        )

        # Save to file
        self._save_memory(filepath, memory_data)

        return query_id

    def _load_or_create_memory(self, filepath: str, original_query: str, query_id: str) -> Dict[str, Any]:
        """
        Load existing memory file or create a new memory structure.

        Args:
            filepath: Path to the memory file
            original_query: The original user query
            query_id: Unique ID for the query

        Returns:
            Memory data dictionary
        """
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load existing memory file: {e}")
                print("📝 Creating new memory structure.")

        # Create new memory structure
        return {
            "query_id": query_id,
            "original_query": original_query,
            "requirements": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_requirements": 0,
            "total_evidences": 0
        }

    def _save_memory(self, filepath: str, memory_data: Dict[str, Any]):
        """
        Save memory data to file.

        Args:
            filepath: Path to save the memory file
            memory_data: Memory data dictionary to save
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Memory saved successfully: {filepath}")
        except Exception as e:
            print(f"❌ Error saving memory file: {e}")

    def load_memory(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Load memory for a specific query.

        Args:
            query_id: The unique query identifier

        Returns:
            Memory data dictionary, or None if not found
        """
        filepath = self._get_memory_filepath(query_id)

        if not os.path.exists(filepath):
            print(f"⚠️  Memory not found for query_id: {query_id}")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            print(f"📖 Loaded memory from: {filepath}")
            return memory_data
        except Exception as e:
            print(f"❌ Error loading memory file: {e}")
            return None

    def get_requirement_evidence(self, original_query: str, requirement_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve evidence for a specific requirement.

        Args:
            original_query: The original user query
            requirement_id: The requirement ID to retrieve

        Returns:
            List of evidence dictionaries, or None if not found
        """
        query_id = self._generate_query_id(original_query)
        memory_data = self.load_memory(query_id)

        if not memory_data:
            return None

        for req in memory_data.get("requirements", []):
            if req.get("requirement_id") == requirement_id:
                print(f"✅ Found evidence for requirement '{requirement_id}'")
                return req.get("evidences", [])

        print(f"⚠️  No evidence found for requirement '{requirement_id}'")
        return None

    def list_all_memories(self) -> List[str]:
        """
        List all query IDs in memory.

        Returns:
            List of query IDs
        """
        if not os.path.exists(self.memory_dir):
            return []

        query_ids = []
        for filename in os.listdir(self.memory_dir):
            if filename.endswith('.json'):
                query_id = filename[:-5]  # Remove .json extension
                query_ids.append(query_id)

        return query_ids

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories.

        Returns:
            Dictionary with memory statistics
        """
        query_ids = self.list_all_memories()

        total_requirements = 0
        total_evidences = 0

        for query_id in query_ids:
            memory_data = self.load_memory(query_id)
            if memory_data:
                total_requirements += memory_data.get("total_requirements", 0)
                total_evidences += memory_data.get("total_evidences", 0)

        return {
            "total_queries": len(query_ids),
            "total_requirements": total_requirements,
            "total_evidences": total_evidences,
            "memory_directory": self.memory_dir
        }

    def get_full_execution_history(self, original_query: str, collected_facts: dict, pending_requirements: list) -> Dict[str, Any]:
        """
        Retrieve the complete execution history including requirements, their status, answers, and evidence.
        This is used for failure attribution and replanning.

        Args:
            original_query: The original user query
            collected_facts: Dictionary containing all facts collected so far
            pending_requirements: List of requirements that are still pending

        Returns:
            Dictionary with complete execution history:
            {
                "original_query": "...",
                "completed_requirements": [
                    {
                        "requirement_id": "...",
                        "question": "...",
                        "answer": "...",  # Extracted from collected_facts
                        "evidence_count": N,
                        "evidences": [...]  # From memory if available
                    }
                ],
                "pending_requirements": [...],
                "execution_summary": "..."
            }
        """
        query_id = self._generate_query_id(original_query)
        memory_data = self.load_memory(query_id)

        # Build completed requirements from collected_facts
        completed_requirements = []

        # Group facts by requirement_id
        facts_by_req = {}
        for fact in collected_facts.get("reasoned_facts", []):
            req_id = fact.get("fulfills_requirement_id")
            if req_id:
                if req_id not in facts_by_req:
                    facts_by_req[req_id] = []
                facts_by_req[req_id].append(fact)

        # Build completed requirements list
        for req_id, facts in facts_by_req.items():
            # Get the most direct answer
            direct_facts = [f for f in facts if f.get("fulfillment_level") == "DIRECT_ANSWER"]
            if direct_facts:
                answer_fact = direct_facts[0]
            else:
                answer_fact = facts[0]

            # Try to get evidence from memory
            evidences = []
            if memory_data:
                for req in memory_data.get("requirements", []):
                    if req.get("requirement_id") == req_id:
                        evidences = req.get("evidences", [])
                        break

            completed_requirements.append({
                "requirement_id": req_id,
                "question": answer_fact.get("requirement", "Unknown"),
                "answer": answer_fact.get("statement", ""),
                "reasoning": answer_fact.get("reasoning", ""),
                "fulfillment_level": answer_fact.get("fulfillment_level", "UNKNOWN"),
                "evidence_count": len(evidences),
                "evidences": evidences[:10]  # Include top 10 evidences for context
            })

        # Build execution summary
        completed_count = len(completed_requirements)
        pending_count = len(pending_requirements)
        execution_summary = f"Completed {completed_count} requirements, {pending_count} requirements pending. {len(collected_facts.get('reasoned_facts', []))} facts collected."

        return {
            "original_query": original_query,
            "completed_requirements": completed_requirements,
            "pending_requirements": pending_requirements,
            "total_facts_collected": len(collected_facts.get("reasoned_facts", [])),
            "execution_summary": execution_summary
        }


# Global instance for convenience
_global_memory_manager = None
_global_memory_manager_lock = Lock()


def get_memory_manager(
    memory_dir: str = "data/evidence_memory",
    run_timestamp: Optional[str] = None
) -> EvidenceMemory:
    """
    Get the global memory manager instance (singleton pattern).

    Args:
        memory_dir: Root directory for memory storage
        run_timestamp: Optional timestamp folder for this pipeline run

    Returns:
        EvidenceMemory instance
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        with _global_memory_manager_lock:
            if _global_memory_manager is None:
                _global_memory_manager = EvidenceMemory(memory_dir, run_timestamp=run_timestamp)
    return _global_memory_manager

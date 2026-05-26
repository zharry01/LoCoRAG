"""
Pipeline Logger - Records all pipeline stage information for each query.

Saves a detailed txt file per query under:
    ./data/save_pipeline/<session_timestamp>/<query_id>.txt
"""
import os
import json
import hashlib
import threading
from datetime import datetime
from contextvars import ContextVar

# Session timestamp - set once per Python process, shared by all queries
_SESSION_TIMESTAMP: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Context variable for accessing the logger from anywhere in the call chain
pipeline_logger_ctx: ContextVar = ContextVar('pipeline_logger_ctx', default=None)


class PipelineLogger:
    """Thread-safe accumulator for pipeline stage data."""

    def __init__(self, query: str, query_id: str | None = None):
        self.query = query
        self.query_id = query_id or hashlib.md5(query.encode('utf-8')).hexdigest()[:12]
        self._sections: list[str] = []
        self._lock = threading.Lock()

    def log(self, title: str, content):
        ts = datetime.now().strftime("%H:%M:%S")
        sep = "=" * 60
        if isinstance(content, (dict, list)):
            text = json.dumps(content, indent=2, ensure_ascii=False)
        else:
            text = str(content)
        entry = f"\n{sep}\n[{ts}] {title}\n{sep}\n{text}\n"
        with self._lock:
            self._sections.append(entry)

    def save(self, save_dir: str = "data/save_pipeline"):
        dir_path = os.path.join(save_dir, _SESSION_TIMESTAMP)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"{self.query_id}.txt")
        header = (
            f"Pipeline Log\n"
            f"Session: {_SESSION_TIMESTAMP}\n"
            f"Query: {self.query}\n"
            f"Query ID: {self.query_id}\n"
            f"Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 60}\n"
        )
        with self._lock:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(header)
                for section in self._sections:
                    f.write(section)
        return filepath


def get_pipeline_logger() -> "PipelineLogger | None":
    """Get the current pipeline logger from context."""
    return pipeline_logger_ctx.get()

import os
import asyncio
import traceback
import sys
import random
sys.path.insert(0, 'src')

from typing import List, Dict
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from concurrent.futures import ThreadPoolExecutor

from search.e5_searcher import E5Searcher
from logger_config import logger


async def search(request: Request):
    payload = await request.json()
    query: str = payload['query']
    # Get 'k' from payload, default to TOP_K env var or 5
    top_k_env = int(os.getenv('TOP_K', 5))
    k: int = payload.get('k', top_k_env)

    response_q = asyncio.Queue()
    # Pass query and k to the model_queue
    await request.app.model_queue.put(((query, k), response_q))
    output = await response_q.get()
    return JSONResponse(output)


async def server_loop(q):
    searcher: E5Searcher = E5Searcher(
        index_dir='/path/to/corag_kilt_corpus_embeddings/',
        model_name_or_path='/path/to/e5-large-v2',
        verbose=True # Ensure E5Searcher returns document content
    )
    # Default k from environment, primarily for warmup or if not specified in request
    default_top_k = int(os.getenv('TOP_K', 5))
    logger.info(f'E5Searcher initialized, ready to serve requests, default_top_k={default_top_k}, verbose=True')
    
    # Warmup the server
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        app.executor,
        searcher.batch_search, 
        [f'test query {random.random()} {i}' for i in range(min(64, default_top_k * 2))], 
        default_top_k
    )

    while True:
        try:
            (request_data, response_q) = await q.get()
            query, k_val = request_data

            logger.info(f"Processing query: '{query}' with k={k_val}")

            loop = asyncio.get_event_loop()
            results_list: List[List[Dict]] = await loop.run_in_executor(
                app.executor,
                searcher.batch_search, 
                [query],
                k_val
            )
            
            await response_q.put(results_list[0] if results_list else [])

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            logger.error(traceback.format_exc())
            if 'response_q' in locals() and response_q:
                try:
                    await response_q.put({"error": str(e)})
                except Exception as e_resp:
                    logger.error(f"Error sending error to response_q: {e_resp}")


app = Starlette(
    routes=[
        Route("/", search, methods=["POST"]),
    ],
)


@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    app.executor = ThreadPoolExecutor(max_workers= 10) 
    asyncio.create_task(server_loop(q))

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app, 'executor') and app.executor:
        app.executor.shutdown(wait=True)

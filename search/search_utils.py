import requests
from typing import List, Dict
from logger_config import logger
import config


def search_by_http(query: str, k: int, host: str = None, port: int = None, endpoint: str = None) -> List[Dict]:
    """
    Retrieve documents from the configured external search API.

    Args:
        query: Search query string.
        k: Number of documents to return.
        host: Search service host. Uses the configured default when omitted.
        port: Search service port. Uses the configured default when omitted.
        endpoint: API endpoint path. Uses the configured default when omitted.

    Returns:
        List[Dict]: Retrieved results containing document content and metadata.
    """
    # Use configured defaults when explicit connection settings are not provided.
    if host is None:
        host = config.SEARCH_SERVICE_HOST
    if port is None:
        port = config.SEARCH_SERVICE_PORT
    if endpoint is None:
        endpoint = config.SEARCH_SERVICE_ENDPOINT

    # Build the full API URL.
    url = f"http://{host}:{port}{endpoint}"

    # Match the expected search API payload shape.
    payload = {
        'queries': [query],
        'topk': k,
        'return_scores': True
    }

    logger.info(f"Sending search request to: {url}")
    logger.info(f"Query: '{query}', Top_k: {k}")

    try:
        response = requests.post(url, json=payload, timeout=660)
        response.raise_for_status()

        # Parse the response data.
        response_data = response.json()
        
        if isinstance(response_data, dict) and 'result' in response_data:
            # Extract the first query's results since we only sent one query
            results = response_data['result'][0] if response_data['result'] else []
            logger.info(f"Search successful, received {len(results)} results")
            return results
        elif isinstance(response_data, list):
            logger.info(f"Search successful, received {len(response_data)} results")
            return response_data
        elif isinstance(response_data, dict) and 'results' in response_data:
            return response_data['results']
        elif isinstance(response_data, dict) and 'documents' in response_data:
            return response_data['documents']
        else:
            logger.warning(f"Unexpected response format: {type(response_data)}")
            return response_data if isinstance(response_data, list) else [response_data]

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Error Connecting: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout Error: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"An error occurred: {req_err}")
    except Exception as e:
        logger.error(f"Failed to get a response. Status code: {response.status_code if 'response' in locals() else 'N/A'}, Error: {e}")
    return []

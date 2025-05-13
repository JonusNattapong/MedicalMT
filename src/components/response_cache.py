# -*- coding: utf-8 -*-
"""
Response caching utilities for MedMT dataset generation
to improve efficiency when running multiple times
"""
import os
import json
import hashlib
from typing import Dict, Any, Optional

CACHE_DIR = "cache/api_responses"

def get_cache_key(model: str, messages: list, params: Dict[str, Any]) -> str:
    """Generate a cache key based on the API request parameters"""
    # Create a dictionary containing all relevant parameters
    cache_dict = {
        "model": model,
        "messages": messages,
        # Include only relevant parameters
        "params": {
            k: v for k, v in params.items() 
            if k in ["max_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty"]
        }
    }
    
    # Convert to a stable JSON string and hash it
    cache_json = json.dumps(cache_dict, sort_keys=True)
    return hashlib.md5(cache_json.encode()).hexdigest()

def save_to_cache(cache_key: str, response: Dict[str, Any]) -> None:
    """Save API response to cache"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

def load_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Load API response from cache if it exists"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] Error loading cache file {cache_file}: {e}")
        return None

def cached_api_call(client, model: str, messages: list, **params) -> Dict[str, Any]:
    """Make an API call with caching support"""
    cache_key = get_cache_key(model, messages, params)
    cached_response = load_from_cache(cache_key)
    
    if cached_response:
        print(f"[INFO] Using cached response for {cache_key[:8]}...")
        # Create a structure similar to what the actual API would return
        return cached_response
    
    # Make the actual API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **params
    )
    
    # Convert response to a JSON-serializable dict
    response_dict = {
        "id": response.id,
        "object": response.object,
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "index": choice.index,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content
                },
                "finish_reason": choice.finish_reason
            }
            for choice in response.choices
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }
    
    # Save to cache
    save_to_cache(cache_key, response_dict)
    
    return response_dict

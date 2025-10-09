"""
NLP utilities for DS776 Deep Learning course.
OpenRouter-based text and JSON generation with cost tracking.
"""

import os
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from openai import OpenAI

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Global Configuration and Caching
# ============================================================================

# Session-level caches (reset on import/restart)
_PRICING_CACHE: Dict[str, Tuple[float, float]] = {}  # model_id -> (input_$/token, output_$/token)
_MODEL_MAP: Dict[str, str] = {}  # short_name -> full_model_id
_CONFIG: Dict[str, Any] = {}  # From openrouter_models.json
_CLIENT: Optional[OpenAI] = None


def _init_client():
    """Initialize OpenRouter client (lazy initialization)."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Set it in your environment or api_keys.env file."
            )
        _CLIENT = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "http://localhost"),
                "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "DS776 Deep Learning"),
            },
        )
    return _CLIENT


def _load_model_config():
    """
    Load model configuration from openrouter_models.json.

    Returns:
        Tuple of (model_map, config) where model_map contains full model metadata
    """
    global _MODEL_MAP, _CONFIG
    if not _MODEL_MAP:
        config_path = Path(__file__).parent / "openrouter_models.json"
        try:
            with open(config_path, 'r') as f:
                _CONFIG = json.load(f)
                models = _CONFIG.get('models', {})

                # Handle both old (string) and new (dict) formats
                _MODEL_MAP = {}
                for short_name, model_data in models.items():
                    if isinstance(model_data, str):
                        # Old format: "short_name": "model_id"
                        _MODEL_MAP[short_name] = {
                            'id': model_data,
                            'provider': model_data.split('/')[0] if '/' in model_data else 'unknown',
                            'json_support': {
                                'json_object': True,
                                'json_schema': False,
                                'strict_schema': False
                            }
                        }
                    else:
                        # New format: "short_name": {...metadata...}
                        _MODEL_MAP[short_name] = model_data

        except Exception as e:
            print(f"⚠️  Could not load openrouter_models.json: {e}")
            _CONFIG = {
                'default_model': 'gemini-flash-lite',
                'cost_warning_thresholds': [0.5, 0.75, 0.9],
                'student_credit_limit': 15.0
            }
            _MODEL_MAP = {
                'gpt-4o-mini': {
                    'id': 'openai/gpt-4o-mini',
                    'provider': 'openai',
                    'json_support': {'json_object': True, 'json_schema': True, 'strict_schema': False}
                },
                'gemini-flash-lite': {
                    'id': 'google/gemini-2.5-flash-lite',
                    'provider': 'google',
                    'json_support': {'json_object': True, 'json_schema': True, 'strict_schema': True}
                }
            }
    return _MODEL_MAP, _CONFIG


def get_model_metadata(model: str):
    """
    Get full metadata for a model.

    Args:
        model: Short name or full model ID

    Returns:
        Dictionary with model metadata (id, provider, json_support)
    """
    model_map, _ = _load_model_config()

    # If it's a short name, get metadata
    if model in model_map:
        return model_map[model]

    # If it's a full ID, try to find by ID
    for metadata in model_map.values():
        if metadata.get('id') == model:
            return metadata

    # Unknown model - return minimal metadata
    return {
        'id': model,
        'provider': model.split('/')[0] if '/' in model else 'unknown',
        'json_support': {
            'json_object': True,  # Assume basic JSON support
            'json_schema': False,
            'strict_schema': False
        }
    }


def resolve_model_name(model: str) -> str:
    """
    Resolve short name to full OpenRouter model ID.

    Args:
        model: Short name (e.g., 'gpt-4o-mini') or full ID (e.g., 'openai/gpt-4o-mini')

    Returns:
        Full OpenRouter model ID
    """
    metadata = get_model_metadata(model)
    return metadata['id']


# ============================================================================
# Pricing Functions (Session-Cached)
# ============================================================================

def _fetch_openrouter_pricing() -> Dict[str, Tuple[float, float]]:
    """
    Fetch pricing from OpenRouter API.
    Returns dict of model_id -> (input_$/token, output_$/token)
    """
    try:
        client = _init_client()
        models = client.models.list()
        prices = {}

        for m in models.data:
            mid = getattr(m, 'id', None) or getattr(m, 'name', None)
            pr = getattr(m, 'pricing', None)

            # Handle dict or object pricing
            if not pr:
                try:
                    d = m.to_dict() if hasattr(m, 'to_dict') else getattr(m, '__dict__', {})
                    pr = d.get('pricing') or d.get('top_provider', {}).get('pricing', {})
                except Exception:
                    pr = {}

            def _to_float(x):
                try:
                    return float(x) if x is not None else None
                except Exception:
                    return None

            # OpenRouter returns per-token USD prices
            if isinstance(pr, dict):
                pin = _to_float(pr.get('input') or pr.get('prompt'))
                pout = _to_float(pr.get('output') or pr.get('completion'))

                if mid and pin is not None and pout is not None:
                    prices[mid] = (pin, pout)

        return prices if prices else {}

    except Exception as e:
        print(f"⚠️  Could not fetch OpenRouter pricing: {e}")
        return {}


def get_openrouter_credit() -> Optional[float]:
    """
    Fetch account credit from OpenRouter API.

    Returns:
        float: Available credit in USD, or None if unavailable
    """
    try:
        import requests
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            return None

        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            # OpenRouter returns credit info in the 'data' field
            credit = data.get('data', {}).get('limit_remaining')
            if credit is not None:
                return float(credit)

        return None

    except Exception as e:
        # Silently fail - credit check is not critical
        return None


def get_model_price(model_id: str) -> Tuple[float, float]:
    """
    Get pricing for a model (cached per session).

    Args:
        model_id: Full OpenRouter model ID

    Returns:
        (input_$/token, output_$/token) or (0.0, 0.0) if unavailable
    """
    global _PRICING_CACHE

    # Check cache first
    if model_id in _PRICING_CACHE:
        return _PRICING_CACHE[model_id]

    # Check with variant stripped (e.g., 'model:free' -> 'model')
    base_id = model_id.split(':')[0]
    if base_id in _PRICING_CACHE:
        return _PRICING_CACHE[base_id]

    # Fetch all pricing if cache is empty
    if not _PRICING_CACHE:
        _PRICING_CACHE = _fetch_openrouter_pricing()

    # Try exact match
    if model_id in _PRICING_CACHE:
        return _PRICING_CACHE[model_id]

    # Try base match
    if base_id in _PRICING_CACHE:
        return _PRICING_CACHE[base_id]

    # Try fuzzy match (for models with variants)
    for cached_id, price in _PRICING_CACHE.items():
        cached_base = cached_id.split(':')[0]
        if base_id == cached_base or base_id.startswith(cached_base):
            return price

    # Return zero if no pricing found
    print(f"⚠️  No pricing found for {model_id}, assuming free")
    return (0.0, 0.0)


def show_pricing_table():
    """Display pricing table for available models."""
    model_map, _ = _load_model_config()

    print("OpenRouter Model Pricing (USD)")
    print("=" * 110)
    print(f"{'Short Name':<20}{'Model ID':<40}{'In/1K':>12}{'Out/1K':>12}{'In/1M':>11}{'Out/1M':>11}")
    print("-" * 110)

    for short, mid in sorted(model_map.items()):
        inp_tok, out_tok = get_model_price(mid)
        inp_1k = inp_tok * 1000
        out_1k = out_tok * 1000
        inp_1m = inp_tok * 1_000_000
        out_1m = out_tok * 1_000_000

        print(f"{short:<20}{mid:<40}${inp_1k:>11.4f}${out_1k:>11.4f}${inp_1m:>10.2f}${out_1m:>10.2f}")

    print("\nNote: Prices fetched from OpenRouter API. Add new models to openrouter_models.json")


def llm_get_credits() -> Dict[str, float]:
    """
    Get OpenRouter credit information.

    Returns:
        Dictionary with 'limit' (baseline credit), 'usage' (total spent), and 'remaining'

    Example:
        >>> credits = llm_get_credits()
        >>> print(f"Remaining: ${credits['limit'] - credits['usage']:.2f}")
    """
    data = _load_cost_tracker()
    limit = data.get("baseline_credit", 15.0)
    usage = data.get("total_spend", 0.0)

    return {
        'limit': limit,
        'usage': usage,
        'remaining': limit - usage
    }


# ============================================================================
# Cost Tracking (Persistent)
# ============================================================================

def _get_cost_tracker_path() -> Path:
    """Get path to cost tracking file in ~/home_workspace/"""
    return Path.home() / "home_workspace" / "openrouter_costs.json"


def _load_cost_tracker() -> Dict[str, Any]:
    """Load cost tracking data from file."""
    path = _get_cost_tracker_path()

    if not path.exists():
        # Initialize new tracker
        _, config = _load_model_config()
        return {
            "initialized": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "baseline_credit": config.get('student_credit_limit', 15.0),
            "total_spend": 0.0,
            "by_model": {},
            "by_provider": {},
            "call_count": 0,
            "warnings_shown": {"50%": False, "75%": False, "90%": False},
            "history": []
        }

    try:
        with open(path, 'r') as f:
            data = json.load(f)
            # Ensure all required fields exist
            if "warnings_shown" not in data:
                data["warnings_shown"] = {"50%": False, "75%": False, "90%": False}
            return data
    except Exception as e:
        print(f"⚠️  Error loading cost tracker: {e}")
        return _load_cost_tracker()  # Return fresh tracker


def update_openrouter_credit() -> Optional[float]:
    """
    Fetch current OpenRouter credit and update the cost tracker.

    Returns:
        float: Current credit amount, or None if unavailable
    """
    credit = get_openrouter_credit()
    if credit is not None:
        # Update the cost tracker with actual credit
        tracker = _load_cost_tracker()
        tracker['baseline_credit'] = credit
        tracker['actual_credit'] = credit  # Store for reference
        tracker['credit_last_checked'] = datetime.now().isoformat()
        _save_cost_tracker(tracker)
    return credit


def _save_cost_tracker(data: Dict[str, Any]):
    """Save cost tracking data to file."""
    path = _get_cost_tracker_path()

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Update timestamp
    data["last_updated"] = datetime.now().isoformat()

    try:
        # Create backup before writing
        if path.exists():
            backup_path = path.with_suffix('.json.bak')
            path.rename(backup_path)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        # Remove backup after successful write
        if path.with_suffix('.json.bak').exists():
            path.with_suffix('.json.bak').unlink()

    except Exception as e:
        print(f"⚠️  Error saving cost tracker: {e}")
        # Restore backup if write failed
        backup_path = path.with_suffix('.json.bak')
        if backup_path.exists():
            backup_path.rename(path)


def _update_spend(model_id: str, prompt_tokens: int, completion_tokens: int, cost: float):
    """Update cost tracking after API call."""
    data = _load_cost_tracker()

    # Update totals
    data["total_spend"] += cost
    data["call_count"] += 1

    # Update by model
    if model_id not in data["by_model"]:
        data["by_model"][model_id] = {"spend": 0.0, "calls": 0}
    data["by_model"][model_id]["spend"] += cost
    data["by_model"][model_id]["calls"] += 1

    # Update by provider
    provider = model_id.split('/')[0] if '/' in model_id else model_id
    if provider not in data["by_provider"]:
        data["by_provider"][provider] = 0.0
    data["by_provider"][provider] += cost

    # Add to history (keep last 100)
    data["history"].append({
        "timestamp": datetime.now().isoformat(),
        "model": model_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": cost
    })
    if len(data["history"]) > 100:
        data["history"] = data["history"][-100:]

    _save_cost_tracker(data)

    # Check and show warnings
    _check_cost_warnings(data)


def _check_cost_warnings(data: Dict[str, Any]):
    """Check if cost warnings should be shown."""
    total_spend = data["total_spend"]
    baseline = data.get("baseline_credit", 15.0)
    _, config = _load_model_config()
    thresholds = config.get('cost_warning_thresholds', [0.5, 0.75, 0.9])

    for threshold in thresholds:
        threshold_label = f"{int(threshold*100)}%"
        if total_spend >= baseline * threshold and not data["warnings_shown"].get(threshold_label, False):
            remaining = baseline - total_spend
            print(f"\n⚠️  COST WARNING: You have spent ${total_spend:.4f} ({threshold_label} of ${baseline:.2f} credit)")
            print(f"   Remaining credit: ${remaining:.4f}\n")

            # Mark warning as shown
            data["warnings_shown"][threshold_label] = True
            _save_cost_tracker(data)


def show_cost_summary(detailed: bool = False):
    """
    Display current cost summary.

    Args:
        detailed: If True, show per-model breakdown
    """
    data = _load_cost_tracker()

    total_spend = data["total_spend"]
    baseline = data.get("baseline_credit", 15.0)
    remaining = baseline - total_spend
    percent_used = (total_spend / baseline * 100) if baseline > 0 else 0

    print("\n" + "=" * 60)
    print("OpenRouter Cost Summary")
    print("=" * 60)
    print(f"Total Spend:        ${total_spend:.4f}")
    print(f"Baseline Credit:    ${baseline:.2f}")
    print(f"Remaining:          ${remaining:.4f}")
    print(f"Percent Used:       {percent_used:.1f}%")
    print(f"Total API Calls:    {data['call_count']}")

    if detailed and data["by_provider"]:
        print("\n" + "-" * 60)
        print("By Provider:")
        for provider, spend in sorted(data["by_provider"].items(), key=lambda x: -x[1]):
            print(f"  {provider:<30} ${spend:.4f}")

        print("\n" + "-" * 60)
        print("By Model:")
        for model, info in sorted(data["by_model"].items(), key=lambda x: -x[1]["spend"]):
            print(f"  {model:<40} ${info['spend']:.4f} ({info['calls']} calls)")

    print("=" * 60 + "\n")


def reset_cost_tracker():
    """Reset cost tracking (for new semester/testing)."""
    path = _get_cost_tracker_path()
    if path.exists():
        # Archive old tracker
        archive_path = path.with_name(f"openrouter_costs_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        path.rename(archive_path)
        print(f"✅ Cost tracker archived to {archive_path.name}")

    # Create fresh tracker
    data = _load_cost_tracker()
    _save_cost_tracker(data)
    print("✅ Cost tracker reset")


# ============================================================================
# JSON Helpers
# ============================================================================

def _parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from response content, handling markdown code blocks."""
    if not content:
        return None

    content = content.strip()

    # Remove markdown code blocks
    if content.startswith('```'):
        content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


# ============================================================================
# Main Generation Function
# ============================================================================

def llm_generate(
    model_name: str,
    prompts: Union[str, List[str]],
    mode: str = 'text',
    user_schema: Optional[dict] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    llm_provider: str = 'openrouter',
    print_cost: bool = False,
    track_cost: bool = True,
    **kwargs
) -> Union[str, dict, List[str], List[dict]]:
    """
    Generate text or JSON from LLM.

    Args:
        model_name: Model name (short name or full OpenRouter ID)
        prompts: Single prompt or list of prompts
        mode: 'text' or 'json'
        user_schema: Optional JSON schema for structured output (mode='json' only)
        system_prompt: Optional system message
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        llm_provider: 'openrouter' (default) or 'huggingface' (local models, not priority)
        print_cost: Print cost after generation
        track_cost: Update persistent cost tracking
        **kwargs: Additional parameters (for future extensions)

    Returns:
        - mode='text': str or List[str]
        - mode='json': dict or List[dict]

    Examples:
        >>> # Text generation
        >>> response = llm_generate('gpt-4o-mini', 'What is deep learning?')

        >>> # Batch text generation
        >>> responses = llm_generate('gemini-flash-lite', ['Question 1', 'Question 2'])

        >>> # JSON generation with schema
        >>> schema = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}
        >>> response = llm_generate('gpt-4o', 'Answer in JSON', mode='json', user_schema=schema)
    """
    if llm_provider != 'openrouter':
        raise NotImplementedError(
            f"Provider '{llm_provider}' not yet supported. Use 'openrouter' or access HuggingFace models directly."
        )

    # Resolve model name
    model_id = resolve_model_name(model_name)

    # Initialize client
    client = _init_client()

    # Handle batch vs single prompt
    is_batch = isinstance(prompts, list)
    prompt_list = prompts if is_batch else [prompts]

    # Get pricing
    inp_price, out_price = get_model_price(model_id)

    # Generate responses
    responses = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    # Prepare system prompt
    if system_prompt is None:
        if mode == 'json':
            system_prompt = "Generate valid JSON strictly following the provided schema." if user_schema else "Always respond with valid JSON."
        else:
            system_prompt = "You are a helpful AI assistant."

    for prompt in prompt_list:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Build API call parameters
        api_params = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Add response format for JSON mode (using model capabilities)
        if mode == 'json':
            metadata = get_model_metadata(model_name)
            json_support = metadata.get('json_support', {})

            if user_schema:
                # Check if model supports json_schema
                if json_support.get('json_schema', False):
                    # Model supports schemas - use them
                    strict = json_support.get('strict_schema', False)
                    api_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "schema": user_schema,
                            "strict": strict
                        }
                    }
                else:
                    # Model doesn't support schemas - warn and fall back
                    if print_cost:
                        print(f"⚠️  Model '{model_name}' doesn't support json_schema - using basic json_object mode")
                    api_params["response_format"] = {"type": "json_object"}
            else:
                # Basic JSON mode
                api_params["response_format"] = {"type": "json_object"}

        # Make API call
        try:
            response = client.chat.completions.create(**api_params)

            # Extract response
            content = response.choices[0].message.content

            # Parse based on mode
            if mode == 'json':
                parsed = _parse_json_response(content)
                responses.append(parsed if parsed is not None else {"error": "Failed to parse JSON", "raw": content})
            else:
                responses.append(content.strip() if content else "")

            # Track usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            # Calculate cost
            cost = (prompt_tokens * inp_price) + (completion_tokens * out_price)
            total_cost += cost

        except Exception as e:
            error_msg = f"API Error: {str(e)}"
            if mode == 'json':
                responses.append({"error": error_msg})
            else:
                responses.append(error_msg)

    # Update cost tracking
    if track_cost and total_cost > 0:
        _update_spend(model_id, total_prompt_tokens, total_completion_tokens, total_cost)

    # Print cost if requested
    if print_cost:
        print(f"💰 Cost: ${total_cost:.6f} | Tokens: {total_prompt_tokens} in / {total_completion_tokens} out | Model: {model_id}")

    # Return single result or list
    return responses if is_batch else responses[0]


# ============================================================================
# Convenience Functions
# ============================================================================

def llm_list_models(verbose: bool = True, json_schema: bool = None, strict_schema: bool = None):
    """
    List available models from openrouter_models.json.

    Args:
        verbose: If True, print model details
        json_schema: If True, only show models that support json_schema
        strict_schema: If True, only show models that support strict_schema

    Returns:
        List of (index, short_name) tuples
    """
    model_map, config = _load_model_config()

    # Filter by JSON capabilities if requested
    filtered_models = {}
    for short_name, metadata in model_map.items():
        json_support = metadata.get('json_support', {})

        if json_schema and not json_support.get('json_schema', False):
            continue
        if strict_schema and not json_support.get('strict_schema', False):
            continue

        filtered_models[short_name] = metadata

    if verbose:
        filter_note = ""
        if json_schema:
            filter_note = " (json_schema support)"
        elif strict_schema:
            filter_note = " (strict_schema support)"

        print(f"Available OpenRouter Models{filter_note}:")
        print("-" * 90)
        for i, (short, metadata) in enumerate(sorted(filtered_models.items())):
            model_id = metadata['id']
            json_support = metadata.get('json_support', {})

            inp_tok, out_tok = get_model_price(model_id)
            print(f"{i+1}. {short:<20} -> {model_id:<40}", end="")

            # Show JSON capabilities
            caps = []
            if json_support.get('json_object'): caps.append('json')
            if json_support.get('json_schema'): caps.append('schema')
            if json_support.get('strict_schema'): caps.append('strict')
            if caps:
                print(f" [{', '.join(caps)}]")
            else:
                print()

            if inp_tok > 0 or out_tok > 0:
                print(f"   ${inp_tok*1000:.4f}/1K in, ${out_tok*1000:.4f}/1K out")

        print("-" * 90)
        print(f"Default model: {config.get('default_model', 'gemini-flash-lite')}")
        if not filter_note:
            print("\nJSON Capabilities: [json]=basic JSON, [schema]=user schemas, [strict]=strict validation")
        print("\nYou can also use any OpenRouter model by its full ID (e.g., 'openai/gpt-4o')")

    return list(enumerate(sorted(filtered_models.keys()), 1))


def display_markdown(text: str):
    """Display text as markdown in Jupyter."""
    try:
        from IPython.display import display, Markdown
        display(Markdown(text))
    except ImportError:
        print(text)


# ============================================================================
# Backward Compatibility (from nlp_orig.py)
# ============================================================================

def llm_configure(model_str: str, **kwargs):
    """
    DEPRECATED: Use llm_generate() directly with model names.

    This function is kept for backward compatibility but will be removed in a future version.
    """
    warnings.warn(
        "llm_configure() is deprecated. Use llm_generate(model_name, ...) directly.",
        DeprecationWarning,
        stacklevel=2
    )
    # Return a simple dict that can be passed to old code
    return {"model_str": model_str, **kwargs}


def generate_text(prompts, model_name='gemini-flash-lite', **kwargs):
    """
    DEPRECATED: Use llm_generate() directly.

    This function is kept for backward compatibility.
    """
    warnings.warn(
        "generate_text() is deprecated. Use llm_generate(model_name, prompts, mode='text', ...) directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return llm_generate(model_name, prompts, mode='text', **kwargs)


# ============================================================================
# Pipeline Utility Functions (moved from nlp_orig.py)
# ============================================================================

def clear_pipeline(pipe, verbosity=0):
    """
    Clears a Hugging Face pipeline and frees CUDA memory.

    Args:
        pipe: Hugging Face pipeline object to clear
        verbosity: 0 (minimal output), 1+ (detailed memory info)
    """
    import gc
    import torch

    if hasattr(pipe, "model") and next(pipe.model.parameters()).is_cuda:
        initial_allocated = torch.cuda.memory_allocated() / 1e6
        initial_reserved = torch.cuda.memory_reserved() / 1e6

        if verbosity > 0:
            print(f"🔍 Before unloading: {initial_allocated:.2f} MB allocated, {initial_reserved:.2f} MB reserved.")

        try:
            pipe.model.to("cpu")
            for param in pipe.model.parameters():
                param.data = param.data.cpu()
        except Exception as e:
            if verbosity > 0:
                print(f"⚠️ Error moving model to CPU: {e}")

        del pipe.model
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        final_allocated = torch.cuda.memory_allocated() / 1e6
        final_reserved = torch.cuda.memory_reserved() / 1e6

        if verbosity > 0:
            print(f"✅ Pipeline cleared. Freed {initial_allocated - final_allocated:.2f} MB allocated, "
                  f"{initial_reserved - final_reserved:.2f} MB reserved.")
    else:
        if verbosity > 0:
            print("ℹ️ Pipeline already on CPU. Performing standard cleanup.")
        del pipe
        gc.collect()

    if verbosity > 0:
        print("🗑️ Cleanup complete.")
    elif verbosity == 0:
        print("✅ Pipeline cleared.")


def print_pipeline_info(pipe):
    """
    Print information about a Hugging Face pipeline.

    Args:
        pipe: Hugging Face pipeline object
    """
    model = pipe.model
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.name_or_path}, Size: {model_size:,} parameters")



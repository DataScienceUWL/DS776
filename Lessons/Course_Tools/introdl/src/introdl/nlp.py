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
import requests
from tqdm.auto import tqdm

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
_SESSION_START_TIME: Optional[str] = None  # Track when session started for session spending


def _init_openrouter_client():
    """
    Initialize OpenRouter client with caching.

    Returns:
        OpenAI client configured for OpenRouter
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Set it in ~/home_workspace/api_keys.env"
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
        client = _init_openrouter_client()
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

    Automatically fetches pricing from OpenRouter API for any model,
    not just those in our curated list.

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

    # Fetch all pricing if cache is empty (gets ALL OpenRouter models)
    if not _PRICING_CACHE:
        _PRICING_CACHE = _fetch_openrouter_pricing()

    # Try exact match after fetch
    if model_id in _PRICING_CACHE:
        return _PRICING_CACHE[model_id]

    # Try base match after fetch
    if base_id in _PRICING_CACHE:
        return _PRICING_CACHE[base_id]

    # Try fuzzy match (for models with variants)
    for cached_id, price in _PRICING_CACHE.items():
        cached_base = cached_id.split(':')[0]
        if base_id == cached_base or base_id.startswith(cached_base):
            return price

    # Still not found - try one more fetch in case it's a very new model
    # This is a fallback for models added after the initial cache was populated
    try:
        fresh_pricing = _fetch_openrouter_pricing()
        if fresh_pricing:
            _PRICING_CACHE.update(fresh_pricing)

            # Try exact match again
            if model_id in _PRICING_CACHE:
                return _PRICING_CACHE[model_id]

            # Try base match again
            if base_id in _PRICING_CACHE:
                return _PRICING_CACHE[base_id]
    except Exception:
        pass

    # Return zero if no pricing found (likely a free model or pricing unavailable)
    return (0.0, 0.0)


def show_pricing_table():
    """
    Display pricing table for available models.

    NOTE: This function is now redundant since llm_list_models() displays
    pricing information in a more comprehensive table. Kept for backward
    compatibility only.

    Use llm_list_models() instead for a better formatted table with
    additional model metadata.
    """
    model_map, _ = _load_model_config()

    print("\n⚠️  NOTE: This pricing table is now redundant.")
    print("   Use llm_list_models() for a better formatted table with more details.\n")

    print("OpenRouter Model Pricing (USD)")
    print("=" * 110)
    print(f"{'Short Name':<20}{'Model ID':<40}{'In/1K':>12}{'Out/1K':>12}{'In/1M':>11}{'Out/1M':>11}")
    print("-" * 110)

    for short, model_data in sorted(model_map.items()):
        # Extract model ID from metadata (handle both old string format and new dict format)
        model_id = model_data.get('id') if isinstance(model_data, dict) else model_data

        inp_tok, out_tok = get_model_price(model_id)
        inp_1k = inp_tok * 1000
        out_1k = out_tok * 1000
        inp_1m = inp_tok * 1_000_000
        out_1m = out_tok * 1_000_000

        print(f"{short:<20}{model_id:<40}${inp_1k:>11.4f}${out_1k:>11.4f}${inp_1m:>10.2f}${out_1m:>10.2f}")

    print("\nNote: Prices fetched from OpenRouter API. Add new models to openrouter_models.json")


def llm_get_credits() -> Dict[str, float]:
    """
    Get OpenRouter credit information (fetches live from API).

    Always fetches the current credit balance from OpenRouter API and updates
    the local cost tracker. Returns both API-reported credit and tracker-based usage.

    Returns:
        Dictionary with:
        - 'limit': Total credit limit from OpenRouter (USD)
        - 'usage': Total spent according to local tracker (USD)
        - 'remaining': Actual remaining credit from OpenRouter API (USD)
        - 'api_credit': Same as 'remaining' (for clarity)

    Example:
        >>> credits = llm_get_credits()
        >>> print(f"Remaining: ${credits['remaining']:.2f}")
        >>> print(f"Used (tracker): ${credits['usage']:.2f}")
    """
    # Fetch live credit from OpenRouter API
    actual_credit = get_openrouter_credit()

    # Load local tracker
    data = _load_cost_tracker()

    if actual_credit is not None:
        # Update tracker with actual credit
        data['baseline_credit'] = actual_credit
        data['actual_credit'] = actual_credit
        data['credit_last_checked'] = datetime.now().isoformat()
        _save_cost_tracker(data)

        return {
            'limit': actual_credit,  # Use API credit as the limit
            'usage': data.get('total_spend', 0.0),
            'remaining': actual_credit,
            'api_credit': actual_credit
        }
    else:
        # Fallback to tracker-based estimates if API call fails
        limit = data.get("baseline_credit", 15.0)
        usage = data.get("total_spend", 0.0)

        print("⚠️  Could not fetch live credit from OpenRouter API, using local tracker estimate")

        return {
            'limit': limit,
            'usage': usage,
            'remaining': limit - usage,
            'api_credit': None
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


def init_cost_tracking():
    """
    Initialize the cost tracking system.

    Call this at the beginning of your notebook (after config_paths_keys()) to:
    - Initialize cost tracking file if it doesn't exist
    - Fetch current OpenRouter credit and update baseline
    - Pre-load pricing data cache for ALL OpenRouter models (not just curated list)
    - Start session spending tracker

    This ensures cost tracking and estimation works for ANY OpenRouter model,
    including models not in our curated list.

    Example:
        >>> from introdl.utils import config_paths_keys
        >>> from introdl.nlp import init_cost_tracking
        >>>
        >>> paths = config_paths_keys()  # Loads API keys
        >>> init_cost_tracking()         # Initialize cost tracking
    """
    # Set session start time for session spending tracking
    global _SESSION_START_TIME
    _SESSION_START_TIME = datetime.now().isoformat()

    # Ensure cost tracker file exists
    data = _load_cost_tracker()

    # Fetch and update OpenRouter credit baseline
    credit = update_openrouter_credit()

    # Pre-load pricing cache (fetches ALL OpenRouter models, not just curated list)
    global _PRICING_CACHE
    if not _PRICING_CACHE:
        try:
            _PRICING_CACHE = _fetch_openrouter_pricing()
            print(f"✅ Loaded pricing for {len(_PRICING_CACHE)} OpenRouter models")
        except Exception as e:
            print(f"⚠️  Could not pre-load pricing: {e}")

    # Report status
    if credit is not None:
        print(f"✅ Cost tracking initialized (${credit:.2f} credit remaining)")
    else:
        print("✅ Cost tracking initialized (could not fetch live credit)")


def show_session_spending():
    """
    Display spending for the current notebook session.

    Shows all API calls made since init_cost_tracking() was called in this session,
    including per-model breakdown and total session cost.

    Note: Call init_cost_tracking() at the start of your notebook to start tracking
    session spending. Without it, this function will show all recent history instead.

    Example:
        >>> # At start of notebook
        >>> init_cost_tracking()
        >>>
        >>> # ... make API calls ...
        >>>
        >>> # At end of notebook or any time during session
        >>> show_session_spending()
    """
    global _SESSION_START_TIME

    data = _load_cost_tracker()
    history = data.get("history", [])

    if not history:
        print("📊 No API calls recorded in history")
        return

    # Filter to session calls if we have a session start time
    if _SESSION_START_TIME:
        session_calls = [
            call for call in history
            if call.get("timestamp", "") >= _SESSION_START_TIME
        ]
        session_label = "Current Session"
    else:
        # No session start time - show recent history instead
        session_calls = history[-20:] if len(history) > 20 else history
        session_label = "Recent History (no session start time set)"
        print("⚠️  Session start time not set. Call init_cost_tracking() at notebook start for accurate session tracking.\n")

    if not session_calls:
        print(f"📊 No API calls in {session_label.lower()}")
        return

    # Calculate session totals
    session_cost = sum(call.get("cost", 0) for call in session_calls)
    session_prompt_tokens = sum(call.get("prompt_tokens", 0) for call in session_calls)
    session_completion_tokens = sum(call.get("completion_tokens", 0) for call in session_calls)
    session_call_count = len(session_calls)

    # Group by model
    by_model = {}
    for call in session_calls:
        model = call.get("model", "unknown")
        if model not in by_model:
            by_model[model] = {"cost": 0.0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        by_model[model]["cost"] += call.get("cost", 0)
        by_model[model]["calls"] += 1
        by_model[model]["prompt_tokens"] += call.get("prompt_tokens", 0)
        by_model[model]["completion_tokens"] += call.get("completion_tokens", 0)

    # Display summary
    print("\n" + "=" * 70)
    print(f"💰 {session_label} Spending Summary")
    print("=" * 70)
    print(f"Total Cost:          ${session_cost:.6f}")
    print(f"Total API Calls:     {session_call_count}")
    print(f"Total Tokens:        {session_prompt_tokens:,} in / {session_completion_tokens:,} out")

    if by_model:
        print("\n" + "-" * 70)
        print("By Model:")
        for model, stats in sorted(by_model.items(), key=lambda x: -x[1]["cost"]):
            print(f"  {model}")
            print(f"    Cost: ${stats['cost']:.6f} | Calls: {stats['calls']} | Tokens: {stats['prompt_tokens']:,} in / {stats['completion_tokens']:,} out")

    # Fetch live credit from OpenRouter API and show summary
    credit = get_openrouter_credit()

    print("\n" + "-" * 70)
    print(f"Total Spent this session: ${session_cost:.6f}")
    if credit is not None:
        print(f"Approximate Credit remaining: ${credit:.2f}")
        print("(Note: This balance may not reflect the most recent spending)")
    else:
        # Fallback to estimate if API call fails
        baseline = data.get("baseline_credit", 15.0)
        total_spend = data.get("total_spend", 0.0)
        remaining = baseline - total_spend
        print(f"Approximate Credit remaining: ${remaining:.2f} (estimated)")
        print("(Note: Could not fetch live credit from OpenRouter API)")
    print("=" * 70 + "\n")


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
# Direct HTTP Request Helper (for OpenRouter reasoning mode)
# ============================================================================

def _make_openrouter_http_request(
    api_key: str,
    model_id: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
    response_format: Optional[dict] = None,
    reasoning_params: Optional[dict] = None
) -> dict:
    """
    Make direct HTTP POST request to OpenRouter API.

    Required for reasoning mode because the OpenAI SDK doesn't support
    custom top-level parameters like 'reasoning'.

    Args:
        api_key: OpenRouter API key
        model_id: Full model ID (e.g., 'google/gemini-2.5-flash')
        messages: Chat messages array
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        response_format: Optional response format dict for JSON mode
        reasoning_params: Optional reasoning configuration dict

    Returns:
        Response dict with 'choices', 'usage', etc. (compatible with OpenAI format)
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "http://localhost"),
        "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "DS776 Deep Learning"),
    }

    # Build request payload
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # Add response format if specified (JSON mode)
    if response_format:
        payload["response_format"] = response_format

    # Add reasoning parameters at TOP LEVEL (not in extra_body)
    if reasoning_params:
        payload["reasoning"] = reasoning_params

    # Make request
    response = requests.post(url, headers=headers, json=payload, timeout=60)

    # Check for errors and provide detailed error message
    if response.status_code != 200:
        error_detail = response.text
        try:
            error_json = response.json()
            error_msg = error_json.get('error', {}).get('message', error_detail)
        except:
            error_msg = error_detail

        raise requests.HTTPError(
            f"OpenRouter API error ({response.status_code}): {error_msg}"
        )

    return response.json()


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
    max_tokens: Optional[int] = 500,
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    cost_per_m_in: Optional[float] = None,
    cost_per_m_out: Optional[float] = None,
    print_cost: bool = False,
    estimate_cost: bool = None,  # Alias for print_cost
    track_cost: bool = True,
    enable_reasoning: bool = False,
    reasoning_effort: str = "medium",
    reasoning_budget: Optional[int] = None,
    **kwargs
) -> Union[str, dict, List[str], List[dict]]:
    """
    Generate text or JSON from LLM using any OpenAI-compatible API.

    Defaults to OpenRouter (with automatic cost tracking). Works with OpenAI, Anthropic,
    Google, Together.AI, Groq, or any OpenAI-compatible endpoint.

    Args:
        model_name: Model name - can be:
                   - Short name from our curated list (e.g., 'gemini-flash-lite')
                   - Full model ID (e.g., 'google/gemini-2.5-flash-lite')
                   - For other providers, use their native model names
        prompts: Single prompt or list of prompts
        mode: 'text' or 'json'
        user_schema: Optional JSON schema for structured output (mode='json' only)
        system_prompt: Optional system message
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate. If None, uses model's default/maximum output length.
        api_key: API key to use. If None, uses OPENROUTER_API_KEY from environment.
                 For other providers, load keys via config_paths_keys() from
                 ~/home_workspace/api_keys.env and pass as os.environ['PROVIDER_API_KEY']
        base_url: Base URL for the API endpoint. Defaults to OpenRouter.
                  Common endpoints:
                  - OpenRouter: https://openrouter.ai/api/v1 (default)
                  - OpenAI: https://api.openai.com/v1
                  - Anthropic: https://api.anthropic.com/v1
                  - Google: https://generativelanguage.googleapis.com/v1beta/openai/
                  - Together.AI: https://api.together.xyz/v1
                  - Groq: https://api.groq.com/openai/v1
        cost_per_m_in: Cost per million input tokens (for cost tracking with non-OpenRouter providers)
        cost_per_m_out: Cost per million output tokens (for cost tracking with non-OpenRouter providers)
        print_cost: Print cost after generation
        track_cost: Update persistent cost tracking (OpenRouter only)
        enable_reasoning: Enable reasoning/thinking mode for supported models
        reasoning_effort: Effort level for reasoning ("low", "medium", "high") - used by OpenAI and DeepSeek
        reasoning_budget: Token budget for reasoning output (1024+) - used by Anthropic and Gemini
        **kwargs: Additional parameters (for future extensions)

    Returns:
        - mode='text': str or List[str]
        - mode='json': dict or List[dict]

    Note:
        **Cost Tracking:**
        - OpenRouter: Automatic cost tracking and pricing lookup
        - Other providers: Provide cost_per_m_in and cost_per_m_out for manual tracking,
          or monitor costs through provider dashboards:
          - OpenAI: https://platform.openai.com/usage
          - Anthropic: https://console.anthropic.com/settings/usage
          - Google: https://console.cloud.google.com/billing
          - Together.AI: https://api.together.xyz/settings/billing
          - Groq: https://console.groq.com/settings/billing

    Examples:
        >>> # OpenRouter (default): Automatic cost tracking
        >>> response = llm_generate('gpt-4o-mini', 'What is deep learning?')

        >>> # OpenRouter: Use ANY model by full ID
        >>> response = llm_generate('anthropic/claude-opus-4', 'Explain transformers')

        >>> # OpenAI: Use GPT models with manual cost tracking
        >>> response = llm_generate(
        ...     'gpt-4o',
        ...     'Explain deep learning',
        ...     api_key=os.environ['OPENAI_API_KEY'],
        ...     base_url='https://api.openai.com/v1',
        ...     cost_per_m_in=2.50,  # $2.50 per M input tokens
        ...     cost_per_m_out=10.00,  # $10.00 per M output tokens
        ...     print_cost=True
        ... )

        >>> # Anthropic: Use Claude models
        >>> response = llm_generate(
        ...     'claude-3-5-sonnet-20241022',
        ...     'What is deep learning?',
        ...     api_key=os.environ['ANTHROPIC_API_KEY'],
        ...     base_url='https://api.anthropic.com/v1'
        ... )

        >>> # Google: Use Gemini models
        >>> response = llm_generate(
        ...     'gemini-2.0-flash-exp',
        ...     'Explain transformers',
        ...     api_key=os.environ['GOOGLE_API_KEY'],
        ...     base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
        ... )

        >>> # Together.AI: Use their models
        >>> response = llm_generate(
        ...     'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        ...     'What is ML?',
        ...     api_key=os.environ['TOGETHER_API_KEY'],
        ...     base_url='https://api.together.xyz/v1'
        ... )

        >>> # Groq: Ultra-fast inference
        >>> response = llm_generate(
        ...     'llama-3.3-70b-versatile',
        ...     'Define ML',
        ...     api_key=os.environ['GROQ_API_KEY'],
        ...     base_url='https://api.groq.com/openai/v1'
        ... )

        >>> # JSON generation with schema (works with all providers)
        >>> schema = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}
        >>> response = llm_generate('deepseek-v3.1', 'Answer in JSON',
        ...                         mode='json', user_schema=schema)

        >>> # Enable reasoning for Gemini (token budget)
        >>> response = llm_generate('gemini-flash',
        ...                         'Solve this complex problem: ...',
        ...                         enable_reasoning=True,
        ...                         reasoning_budget=2000)  # 2000 tokens for reasoning

        >>> # Enable reasoning for Claude (token budget)
        >>> response = llm_generate('anthropic/claude-opus-4',
        ...                         'Think through this step-by-step: ...',
        ...                         enable_reasoning=True,
        ...                         reasoning_budget=3000)

        >>> # Enable reasoning for OpenAI o3 (effort level)
        >>> response = llm_generate('openai/o3-mini',
        ...                         'Complex reasoning task...',
        ...                         enable_reasoning=True,
        ...                         reasoning_effort='high')
    """
    # Handle estimate_cost as alias for print_cost
    if estimate_cost is not None:
        print_cost = estimate_cost

    # Check if this is OpenRouter (for automatic cost tracking)
    is_openrouter = 'openrouter' in base_url.lower()

    # Resolve model name for OpenRouter (short names -> full IDs)
    if is_openrouter:
        model_id = resolve_model_name(model_name)
    else:
        model_id = model_name  # Use model name as-is for other providers

    # Initialize client
    if api_key is None:
        # No API key provided - default to OpenRouter from environment
        if is_openrouter:
            client = _init_openrouter_client()
        else:
            raise ValueError(
                f"api_key required for non-OpenRouter endpoint: {base_url}\n"
                f"Load keys via config_paths_keys() then pass: api_key=os.environ['YOUR_PROVIDER_API_KEY']"
            )
    else:
        # API key provided - use it with the base_url
        if is_openrouter:
            # For OpenRouter, include special headers
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "http://localhost"),
                    "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "DS776 Deep Learning"),
                },
            )
        else:
            # For other providers, simple client
            client = OpenAI(api_key=api_key, base_url=base_url)

    # Handle batch vs single prompt
    is_batch = isinstance(prompts, list)
    prompt_list = prompts if is_batch else [prompts]

    # Get pricing
    if is_openrouter:
        # OpenRouter: Automatic pricing lookup
        inp_price, out_price = get_model_price(model_id)
    elif cost_per_m_in is not None and cost_per_m_out is not None:
        # Manual pricing provided - convert from $/M to $/token
        inp_price = cost_per_m_in / 1_000_000
        out_price = cost_per_m_out / 1_000_000
    else:
        # No pricing available
        inp_price, out_price = 0.0, 0.0
        # Warn if trying to track/print costs without pricing
        if print_cost or track_cost:
            print("⚠️  Cost tracking requested but no pricing provided. Add cost_per_m_in and cost_per_m_out parameters to track costs for non-OpenRouter providers.")

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

    # Determine if we should show progress bar (only for multiple prompts)
    show_progress = len(prompt_list) > 1

    # Wrap iterator with tqdm if showing progress
    # Use smoothing and dynamic_ncols for better display in notebooks
    iterator = tqdm(
        prompt_list,
        desc="Generating",
        disable=not show_progress,
        smoothing=0.1,  # Smooth ETA estimates
        dynamic_ncols=True,  # Adapt to terminal width
        unit="prompt"
    ) if show_progress else prompt_list

    for prompt in iterator:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Build API call parameters
        api_params = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature
        }

        # Only include max_tokens if specified (None means use model default)
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

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

        # Handle reasoning/thinking mode
        # Check model's default reasoning configuration if enable_reasoning not explicitly set
        if is_openrouter:
            metadata = get_model_metadata(model_name)
            reasoning_config = metadata.get('reasoning_support', {})
            supports_reasoning = reasoning_config.get('supports_reasoning', False)
            default_reasoning = reasoning_config.get('default_reasoning_enabled', False)
            reasoning_type = reasoning_config.get('reasoning_type')
        else:
            # For non-OpenRouter, assume no reasoning support unless explicitly enabled
            supports_reasoning = False
            default_reasoning = False
            reasoning_type = None

        # Determine if reasoning should be enabled
        # Priority: explicit enable_reasoning parameter > model default
        should_enable_reasoning = enable_reasoning if enable_reasoning is not None else default_reasoning

        # Build reasoning parameters for OpenRouter direct HTTP request
        reasoning_params = None
        if is_openrouter and should_enable_reasoning and supports_reasoning:
            if reasoning_type == "effort":
                # OpenAI (o1, o3), DeepSeek, and Meta (Llama-4-maverick) use effort levels
                reasoning_params = {
                    "effort": reasoning_effort  # "low", "medium", "high"
                }
            elif reasoning_type == "budget":
                # Anthropic (Claude) and Gemini use token budgets
                # IMPORTANT: Budget-based reasoning requires max_tokens >= 1024
                # If max_tokens is None, use reasoning_budget or default to 2048
                if reasoning_budget is not None:
                    budget = reasoning_budget
                elif max_tokens is not None:
                    budget = max_tokens
                else:
                    budget = 2048  # Default for budget-based reasoning when max_tokens is None

                # Enforce minimum 1024 tokens for budget-based reasoning
                if max_tokens is not None and max_tokens < 1024:
                    max_tokens = 1024
                    if print_cost:
                        print(f"ℹ️  Budget-based reasoning requires max_tokens >= 1024. Adjusted to {max_tokens}")
                elif max_tokens is None:
                    # When max_tokens is None, we need to set it to at least 1024 for reasoning
                    max_tokens = max(1024, budget)
                    if print_cost:
                        print(f"ℹ️  Budget-based reasoning requires max_tokens >= 1024. Set to {max_tokens}")

                reasoning_params = {
                    "max_tokens": max(1024, budget)  # Minimum 1024 tokens for reasoning budget
                }
            elif should_enable_reasoning:
                # Warn if reasoning requested but type unknown
                if print_cost:
                    print(f"⚠️  Model '{model_id}' reasoning type unknown, attempting with effort-based")
                reasoning_params = {
                    "effort": reasoning_effort
                }
        elif is_openrouter and not should_enable_reasoning and supports_reasoning:
            # Explicitly disable reasoning for models that support it (e.g., deepseek)
            reasoning_params = {
                "enabled": False
            }

        # Make API call
        try:
            # Use direct HTTP request for OpenRouter reasoning mode
            # (OpenAI SDK doesn't support custom top-level parameters like 'reasoning')
            if is_openrouter and reasoning_params is not None:
                # Get API key
                openrouter_key = api_key if api_key else os.environ.get('OPENROUTER_API_KEY')
                if not openrouter_key:
                    raise ValueError("OPENROUTER_API_KEY not found")

                # Extract response_format if present
                response_format_param = api_params.get("response_format")

                # Make direct HTTP request
                response_json = _make_openrouter_http_request(
                    api_key=openrouter_key,
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format_param,
                    reasoning_params=reasoning_params
                )

                # Convert to object-like structure for compatibility
                class Response:
                    def __init__(self, data):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': data['choices'][0]['message']['content']
                            })()
                        })()]
                        self.usage = type('obj', (object,), {
                            'prompt_tokens': data['usage']['prompt_tokens'],
                            'completion_tokens': data['usage']['completion_tokens']
                        })()

                response = Response(response_json)
            else:
                # Standard OpenAI SDK call (no reasoning)
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

    # Update cost tracking (OpenRouter only)
    if is_openrouter and track_cost and total_cost > 0:
        _update_spend(model_id, total_prompt_tokens, total_completion_tokens, total_cost)

    # Print cost if requested
    if print_cost and total_cost > 0:
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
        verbose: If True, display formatted table
        json_schema: If True, only show models that support json_schema
        strict_schema: If True, only show models that support strict_schema

    Returns:
        Dictionary keyed by short name with model information:
        {
            'gemini-flash-lite': {
                'model_id': 'google/gemini-2.5-flash-lite',
                'size': '?',
                'release_date': '2025-01',
                'cost_in_per_m': 0.10,
                'cost_out_per_m': 0.40,
                'json_schema': True,
                'open_weights': False,
                'provider': 'google'
            },
            ...
        }
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

    # Build result dictionary with model info
    result = {}
    rows = []  # For verbose display

    for short, metadata in sorted(filtered_models.items()):
        model_id = metadata['id']
        json_support = metadata.get('json_support', {})

        inp_tok, out_tok = get_model_price(model_id)
        inp_1m = inp_tok * 1_000_000
        out_1m = out_tok * 1_000_000

        # Release date (format as YYYY-MM for display, keep full for dict)
        release_date = metadata.get('release_date', 'Unknown')
        release_date_display = release_date
        if release_date != 'Unknown':
            # Format as YYYY-MM for compact display
            release_date_display = '-'.join(release_date.split('-')[:2])

        # Size (display size field which handles MoE specially)
        size = metadata.get('size', metadata.get('parameters', '?'))

        # Build result dictionary entry
        result[short] = {
            'model_id': model_id,
            'size': size,
            'release_date': release_date,
            'cost_in_per_m': inp_1m,
            'cost_out_per_m': out_1m,
            'json_schema': json_support.get('json_schema', False),
            'open_weights': metadata.get('open_weights', False),
            'provider': metadata.get('provider', model_id.split('/')[0] if '/' in model_id else 'unknown')
        }

        # Build row for display table
        if verbose:
            schema_support = "✅" if json_support.get('json_schema', False) else "❌"
            open_weights = "✅" if metadata.get('open_weights', False) else "❌"

            rows.append({
                'Short Name': short,
                'Size': size,
                'Released': release_date_display,
                'In/M': f'${inp_1m:.2f}',
                'Out/M': f'${out_1m:.2f}',
                'JSON Schema': schema_support,
                'Open Weights': open_weights
            })

    if verbose:

        # Try to display as pandas DataFrame for best formatting
        try:
            import pandas as pd
            from IPython.display import display

            df = pd.DataFrame(rows)

            filter_note = ""
            if json_schema:
                filter_note = " (json_schema support only)"
            elif strict_schema:
                filter_note = " (strict_schema support only)"

            print(f"\nAvailable OpenRouter Models{filter_note}:\n")
            display(df)
            print(f"\nDefault model: {config.get('default_model', 'gemini-flash-lite')}")
            print(f"Size format: Dense models show total params (e.g., '70B'), MoE models show active×experts (e.g., '17B×128E')")
            print(f"JSON Schema = User-defined JSON schemas supported")
            print(f"\nYou can also use any OpenRouter model by its full ID (e.g., 'openai/gpt-4o')")

        except ImportError:
            # Fallback to markdown table if pandas not available
            from IPython.display import display, Markdown

            filter_note = ""
            if json_schema:
                filter_note = " (json_schema support only)"
            elif strict_schema:
                filter_note = " (strict_schema support only)"

            # Build markdown table
            md = f"\n**Available OpenRouter Models{filter_note}:**\n\n"
            md += "| Short Name | Size | Released | In/M | Out/M | JSON Schema | Open Weights |\n"
            md += "|------------|------|----------|------|-------|-------------|-------------|\n"

            for row in rows:
                md += f"| {row['Short Name']} | {row['Size']} | {row['Released']} | {row['In/M']} | {row['Out/M']} | {row['JSON Schema']} | {row['Open Weights']} |\n"

            md += f"\n**Default model:** {config.get('default_model', 'gemini-flash-lite')}\n\n"
            md += "**Size format:** Dense models show total params (e.g., '70B'), MoE models show active×experts (e.g., '17B×128E')\n\n"
            md += "**JSON Schema** = User-defined JSON schemas supported\n\n"
            md += "You can also use any OpenRouter model by its full ID (e.g., 'openai/gpt-4o')\n"

            display(Markdown(md))

    return result


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


# ============================================================================
# TrainerWithPretend - HuggingFace Trainer with Smart Loading
# ============================================================================

try:
    from transformers import Trainer as HFTrainer
    from transformers import AutoModelForSequenceClassification
    from transformers.trainer_utils import TrainOutput
    import pandas as pd
    import time
    from tqdm.auto import tqdm

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


if _HAS_TRANSFORMERS:
    class TrainerWithPretend(HFTrainer):
        """
        Drop-in replacement for HuggingFace Trainer with smart local caching.

        Extends the standard Trainer with a pretend_train mode that:
        1. Checks HuggingFace Hub for instructor-hosted model (hobbes99/DS776-models/{model_name}/)
        2. If not on Hub, checks for existing trained model in {output_dir}/best_model/
        3. If found, loads it and displays training metrics
        4. If not found, proceeds with actual training
        5. Always saves to best_model/ after training

        This enables students to re-run notebooks without retraining,
        similar to train_network's pretend_train behavior.

        Usage (identical to standard HuggingFace Trainer):
            from introdl.nlp import Trainer  # Extended version

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                pretend_train=True  # Only extra parameter
            )

            trainer.train()  # First run: trains, second run: loads
            history = trainer.get_training_history()  # Access metrics DataFrame

        Args:
            pretend_train (bool): If True, loads existing model instead of training
            All other args: Same as standard HuggingFace Trainer
        """

        def __init__(self, *args, pretend_train=False, **kwargs):
            self.pretend_train = pretend_train
            self._training_history = None
            self._detailed_training_history = None  # For rich metrics display (NER, etc.)
            super().__init__(*args, **kwargs)

        def train(self, *args, **kwargs):
            """
            Standard train() with smart loading.

            If pretend_train=True and model exists locally or on HF Hub:
                - Loads the model
                - Displays training history
                - Skips training
            Otherwise:
                - Performs actual training
                - Saves to best_model/ directory
                - Saves training_history.json

            Returns:
                TrainOutput: Standard HuggingFace training output
            """
            if self.pretend_train and self._try_load_local():
                # Model already trained, simulate and return
                self._simulate_training_with_metrics()
                return self._create_train_output()

            # No local model found, or pretend_train=False
            # Proceed with actual training
            output = super().train(*args, **kwargs)

            # Save model and training history
            self._save_best_model()
            self._save_training_history()

            return output

        def _try_load_local(self):
            """
            Try loading model from HuggingFace Hub or local directory.

            Priority order:
            1. HuggingFace Hub: hobbes99/DS776-models/{model_name}/ (auto-derived from output_dir)
            2. Local best_model/ directory
            3. Train from scratch

            Returns:
                bool: True if model loaded successfully
            """
            # Derive model name from output_dir
            # Example: 'Lesson_08_Models/L08_fine_tune_distilbert' → 'L08_fine_tune_distilbert'
            output_path = Path(self.args.output_dir)
            model_name = output_path.name  # Last component of path

            # FIRST: Try HuggingFace Hub
            hf_repo_id = "hobbes99/DS776-models"

            try:
                print(f"✓ Checking HuggingFace Hub: {hf_repo_id}/{model_name}")

                # Load model from HuggingFace Hub subdirectory
                # Detect model type from config and use appropriate Auto class
                from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification
                config = AutoConfig.from_pretrained(hf_repo_id, subfolder=model_name)

                # Choose the correct model class based on architecture in config
                if 'TokenClassification' in config.architectures[0]:
                    self.model = AutoModelForTokenClassification.from_pretrained(
                        hf_repo_id,
                        subfolder=model_name,
                        trust_remote_code=False
                    )
                else:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        hf_repo_id,
                        subfolder=model_name,
                        trust_remote_code=False
                    )

                # Load training history from HF Hub if available
                self._training_history = self._load_training_history_from_hub(hf_repo_id, model_name)

                # Cache locally for future use
                best_model_dir = Path(self.args.output_dir) / 'best_model'
                best_model_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(best_model_dir)
                self.tokenizer.save_pretrained(best_model_dir)

                # Save training history locally if we got it from HF Hub
                if self._training_history is not None:
                    history_file = best_model_dir / 'training_history.json'
                    self._training_history.to_json(history_file, orient='records', indent=2)

                try:
                    print(f"✓ Model cached locally to: {best_model_dir.relative_to(Path.cwd())}")
                except ValueError:
                    print(f"✓ Model cached locally to: {best_model_dir}")
                return True

            except Exception as e:
                print(f"⚠ HuggingFace Hub load failed: {e}")
                print("  Falling back to local model or training from scratch...")

            # SECOND: Try local best_model/ directory
            best_model_dir = Path(self.args.output_dir) / 'best_model'

            if not self._is_complete_model(best_model_dir):
                return False

            try:
                try:
                    print(f"✓ Loading pre-trained model from: {best_model_dir.relative_to(Path.cwd())}")
                except ValueError:
                    print(f"✓ Loading pre-trained model from: {best_model_dir}")

                # Load model
                # Detect model type from config and use appropriate Auto class
                from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification
                config = AutoConfig.from_pretrained(best_model_dir, local_files_only=True)

                # Choose the correct model class based on architecture in config
                if 'TokenClassification' in config.architectures[0]:
                    self.model = AutoModelForTokenClassification.from_pretrained(
                        best_model_dir,
                        local_files_only=True
                    )
                else:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        best_model_dir,
                        local_files_only=True
                    )

                # Load training history
                self._training_history = self._load_training_history(best_model_dir)

                return True

            except Exception as e:
                print(f"⚠ Load failed: {e}. Training from scratch...")
                return False

        def _is_complete_model(self, model_dir):
            """Check if directory has required model files."""
            if not model_dir.exists():
                return False

            has_config = (model_dir / 'config.json').exists()
            has_weights = (
                (model_dir / 'model.safetensors').exists() or
                (model_dir / 'pytorch_model.bin').exists()
            )

            return has_config and has_weights

        def _save_best_model(self):
            """Save model to best_model/ after training."""
            best_model_dir = Path(self.args.output_dir) / 'best_model'
            best_model_dir.mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(best_model_dir)
            self.tokenizer.save_pretrained(best_model_dir)

            try:
                print(f"\n✓ Model saved to: {best_model_dir.relative_to(Path.cwd())}")
            except ValueError:
                print(f"\n✓ Model saved to: {best_model_dir}")

        def _save_training_history(self):
            """Save training metrics after actual training."""
            best_model_dir = Path(self.args.output_dir) / 'best_model'

            # Extract metrics from HuggingFace's log_history
            history_df = self._extract_metrics_dataframe()

            # Save simple format (backward compatible)
            history_file = best_model_dir / 'training_history.json'
            history_df.to_json(history_file, orient='records', indent=2)

            # Check if we have detailed metrics with nested dictionaries
            detailed_metrics = self._extract_detailed_metrics()
            if detailed_metrics:
                # Save detailed format for rich display
                import json
                detailed_file = best_model_dir / 'training_history_detailed.json'
                with open(detailed_file, 'w') as f:
                    json.dump(detailed_metrics, f, indent=2)

        def _load_training_history(self, model_dir):
            """Load training metrics from saved file."""
            # Try loading detailed metrics first (for rich display)
            import json
            detailed_file = model_dir / 'training_history_detailed.json'
            if detailed_file.exists():
                with open(detailed_file, 'r') as f:
                    self._detailed_training_history = json.load(f)

            # Always load simple format for backward compatibility
            history_file = model_dir / 'training_history.json'
            if history_file.exists():
                return pd.read_json(history_file)
            return None

        def _load_training_history_from_hub(self, repo_id, model_name):
            """
            Load training history from HuggingFace Hub.

            Args:
                repo_id: Base HuggingFace repository ID (e.g., 'hobbes99/DS776-models')
                model_name: Model subdirectory name (e.g., 'L08_fine_tune_distilbert')

            Returns:
                pd.DataFrame or None: Training history if available
            """
            try:
                from huggingface_hub import hf_hub_download
                import json

                # Try loading detailed metrics first (for rich display)
                try:
                    detailed_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=f'{model_name}/training_history_detailed.json',
                        repo_type='model'
                    )
                    with open(detailed_file, 'r') as f:
                        self._detailed_training_history = json.load(f)
                except Exception:
                    # Detailed history not available (optional)
                    pass

                # Download training_history.json from model's subdirectory
                # Path in repo: {model_name}/training_history.json
                history_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=f'{model_name}/training_history.json',
                    repo_type='model'
                )

                return pd.read_json(history_file)

            except Exception:
                # Training history not available on Hub (not critical)
                return None

        def _extract_metrics_dataframe(self):
            """
            Convert Trainer.state.log_history to clean DataFrame.

            Returns DataFrame with columns:
                epoch, train_loss, eval_loss, eval_accuracy, eval_f1, etc.
            """
            logs = self.state.log_history

            # Separate training and eval logs
            train_logs = [log for log in logs if 'loss' in log and 'epoch' in log]
            eval_logs = [log for log in logs if 'eval_loss' in log]

            # Build structured data
            data = []
            for eval_log in eval_logs:
                epoch = eval_log.get('epoch')

                # Find corresponding training loss
                train_loss = None
                for train_log in train_logs:
                    if train_log.get('epoch') == epoch:
                        train_loss = train_log.get('loss')
                        break

                # Start with epoch and train_loss
                row = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                }

                # Dynamically add all eval_* metrics from this log
                # This ensures we capture custom metrics like eval_overall_f1, eval_overall_precision, etc.
                for key, value in eval_log.items():
                    if key.startswith('eval_'):
                        row[key] = value

                data.append(row)

            df = pd.DataFrame(data)

            # Drop columns that are entirely null/None (except epoch which we always keep)
            # This removes unused metric columns like eval_accuracy, eval_f1 when doing NER
            cols_to_keep = ['epoch']  # Always keep epoch
            for col in df.columns:
                if col not in cols_to_keep and df[col].notna().any():
                    cols_to_keep.append(col)
            df = df[cols_to_keep]

            # Round to 4 decimal places for readability
            numeric_cols = df.select_dtypes(include=['float64']).columns
            df[numeric_cols] = df[numeric_cols].round(4)

            return df

        def _extract_detailed_metrics(self):
            """
            Extract detailed metrics including nested dictionaries (e.g., per-entity NER metrics).

            Returns list of dicts with structure:
                [{"epoch": 1, "train_loss": None, "eval_metrics": {complete_dict_from_compute_metrics}}, ...]

            Returns None if no nested dictionaries found (simple classification metrics only).
            """
            logs = self.state.log_history

            # Separate training and eval logs
            train_logs = [log for log in logs if 'loss' in log and 'epoch' in log]
            eval_logs = [log for log in logs if 'eval_loss' in log]

            # Check if any eval log contains nested dictionaries (like per-entity metrics)
            has_nested_metrics = False
            for eval_log in eval_logs:
                for key, value in eval_log.items():
                    if isinstance(value, dict):
                        has_nested_metrics = True
                        break
                if has_nested_metrics:
                    break

            # If no nested metrics, return None (use simple format)
            if not has_nested_metrics:
                return None

            # Build detailed data with complete nested structures
            data = []
            for eval_log in eval_logs:
                epoch = eval_log.get('epoch')

                # Find corresponding training loss
                train_loss = None
                for train_log in train_logs:
                    if train_log.get('epoch') == epoch:
                        train_loss = train_log.get('loss')
                        break

                # Collect all eval metrics (including nested dicts)
                eval_metrics = {}
                for key, value in eval_log.items():
                    if key.startswith('eval_'):
                        eval_metrics[key] = value

                data.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'eval_metrics': eval_metrics
                })

            return data

        def _format_detailed_metrics_display(self):
            """
            Format detailed metrics as DataFrame mimicking actual training display.

            Returns pandas DataFrame showing per-entity metrics in table format.
            """
            if not self._detailed_training_history:
                return None

            import pandas as pd

            # Build rows for the DataFrame
            rows = []
            for epoch_data in self._detailed_training_history:
                epoch = int(epoch_data['epoch'])
                train_loss = epoch_data.get('train_loss')
                eval_metrics = epoch_data['eval_metrics']

                row = {
                    'Epoch': epoch,
                    'Training Loss': train_loss if train_loss is not None else '',
                    'Validation Loss': eval_metrics.get('eval_loss', '')
                }

                # Add entity columns (LOC, MISC, ORG, PER, etc.)
                for key, value in sorted(eval_metrics.items()):
                    if isinstance(value, dict):
                        # Format entity metrics as a compact dict string
                        entity_name = key.replace('eval_', '')
                        # Create formatted string matching live training display
                        formatted = (
                            f"{{'precision': {value.get('precision', 0):.6f}, "
                            f"'recall': {value.get('recall', 0):.6f}, "
                            f"'f1': {value.get('f1', 0):.6f}, "
                            f"'number': {int(value.get('number', 0))}}}"
                        )
                        row[entity_name.capitalize()] = formatted

                # Add overall metrics
                if 'eval_overall_precision' in eval_metrics:
                    row['Overall Precision'] = f"{eval_metrics['eval_overall_precision']:.6f}"
                if 'eval_overall_recall' in eval_metrics:
                    row['Overall Recall'] = f"{eval_metrics['eval_overall_recall']:.6f}"
                if 'eval_overall_f1' in eval_metrics:
                    row['Overall F1'] = f"{eval_metrics['eval_overall_f1']:.6f}"
                if 'eval_overall_accuracy' in eval_metrics:
                    row['Overall Accuracy'] = f"{eval_metrics['eval_overall_accuracy']:.6f}"

                rows.append(row)

            return pd.DataFrame(rows)

        def _simulate_training_with_metrics(self):
            """Display training metrics when loading pre-trained model."""
            num_epochs = int(self.args.num_train_epochs)

            print("Model already trained. Loading checkpoint...\n")

            # Quick loading animation
            with tqdm(total=num_epochs, desc="Loading", leave=False) as pbar:
                for _ in range(num_epochs):
                    time.sleep(1 / num_epochs)  # Fast loading
                    pbar.update(1)

            # Display training history
            if self._training_history is not None:
                # Check if we have detailed metrics for rich display
                detailed_display = self._format_detailed_metrics_display()

                if detailed_display is not None:
                    # Rich display with per-entity metrics (NER tasks) - returns DataFrame
                    print("\n📊 Training History:")
                    try:
                        from IPython.display import display
                        display(detailed_display)
                    except ImportError:
                        print(detailed_display.to_string(index=False))
                else:
                    # Simple display (classification tasks or older models)
                    print("\n📊 Training History:")
                    try:
                        from IPython.display import display
                        display(self._training_history)
                    except ImportError:
                        print(self._training_history.to_string(index=False))

                # Summary of best performance
                # Check for NER metrics first (eval_overall_f1), then classification metrics (eval_accuracy)
                if 'eval_overall_f1' in self._training_history.columns:
                    # NER task - use eval_overall_f1
                    best_idx = self._training_history['eval_overall_f1'].idxmax()
                    best_epoch = self._training_history.loc[best_idx, 'epoch']
                    best_f1 = self._training_history.loc[best_idx, 'eval_overall_f1']
                    print(f"\n✓ Best model: Epoch {best_epoch:.0f} | Overall F1: {best_f1:.4f}")
                elif 'eval_accuracy' in self._training_history.columns and not self._training_history['eval_accuracy'].isna().all():
                    # Classification task - use eval_accuracy (only if not all NaN)
                    best_idx = self._training_history['eval_accuracy'].idxmax()
                    best_epoch = self._training_history.loc[best_idx, 'epoch']
                    best_acc = self._training_history.loc[best_idx, 'eval_accuracy']
                    print(f"\n✓ Best model: Epoch {best_epoch:.0f} | Accuracy: {best_acc:.4f}")
            else:
                print("⚠ No training history found. Model loaded but metrics unavailable.")

        def _create_train_output(self):
            """Create TrainOutput for compatibility."""
            return TrainOutput(global_step=0, training_loss=0.0, metrics={})

        def get_training_history(self):
            """
            Get training history as DataFrame (useful for plotting).

            Returns:
                pd.DataFrame: Training metrics with columns:
                    epoch, train_loss, eval_loss, eval_accuracy, eval_f1, etc.

            Example:
                >>> history = trainer.get_training_history()
                >>>
                >>> # Plot training curves
                >>> import matplotlib.pyplot as plt
                >>> plt.plot(history['epoch'], history['train_loss'], label='Train')
                >>> plt.plot(history['epoch'], history['eval_loss'], label='Eval')
                >>> plt.xlabel('Epoch')
                >>> plt.ylabel('Loss')
                >>> plt.legend()
                >>> plt.show()
            """
            if self._training_history is not None:
                return self._training_history
            else:
                # Just trained, extract from logs
                return self._extract_metrics_dataframe()

    # Export as "Trainer" for drop-in replacement
    Trainer = TrainerWithPretend

else:
    # Transformers not installed - provide stub
    def Trainer(*args, **kwargs):
        raise ImportError(
            "transformers package not installed. "
            "Install with: pip install transformers"
        )



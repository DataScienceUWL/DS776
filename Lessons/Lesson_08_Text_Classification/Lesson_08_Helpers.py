from introdl.nlp import llm_generate

def llm_classifier(model_name,
                   texts,
                   system_prompt,
                   prompt_template,
                   temperature=0,
                   mode='text',
                   schema=None,
                   estimate_cost=False,
                   api_key=None,
                   base_url=None,
                   enable_reasoning=False,
                   reasoning_effort=None,
                   reasoning_budget=None,
                   **kwargs):
    """
    Classify text using a Large Language Model (LLM).

    By default, uses OpenRouter with models like 'gemini-flash-lite', 'gpt-4o-mini', etc.
    Can be configured to use other OpenAI-compatible providers by passing api_key and base_url.

    Args:
        model_name (str): Name of the LLM model to use (e.g., 'gemini-flash-lite', 'gpt-4o-mini')
        texts (list of str): List of text documents to classify.
        system_prompt (str): System prompt to guide the LLM.
        prompt_template (str): Template for user prompts to classify each text.
                              Use {text} placeholder for document text.
        temperature (float): Sampling temperature (0 for deterministic, higher for more random).
                            Default 0 for consistent classification results.
        mode (str): Output mode - 'text' for plain text, 'json' for structured JSON output.
                   Default 'text'.
        schema (dict, optional): Pydantic model or JSON schema dict for structured output.
                                Required when mode='json'. Use for enforcing output format.
                                Default None.
        estimate_cost (bool): Whether to print cost estimates (OpenRouter only). Default False.
        api_key (str, optional): API key for non-OpenRouter providers. If None, uses OpenRouter.
        base_url (str, optional): Base URL for non-OpenRouter providers. If None, uses OpenRouter.
        enable_reasoning (bool): Enable reasoning mode for supported models. Default False.
        reasoning_effort (str, optional): Reasoning effort level ('low', 'medium', 'high') for
                                         effort-based models (OpenAI, DeepSeek, Llama-4-maverick).
        reasoning_budget (int, optional): Token budget for reasoning in budget-based models
                                         (Claude, Gemini). Minimum 1024 tokens.
        **kwargs: Additional arguments to pass to llm_generate() (e.g., max_tokens, cost_per_m_in, etc.)

    Returns:
        list of str or list of dict: Predicted labels for the input texts.
                                     Returns strings for mode='text', dicts for mode='json'.

    Examples:
        # Using OpenRouter (default text mode)
        predictions = llm_classifier(
            'gemini-flash-lite',
            texts,
            system_prompt="You are an expert classifier.",
            prompt_template="Classify: {text}\nLabel:"
        )

        # Using JSON mode with schema for structured output
        from pydantic import BaseModel
        class Classification(BaseModel):
            label: str
            confidence: float

        predictions = llm_classifier(
            'gemini-flash-lite',
            texts,
            system_prompt="You are an expert classifier.",
            prompt_template="Classify: {text}",
            mode='json',
            schema=Classification
        )

        # Using reasoning mode for complex classification
        predictions = llm_classifier(
            'claude-haiku',
            texts,
            system_prompt="You are an expert classifier.",
            prompt_template="Classify: {text}\nLabel:",
            enable_reasoning=True,
            reasoning_effort='high'
        )

        # Using another OpenAI-compatible provider
        predictions = llm_classifier(
            'gpt-4',
            texts,
            system_prompt="You are an expert classifier.",
            prompt_template="Classify: {text}\nLabel:",
            api_key="your-api-key",
            base_url="https://api.yourprovider.com/v1"
        )
    """

    user_prompts = [prompt_template.format(text=text) for text in texts]

    # Build kwargs for llm_generate
    llm_kwargs = {
        'model_name': model_name,
        'prompts': user_prompts,
        'mode': mode,
        'system_prompt': system_prompt,
        'temperature': temperature,
        'estimate_cost': estimate_cost,
        **kwargs
    }

    # Add optional parameters only if provided
    if schema is not None:
        llm_kwargs['user_schema'] = schema
    if api_key is not None:
        llm_kwargs['api_key'] = api_key
    if base_url is not None:
        llm_kwargs['base_url'] = base_url

    # Pass reasoning parameters if provided
    if enable_reasoning:
        llm_kwargs['enable_reasoning'] = enable_reasoning
    if reasoning_effort is not None:
        llm_kwargs['reasoning_effort'] = reasoning_effort
    if reasoning_budget is not None:
        llm_kwargs['reasoning_budget'] = reasoning_budget

    predicted_labels = llm_generate(**llm_kwargs)

    return predicted_labels
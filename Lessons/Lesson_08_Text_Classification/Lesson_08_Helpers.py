from introdl.nlp import llm_generate

def llm_classifier(model_name,
                   texts,
                   system_prompt,
                   prompt_template,
                   temperature=0,
                   estimate_cost=False,
                   api_key=None,
                   base_url=None,
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
        estimate_cost (bool): Whether to print cost estimates (OpenRouter only). Default False.
        api_key (str, optional): API key for non-OpenRouter providers. If None, uses OpenRouter.
        base_url (str, optional): Base URL for non-OpenRouter providers. If None, uses OpenRouter.
        **kwargs: Additional arguments to pass to llm_generate() (e.g., max_tokens, cost_per_m_in, etc.)

    Returns:
        list of str: Predicted labels for the input texts.

    Examples:
        # Using OpenRouter (default)
        predictions = llm_classifier(
            'gemini-flash-lite',
            texts,
            system_prompt="You are an expert classifier.",
            prompt_template="Classify: {text}\nLabel:"
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

    # Call llm_generate with provider parameters
    predicted_labels = llm_generate(
        model_name,
        user_prompts,
        system_prompt=system_prompt,
        temperature=temperature,
        estimate_cost=estimate_cost,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )

    return predicted_labels
from introdl.nlp import llm_generate

def llm_classifier( llm_config, 
                    texts, 
                    system_prompt,
                    prompt_template,
                    batch_size=1,
                    estimate_cost=False,
                    rate_limit=None):
    """
    Classify text using a Large Language Model (LLM).

    Args:
        llm_config (ModelConfig): Configuration for the LLM.
        texts (list of str): List of text documents to classify.
        system_prompt (str): System prompt to guide the LLM.
        prompt_template (str): Template for user prompts to classify each text.
        batch_size (int): Number of texts to process in a batch for local models.
        estimate_cost (bool): Whether to estimate the cost of the LLM request for API models.
        rate_limit (int): Rate limit per minutes for API requests to avoid overloading the LLM service.

    Returns:
        list of str: Predicted labels for the input texts.
    """

    user_prompts = [prompt_template.format(text=text) for text in texts]
    predicted_labels = llm_generate(llm_config, 
                                    user_prompts, 
                                    system_prompt=system_prompt,
                                    search_strategy='deterministic',
                                    batch_size=batch_size,
                                    estimate_cost=estimate_cost,
                                    rate_limit=rate_limit)

    return predicted_labels
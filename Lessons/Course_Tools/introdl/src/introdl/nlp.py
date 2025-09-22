import os
import gc
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import ipywidgets as widgets
import pyperclip
from IPython.display import display, clear_output, Markdown, HTML
from accelerate import cpu_offload
import contextlib
import time
from tqdm.autonotebook import tqdm
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers.utils import logging
logging.set_verbosity_error()

# Suppress logging warnings from PyTorch and Transformers
import logging
logging.getLogger("transformers.models.mistral.modeling_mistral").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Disable CUDA memory caching warning
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not compiled with flash attention.*")

# Global caches for Hugging Face models and tokenizers.
LOADED_MODELS = {}
LOADED_TOKENIZERS = {}

# --- Available Model Definitions ---
SHORT_NAME_MAP = {
    "llama-3p2-3B": "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
    "llama-3p1-8B": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "mistral-7B": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "qwen-2p5-3B": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    "qwen-2p5-7B": 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit',
    "gemini-flash-lite": "gemini-2.0-flash-lite-preview-02-05",
    "gemini-flash": "gemini-2.0-flash",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}

OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini"]
GEMINI_MODELS = ["gemini-2.0-flash-lite-preview-02-05", "gemini-2.0-flash"]

def llm_list_models(verbose=True):
    """
    Lists all available models grouped by category:
    - Short Name Models (Hugging Face & Gemini)
    - OpenAI Models
    - Gemini Models
    """
    if verbose:
        print("Available models:")
        for short_name in SHORT_NAME_MAP.keys():
            if short_name in OPENAI_MODELS:
                print(f" {short_name} => needs OPENAI_API_KEY")
            elif short_name.startswith('gemini'):
                print(f" {short_name} => needs GEMINI_API_KEY")
            else:
                print(f" {short_name} => HuggingFace: {SHORT_NAME_MAP[short_name]}")
    models = zip(range(len(SHORT_NAME_MAP)), SHORT_NAME_MAP.keys())
    return models

def release_model():
    """Unloads the currently loaded Hugging Face model from memory and GPU."""
    global LOADED_MODELS, LOADED_TOKENIZERS
    if LOADED_MODELS:
        model_str = next(iter(LOADED_MODELS))
        print(f"🛑 Unloading model: {model_str} from GPU...")
        model = LOADED_MODELS[model_str]
        try:
            if hasattr(model, "hf_device_map"):
                cpu_offload(model)
            else:
                print("⬇️ Moving model to CPU manually...")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="You shouldn't move a model that is dispatched using accelerate hooks")
                    model.to("cpu")
        except Exception as e:
            print(f"⚠️ Error moving model to CPU: {e}")
        del LOADED_MODELS[model_str]
        del LOADED_TOKENIZERS[model_str]
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
        print(f"✅ Model {model_str} has been fully unloaded.")
    else:
        print("❌ No model to unload.")

class ModelConfig:
    """
    Configuration object to store model, tokenizer, and API client settings.
    
    When using API-based models (e.g. OpenAI, Gemini, Together, Groq, or OpenRouter), 
    the model and tokenizer may be None.
    """
    def __init__(self, model_str, model=None, tokenizer=None, api_type=None,
                 cost_per_M_input=None, cost_per_M_output=None):
        self.model_str = model_str
        self.model = model
        self.tokenizer = tokenizer
        self.api_type = api_type  # Expected values: "openai", "gemini", "together", "groq", "openrouter" or None
        self.client = None
        self.cost_per_M_input = cost_per_M_input
        self.cost_per_M_output = cost_per_M_output

        # Initialize API client if applicable
        if api_type in ["openai", "gemini", "together", "groq", "openrouter"]:
            if api_type == "together":
                api_key = os.getenv("TOGETHER_API_KEY")
                base_url = "https://api.together.xyz/v1"
            elif api_type == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                base_url = "https://api.groq.com/openai/v1"
            elif api_type == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY")
                base_url = "https://openrouter.ai/api/v1"
            else:  # openai or gemini
                api_key = os.getenv("GEMINI_API_KEY" if api_type == "gemini" else "OPENAI_API_KEY")
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/" if api_type == "gemini" else None

            if not api_key:
                raise ValueError(f"Missing {api_type.upper()} API key. Set the appropriate environment variable.")

            self.client = OpenAI(api_key=api_key, base_url=base_url)

# === REMOVED_FUNCTION_START: unload ===
#     def unload(self):
#         release_model()
# === REMOVED_FUNCTION_END: unload ===



def llm_configure(model_str, cost_per_M_input=None, cost_per_M_output=None,
                  llm_provider=None):
    """
    Configures the model based on the provided model_str and optional llm_provider.
    
    If llm_provider is None:
      - If the expanded model_str is in OPENAI_MODELS or GEMINI_MODELS, an API-based configuration is returned.
      - Otherwise, a local Hugging Face model is loaded (with caching).
    
    If llm_provider is provided, it must be one of "openai", "gemini", "together", "groq", or "openrouter",
    and the corresponding API is used regardless of the model name.
    
    Parameters:
      model_str (str): A model name string (can be a short alias or full model name).
      cost_per_M_input (float, optional): Cost per million input tokens (for API-based models).
      cost_per_M_output (float, optional): Cost per million output tokens (for API-based models).
      llm_provider (str, optional): Either None (to infer provider) or one of the API providers.
    
    Returns:
      ModelConfig: The configuration object with model, tokenizer, and API client settings.
    """
    model_str = model_str.strip()
    valid_api_providers = ["openai", "gemini", "together", "groq", "openrouter"]

    # Expand short names
    if model_str in SHORT_NAME_MAP:
        model_str = SHORT_NAME_MAP[model_str]

    # If llm_provider is explicitly provided
    if llm_provider is not None:
        if llm_provider.lower() not in valid_api_providers:
            raise ValueError("llm_provider must be either None or one of 'openai', 'gemini', 'together', 'groq', or 'openrouter'.")
        return ModelConfig(model_str, api_type=llm_provider.lower(),
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)

    # Otherwise infer from model_str
    if model_str in OPENAI_MODELS:
        return ModelConfig(model_str, api_type="openai",
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)
    if model_str in GEMINI_MODELS:
        return ModelConfig(model_str, api_type="gemini",
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)

    # Otherwise, load locally
    global LOADED_MODELS, LOADED_TOKENIZERS
    if model_str in LOADED_MODELS:
        return ModelConfig(model_str, LOADED_MODELS[model_str], LOADED_TOKENIZERS[model_str],
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)
    elif LOADED_MODELS:
        release_model()
        torch.cuda.empty_cache()
        gc.collect()

    print(f"🚀 Loading model: {model_str} (this may take a while)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_str,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        LOADED_MODELS[model_str] = model
        LOADED_TOKENIZERS[model_str] = tokenizer
        print(f"🟢 Model {model_str} loaded successfully.\n")
        return ModelConfig(model_str, model, tokenizer,
                           cost_per_M_input=cost_per_M_input, cost_per_M_output=cost_per_M_output)
    except Exception as e:
        print(f"❌ Error loading model {model_str}: {str(e)}")
        return None

import re

def clean_response(response, prompt=None, cleaning_mode="generic", split_string=None):
    """
    Cleans the response by removing unwanted text.

    If split_string is provided (not None), the function searches for split_string in the response
    and removes everything up to and including any whitespace or new lines immediately after it,
    returning the remainder.

    If split_string is None, the function applies the original cleaning methods:
      - In "qwen" mode, it uses the last 10 characters of the prompt as a marker and removes all text
        up to and including the line immediately following that marker.
      - In generic mode, it removes the input prompt, an "assistant" label, and any extraneous blank lines.
    """
    if split_string is not None:
        pos = response.find(split_string)
        if pos != -1:
            pos_end = pos + len(split_string)
            # Skip any whitespace or newlines immediately after split_string.
            while pos_end < len(response) and response[pos_end] in " \t\n\r":
                pos_end += 1
            return response[pos_end:].strip()
        else:
            return response.strip()

    if cleaning_mode == "qwen" and prompt:
        # Use only the last 10 characters of the prompt as the marker.
        marker = prompt[-10:]
        pos = response.find(marker)
        if pos != -1:
            # Find the end of the line containing the marker.
            end_line = response.find("\n", pos)
            if end_line != -1:
                # Remove up to and including the next line.
                next_line_end = response.find("\n", end_line + 1)
                if next_line_end != -1:
                    response = response[next_line_end+1:].strip()
                else:
                    response = response[end_line+1:].strip()
            else:
                response = response[pos+len(marker):].strip()
        else:
            # Fallback: if the marker isn't found, remove the first line.
            lines = response.splitlines()
            if len(lines) > 1:
                response = "\n".join(lines[1:]).strip()
            else:
                response = response.strip()
    else:
        if prompt:
            # --- Original prompt removal logic ---
            prompt_marker_pattern = re.escape(prompt) + r"\s*\.\s*assistant"
            match = re.search(prompt_marker_pattern, response)
            if match:
                response = response[match.end():].strip()
            else:
                match = re.search(re.escape(prompt), response)
                if match:
                    response = response[match.end():].strip()

            # --- Additional marker removal ---
            extra_marker = prompt[-10:] + "assistant"
            pattern = ""
            for char in extra_marker:
                if char.isspace():
                    pattern += r"\s+"
                else:
                    pattern += re.escape(char) + r"\s*"
            match = re.search(pattern, response, flags=re.IGNORECASE)
            if match:
                response = response[match.end():].strip()

    # Finally, remove any lines that consist solely of the word "assistant" (ignoring case)
    response = re.sub(r"^\s*assistant\s*", "", response, flags=re.IGNORECASE).strip()
    response = "\n".join([line for line in response.split("\n") if line.strip()])
    return response

def configure_gen_kwargs(strategy, top_k, top_p, num_beams, temperature, max_new_tokens):
    """
    Returns a dictionary of generation kwargs based on the search strategy.
    """
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if strategy == "top_k":
        gen_kwargs.update({"do_sample": True, "top_k": top_k, "temperature": temperature})
    elif strategy == "top_p":
        gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})
    elif strategy == "contrastive":
        gen_kwargs.update({"do_sample": True, "penalty_alpha": 0.6, "top_k": 4, "temperature": temperature})
    elif strategy == "beam_search":
        gen_kwargs.update({"do_sample": False, "num_beams": num_beams, "temperature": temperature})
    elif strategy == "deterministic":
        gen_kwargs.update({"do_sample": False, "num_beams": 1, "top_p": 1.0, "temperature": None, "top_k": None})
    else:
        gen_kwargs.update({"do_sample": True, "top_p": top_p, "temperature": temperature})
    return gen_kwargs

def generate_api_text(model_config, prompt_list, system_prompt, assistant_prompt,
                      search_strategy, max_new_tokens, temperature, top_p,
                      rate_limit, estimate_cost, remove_input_prompt, split_string):
    """
    Generates text using an API-based model.
    """
    responses = []
    num_input_tokens = 0.0
    num_output_tokens = 0.0
    messages = [{"role": "system", "content": system_prompt}]
    if assistant_prompt is not None:
        messages.append({"role": "assistant", "content": assistant_prompt})
    
    for prompt in prompt_list:
        user_message = {"role": "user", "content": prompt}
        full_messages = messages + [user_message]
        openai_params = {
            "model": model_config.model_str,
            "messages": full_messages,
            "max_tokens": max_new_tokens
        }
        if search_strategy == "deterministic":
            openai_params["temperature"] = 0.0
            openai_params["top_p"] = 1.0
        else:
            openai_params["temperature"] = temperature
            openai_params["top_p"] = top_p
        
        response = model_config.client.chat.completions.create(**openai_params)
        response_text = response.choices[0].message.content.strip()
        responses.append(clean_response(response_text, prompt, split_string=split_string) if remove_input_prompt else response_text)
        
        if rate_limit is not None:
            time.sleep(60.0 / rate_limit)
        if estimate_cost:
            num_input_tokens += response.usage.prompt_tokens
            num_output_tokens += response.usage.completion_tokens
    
    if estimate_cost:
        if model_config.cost_per_M_input is not None and model_config.cost_per_M_output is not None:
            total_cost = ((num_input_tokens / 1_000_000) * model_config.cost_per_M_input) + \
                         ((num_output_tokens / 1_000_000) * model_config.cost_per_M_output)
            print(f"💰 Estimated Cost: ${total_cost:.6f} (Input: {num_input_tokens} tokens, Output: {num_output_tokens} tokens)")
        else:
            print("⚠️ Cost parameters not set. Cannot estimate cost.")
            print(f"💰 Token Usage: Input: {num_input_tokens} tokens, Output: {num_output_tokens} tokens")
    
    return responses


def generate_local_text(model_config, prompt_list, system_prompt, assistant_prompt,
                        search_strategy, max_new_tokens, temperature, top_p,
                        num_beams, top_k, remove_input_prompt, batch_size, disable_tqdm, split_string):
    """
    Generates text using a local Hugging Face model.
    Batches the inputs and uses a tqdm progress bar if there is more than one prompt.
    """
    tokenizer = model_config.tokenizer
    model = model_config.model

    if model is None or tokenizer is None:
        return "❌ Error: Model or tokenizer is not properly initialized."
    
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    
    def process_batch(batch_prompts):
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            conversations = []
            for p in batch_prompts:
                conv = [{"role": "system", "content": system_prompt}]
                if assistant_prompt is not None:
                    conv.append({"role": "assistant", "content": assistant_prompt})
                conv.append({"role": "user", "content": p})
                conversations.append(conv)
            input_ids = tokenizer.apply_chat_template(conversations, return_tensors="pt",
                                                      padding=True, truncation=True).to(model.device)
        else:
            new_prompts = []
            for p in batch_prompts:
                prompt_parts = [system_prompt]
                if assistant_prompt is not None:
                    prompt_parts.append("Assistant: " + assistant_prompt)
                prompt_parts.append("User: " + p)
                new_prompts.append("\n".join(prompt_parts))
            input_ids = tokenizer(new_prompts, return_tensors="pt",
                                  padding=True, truncation=True).to(model.device)
        
        terminators = [token for token in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                       if token is not None]
        
        gen_kwargs = configure_gen_kwargs(search_strategy, top_k, top_p, num_beams, temperature, max_new_tokens)
        gen_kwargs.update({"repetition_penalty": 1.2, "eos_token_id": terminators})
        
        with torch.no_grad():
            output = model.generate(input_ids, **gen_kwargs)
        batch_responses = tokenizer.batch_decode(output, skip_special_tokens=True)
        if remove_input_prompt:
            cleaning_mode = "qwen" if "qwen" in model_config.model_str.lower() else "generic"
            return [clean_response(resp, prompt, cleaning_mode=cleaning_mode, split_string=split_string)
                    for resp, prompt in zip(batch_responses, batch_prompts)]
        else:
            return batch_responses

    responses = []
    use_tqdm = (len(prompt_list) > 1) and (not disable_tqdm)
    if use_tqdm:
        from tqdm.autonotebook import tqdm
        pbar = tqdm(total=len(prompt_list), desc="Local Generation", leave=True, dynamic_ncols=True, file=sys.stdout)
    else:
        pbar = contextlib.nullcontext()
    
    with pbar:
        for i in range(0, len(prompt_list), batch_size):
            batch_prompts = prompt_list[i:i + batch_size]
            responses.extend(process_batch(batch_prompts))
            if use_tqdm:
                pbar.update(len(batch_prompts))
    
    return responses


def llm_generate(model_config, prompts, 
                 batch_size=1, assistant_prompt=None, estimate_cost=False, max_new_tokens=200, 
                 remove_input_prompt=True, search_strategy="top_p", top_k=50, top_p=0.9, num_beams=1, temperature=0.7,
                 system_prompt="You are an AI assistant that provides brief answers. Do not include the input prompt in your answer.",
                 rate_limit=None,
                 disable_tqdm=False,
                 split_string=None):
    """
    Generates responses from a language model.
    Dispatches API-based or local text generation.
    Disables tqdm if there is only one prompt.
    """
    if model_config is None:
        return "❌ Error: Invalid model configuration. Please check the model name."
    
    is_batch = isinstance(prompts, list)
    prompt_list = prompts if is_batch else [prompts]
    if len(prompt_list) == 1:
        disable_tqdm = True
    
    if model_config.api_type in ["openai", "gemini", "together", "groq", "openrouter"]:
        try:
            responses = generate_api_text(model_config, prompt_list, system_prompt, assistant_prompt,
                                          search_strategy, max_new_tokens, temperature, top_p,
                                          rate_limit, estimate_cost, remove_input_prompt, split_string)
        except Exception as e:
            return f"{model_config.api_type.capitalize()} API error: {str(e)}"
    else:
        responses = generate_local_text(model_config, prompt_list, system_prompt, assistant_prompt,
                                        search_strategy, max_new_tokens, temperature, top_p,
                                        num_beams, top_k, remove_input_prompt, batch_size, disable_tqdm, split_string)
    
    return responses if is_batch else responses[0]

def generate_text(prompts, 
                  model_name='llama-3p2-3B', 
                  llm_provider=None, 
                  system_prompt="You are an AI assistant that provides brief answers. Do not include the input prompt in your answer.",
                  assistant_prompt=None,
                  batch_size=1,
                  cost_per_M_input=None, 
                  cost_per_M_output=None,
                  disable_tqdm=False,
                  estimate_cost=False,
                  max_new_tokens=200,
                  num_beams=1,
                  rate_limit=None,
                  remove_input_prompt=True,
                  search_strategy="top_p",
                  split_string=None,
                  temperature=0.7,
                  top_k=50,
                  top_p=0.9):
    """
    Generates responses from a language model given a prompt or list of prompts.
    
    This high-level interface accepts a model name (or short name) along with an optional
    llm_provider. If llm_provider is None, then the model is loaded either locally (if not an API model)
    or using the default API (if the model is one of "gpt-4o", "gpt-4o-mini", "gemini-flash", etc.).
    If llm_provider is specified (e.g. "openai" or "gemini"), the code uses that API.
    
    Parameters:
      prompts (str or list of str): The input prompt(s) for text generation.
      model_name (str, optional): Full or short model name. Defaults to 'llama3p2-3B'.
      llm_provider (str, optional): Provider for the model. If None, a local model is loaded or 
          the default API is used if the model name indicates an API model. Otherwise, must be one of 
          "openai", "gemini", "together", or "groq".
      system_prompt (str, optional): The system-level instruction prompt.
      assistant_prompt (str, optional): An optional previous assistant message.
      batch_size (int, optional): Number of prompts processed per batch. Defaults to 1.
      cost_per_M_input (float, optional): Cost per million input tokens (for API-based models).
      cost_per_M_output (float, optional): Cost per million output tokens (for API-based models).
      disable_tqdm (bool, optional): If True, disables progress bars. Defaults to False.
      estimate_cost (bool, optional): If True, prints estimated token cost when available.
      max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
      num_beams (int, optional): Number of beams for beam search. Defaults to 1.
      rate_limit (float, optional): Maximum API requests per minute (if applicable).
      remove_input_prompt (bool, optional): If True, removes the input prompt from the generated response.
      search_strategy (str, optional): Decoding strategy; options include "top_k", "top_p", "contrastive", 
          "beam_search", and "deterministic". Defaults to "top_p".
      temperature (float, optional): Sampling temperature. Defaults to 0.7.
      top_k (int, optional): Number of top tokens to consider for "top_k" sampling. Defaults to 50.
      top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.9.
    
    Returns:
      str or list of str: The generated response(s) as a string or a list of strings.
    """
    # Expand short names if necessary.
    if model_name in SHORT_NAME_MAP:
        model_str = SHORT_NAME_MAP[model_name]
    else:
        model_str = model_name.strip()
    
    # Use llm_configure to obtain the ModelConfig object.
    config = llm_configure(model_str, cost_per_M_input=cost_per_M_input, 
                           cost_per_M_output=cost_per_M_output, llm_provider=llm_provider)
    if config is None:
        return f"❌ Error: Unable to configure model {model_str}."
    
    # Dispatch to legacy llm_generate.
    return llm_generate(config, prompts, batch_size=batch_size, assistant_prompt=assistant_prompt,
                        estimate_cost=estimate_cost, max_new_tokens=max_new_tokens,
                        remove_input_prompt=remove_input_prompt, search_strategy=search_strategy,
                        top_k=top_k, top_p=top_p, num_beams=num_beams, temperature=temperature,
                        system_prompt=system_prompt, rate_limit=rate_limit, disable_tqdm=disable_tqdm,
                        split_string=split_string)



def clear_pipeline(pipe, verbosity=0):
    """Clears a Hugging Face pipeline and frees CUDA memory."""
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
    model = pipe.model
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.name_or_path}, Size: {model_size:,} parameters")

def display_markdown(response):
    if not response or not isinstance(response, str):
        display(Markdown("⚠️ *No valid response to display.*"))
        return
    display(Markdown(response))

# === REMOVED_FUNCTION_START: JupyterChat ===
# class JupyterChat:
#     def __init__(self, model_str, system_prompt="You are a helpful assistant."):
#         self.model_config = llm_configure(model_str)
#         if not self.model_config:
#             raise ValueError(f"Could not load model: {model_str}")
        
#         self.system_prompt = system_prompt
#         self.chat_history = []
        
#         self.text_input = widgets.Text(
#             placeholder="Type your message here...",
#             description="User:",
#             layout=widgets.Layout(width="100%")
#         )
#         self.send_button = widgets.Button(description="Send", button_style='primary')
#         self.clear_button = widgets.Button(description="Clear Chat", button_style='warning')
#         self.copy_button = widgets.Button(description="Copy", button_style='success')
#         self.output_area = widgets.Output()

#         self.send_button.on_click(self.handle_input)
#         self.text_input.on_submit(self.handle_input)
#         self.clear_button.on_click(self.clear_chat)
#         self.copy_button.on_click(self.copy_chat)

#         button_box = widgets.HBox([self.send_button, self.clear_button, self.copy_button])
#         display(self.text_input, button_box, self.output_area)

# # === REMOVED_FUNCTION_START: handle_input ===
# #     def handle_input(self, event):
# #         user_input = self.text_input.value.strip()
# #         if not user_input:
# #             return
# #         self.text_input.value = ""
# #         self.chat_history.append({"role": "user", "content": user_input})
# #         response = self.generate_response(user_input)
# #         self.chat_history.append({"role": "assistant", "content": response})
# #         self.update_display()
# # === REMOVED_FUNCTION_END: handle_input ===


# # === REMOVED_FUNCTION_START: generate_response ===
# #     def generate_response(self, user_input):
# #         try:
# #             response = llm_generate(self.model_config, prompts=user_input, system_prompt=self.system_prompt)
# #             return response
# #         except Exception as e:
# #             return self.format_api_error(e)
# # === REMOVED_FUNCTION_END: generate_response ===


#     def update_display(self):
#         with self.output_area:
#             clear_output()
#             for entry in self.chat_history:
#                 role = "**User:**" if entry["role"] == "user" else "**Assistant:**"
#                 message = f"{role}\n\n{entry['content']}"
#                 display_markdown(message)

# # === REMOVED_FUNCTION_START: clear_chat ===
# #     def clear_chat(self, event):
# #         self.chat_history = []
# #         with self.output_area:
# #             clear_output()
# #         print("Chat history cleared.")
# # === REMOVED_FUNCTION_END: clear_chat ===


# # === REMOVED_FUNCTION_START: copy_chat ===
# #     def copy_chat(self, event):
# #         chat_text = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in self.chat_history])
# #         pyperclip.copy(chat_text)
# #         print("✅ Chat copied to clipboard!")
# === REMOVED_FUNCTION_END: JupyterChat ===

# === REMOVED_FUNCTION_END: copy_chat ===


# === REMOVED_FUNCTION_START: format_api_error ===
#     def format_api_error(self, error):
#         error_str = str(error)
#         if "429" in error_str:
#             return "⚠️ *API limit reached. Please check your quota.*"
#         elif "401" in error_str:
#             return "⚠️ *Invalid API key. Please verify your credentials.*"
#         elif "RESOURCE_EXHAUSTED" in error_str:
#             return "⚠️ *Quota exceeded. Try again later.*"
#         return f"⚠️ *API Error:* {error_str}"
# === REMOVED_FUNCTION_END: format_api_error ===


def model_report(model, tokenizer):
    # Model details
    model_name = model.config.model_type
    vocab_size = model.config.vocab_size
    embedding_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 'Unknown'
    num_transformer_blocks = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else 'Unknown'
    num_parameters = sum(p.numel() for p in model.parameters())

    # Print report
    print(f"Model Report for {model_name}")
    print("---------------------------")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Embedding Dimension: {embedding_dim}")
    print(f"Number of Transformer Blocks: {num_transformer_blocks}")
    print(f"Total Number of Parameters: {num_parameters:,}")


def generate_top_k_table(model, tokenizer, prompt, top_k=5, temperature=1.0):
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenize input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Generate logits for the prompt
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # Only consider the logits for the last token

    # Apply temperature scaling
    logits = logits / temperature

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Get the top_k choices
    top_probs, top_indices = torch.topk(probabilities, top_k)

    # Build a dictionary of the top_k words and their probabilities
    top_k_words = {}
    for i in range(top_k):
        word = tokenizer.decode(top_indices[0, i])
        prob = top_probs[0, i].item() * 100  # Convert to percentage
        top_k_words[word] = prob

    # Convert dictionary to a DataFrame
    df = pd.DataFrame(list(top_k_words.items()), columns=["Word", "Probability (%)"])
    df.sort_values(by="Probability (%)", ascending=False, inplace=True)
    
    return df


def generate_greedy_decoding_table(model, tokenizer, prompt, num_choices=5, max_length=10):
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenize input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_text = prompt
    rows = []  # Store rows for the DataFrame

    # Generate words step-by-step
    for step in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)

            # Get the top N choices
            top_probs, top_indices = torch.topk(probabilities, num_choices)

            choices = []
            for i in range(num_choices):
                word = tokenizer.decode(top_indices[0, i])
                prob = top_probs[0, i].item() * 100  # Convert to percentage
                choices.append(f"{word} ({prob:.2f}%)")

            # Append the row with prompt and choices
            row = [generated_text] + choices
            rows.append(row)

            # Update the input with the most probable word (greedy decoding)
            input_ids = torch.cat([input_ids, top_indices[:, 0].unsqueeze(0)], dim=-1)
            generated_text += tokenizer.decode(top_indices[0, 0])

    # Create and return a DataFrame
    columns = ["Input"] + [f"Choice {i+1}" for i in range(num_choices)]
    df = pd.DataFrame(rows, columns=columns)
    pd.set_option('display.max_colwidth', None)  # Ensure full text is displayed in the DataFrame
    return df


def generate_detailed_beam_search(model, tokenizer, prompt, num_beams=2, max_length=5, num_choices=3, temperature=1.0):
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Initialize beams
    beams = [(input_ids, prompt, 0.0)]  # (input_ids, text_so_far, cumulative_log_prob)
    
    for step in range(max_length):
        candidates = []
        initial_display = []

        # For each current beam, generate the top-k possible continuations
        for beam_index, (input_ids, generated_text, cumulative_log_prob) in enumerate(beams):
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :] / temperature
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get num_choices predictions
                top_probs, top_indices = torch.topk(probabilities, num_choices)
                
                # Prepare row for this beam
                row = {"Beam": generated_text}
                
                for i in range(num_choices):
                    new_token_id = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_word = tokenizer.decode(new_token_id[0])
                    prob = top_probs[0, i].item()  # Keep as original probability
                    
                    row[f"Completion {i+1}"] = f"{new_word} ({prob:.4f})"
                    
                    # Calculate the new cumulative log probability
                    new_log_prob = torch.log(top_probs[0, i]).item()
                    new_log_prob_sum = cumulative_log_prob + new_log_prob
                    
                    # Form the new candidate beam (retain original order)
                    new_input_ids = torch.cat([input_ids, new_token_id], dim=-1)
                    new_text = generated_text + new_word
                    candidates.append((new_input_ids, new_text, cumulative_log_prob, new_word, new_log_prob, new_log_prob_sum))
                
                initial_display.append(row)
        
        # Display Initial DataFrame with beams and their top-k completions
        clear_output(wait=True)
        print(f"\n=== Step {step + 1}: Top {num_choices} Completions for Each Beam ===")
        
        initial_df = pd.DataFrame(initial_display)
        display(initial_df)
        
        # Keep the candidates in the original order of generation
        step_data = []
        for idx, (new_input_ids, new_text, prev_log_prob_sum, new_word, new_log_prob, new_log_prob_sum) in enumerate(candidates):
            step_data.append({
                "Beam": idx + 1,
                "Text": new_text,
                "Previous Log Prob Sum": prev_log_prob_sum,
                "Log Prob of New Word": new_log_prob,
                "New Log Prob Sum": new_log_prob_sum
            })
        
        # Create DataFrame without sorting
        df = pd.DataFrame(step_data)

        # Find indices of the top two rows by New Log Prob Sum for highlighting
        top_two_indices = df['New Log Prob Sum'].nlargest(2).index.tolist()
        
        def highlight_top_rows(row):
            if row.name in top_two_indices:
                # Pale green background with black text
                return ['background-color: #ecf8ec; color: black;'] * len(row)
            else:
                return [''] * len(row)

        styled_df = df.style.apply(highlight_top_rows, axis=1)
        
        print(f"\n=== Step {step + 1}: Expanded Beams (Unsorted) ===")
        display(styled_df)
        
        # Wait for the user to press Enter
        input("Press Enter to proceed to the next step...")
        
        # Select the top `num_beams` beams for the next step, preserving structure
        beams = [(candidates[i][0], candidates[i][1], candidates[i][5]) for i in top_two_indices[:num_beams]]


def generate_top_k_sampling(model, tokenizer, prompt, max_length=5, top_k=5, temperature=1.0):
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    generated_text = prompt
    
    for step in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1).squeeze()
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Original Probabilities Display
            original_probs = top_probs.cpu().numpy()
            original_tokens = [tokenizer.decode([index]) for index in top_indices]
            original_display = {f"Completion {i+1}": f"{token} ({original_probs[i]:.4f})" for i, token in enumerate(original_tokens)}
            
            # Renormalize probabilities for sampling
            renormalized_probs = original_probs / original_probs.sum()
            renormalized_display = {f"Completion {i+1}": f"{original_tokens[i]} ({renormalized_probs[i]:.4f})" for i in range(top_k)}
            
            # Choose a random token from the top-k
            selected_index = np.random.choice(np.arange(top_k), p=renormalized_probs)
            selected_token = original_tokens[selected_index]
            selected_prob = renormalized_probs[selected_index]
            
            # Create DataFrame for display
            display_data = [
                {"Step": "Original Probabilities", **original_display},
                {"Step": "Renormalized Probabilities", **renormalized_display},
                {"Step": "Selected Completion", **{f"Completion {i+1}": f"{token}" if i != selected_index else f"{token}" for i, token in enumerate(original_tokens)}}
            ]
            
            df = pd.DataFrame(display_data)
            
            # Highlight the selected row
            def highlight_row(row):
                if row.name == 2:  # Selected completion row
                    styles = []
                    for col in df.columns:
                        if col == f"Completion {selected_index + 1}":
                            styles.append('background-color: #ecf8ec; color: black;')
                        else:
                            styles.append('')
                    return styles
                else:
                    return [''] * len(row)
            
            styled_df = df.style.apply(highlight_row, axis=1)
            
            # Display DataFrame
            clear_output(wait=True)
            print(f"\n=== Step {step + 1} ===")
            print(f"Current Sequence: {generated_text}")
            display(styled_df)
            
            # Update the input sequence with the selected token
            new_token_id = top_indices[selected_index].unsqueeze(0).unsqueeze(0).to(device)
            input_ids = torch.cat([input_ids, new_token_id], dim=-1)
            generated_text += selected_token
            
            # Wait for user input before proceeding to the next step
            input("Press Enter to proceed to the next step...")


def generate_top_p_sampling(model, tokenizer, prompt, max_length=5, top_p=0.9, temperature=1.0):
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    generated_text = prompt
    
    for step in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1).squeeze()
            
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            
            # Cumulative sum of probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            
            # Select tokens until cumulative probability exceeds top_p (e.g., 0.9)
            cutoff_index = torch.where(cumulative_probs > top_p)[0][0] + 1
            top_probs = sorted_probs[:cutoff_index]
            top_indices = sorted_indices[:cutoff_index]
            
            # Limit to a maximum of 8 completions and pad with blank cells if necessary
            max_completions = 8
            top_probs = top_probs[:max_completions]
            top_indices = top_indices[:max_completions]
            
            if len(top_probs) < max_completions:
                padding_length = max_completions - len(top_probs)
                
                # Move padding tensors to the same device
                top_probs = torch.cat([top_probs, torch.zeros(padding_length, device=device)])
                top_indices = torch.cat([top_indices, torch.full((padding_length,), -1, device=device)])

            # Original Probabilities Display
            original_probs = top_probs.cpu().numpy()
            original_tokens = [tokenizer.decode([index]) if index != -1 else "" for index in top_indices]
            original_display = {f"Completion {i+1}": f"{token} ({original_probs[i]:.4f})" if token else "" 
                                for i, token in enumerate(original_tokens)}
            
            # Renormalize probabilities for sampling
            valid_probs = top_probs[top_probs > 0]
            renormalized_probs = valid_probs / valid_probs.sum()
            
            renormalized_display = {f"Completion {i+1}": f"{original_tokens[i]} ({renormalized_probs[i]:.4f})" 
                                    if original_tokens[i] else "" 
                                    for i in range(len(original_tokens))}
            
            # Choose a random token from the top-p set
            if len(renormalized_probs) > 0:
                selected_index = np.random.choice(np.arange(len(renormalized_probs)), p=renormalized_probs.cpu().numpy())
            else:
                selected_index = 0  # In case of an error, fallback to the first token
            
            selected_token = original_tokens[selected_index]
            
            # Create DataFrame for display
            display_data = [
                {"Step": "Original Probabilities", **original_display},
                {"Step": "Renormalized Probabilities", **renormalized_display},
                {"Step": "Selected Completion", **{f"Completion {i+1}": f"{token}" if i != selected_index else f"{token}" 
                                                  for i, token in enumerate(original_tokens)}}
            ]
            
            df = pd.DataFrame(display_data)
            
            # Highlight the selected row
            def highlight_row(row):
                if row.name == 2:  # Selected completion row
                    styles = []
                    for col in df.columns:
                        if col == f"Completion {selected_index + 1}":
                            styles.append('background-color: #ecf8ec; color: black;')
                        else:
                            styles.append('')
                    return styles
                else:
                    return [''] * len(row)
            
            styled_df = df.style.apply(highlight_row, axis=1)
            
            # Display DataFrame
            clear_output(wait=True)
            print(f"\n=== Step {step + 1} ===")
            print(f"Current Sequence: {generated_text}")
            display(styled_df)
            
            # Update the input sequence with the selected token if it's valid
            if selected_token:
                new_token_id = top_indices[selected_index].unsqueeze(0).unsqueeze(0).to(device)
                input_ids = torch.cat([input_ids, new_token_id], dim=-1)
                generated_text += selected_token
            
            # Wait for user input before proceeding to the next step
            input("Press Enter to proceed to the next step...")


def plot_top_k_distribution(model, tokenizer, prompt, top_k=10, plot_type='pdf', 
                            temperature=1.0, token_axis='horizontal', figsize=(10, 6)):
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tokenize input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Generate logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :] / temperature  # Only consider the last token's logits
    
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1).squeeze()
    
    # Get the top-k tokens
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Decode the top-k tokens
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices.cpu().numpy()]
    
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)  # Added figsize argument

    if plot_type == 'pdf':
        values = top_probs.cpu().numpy()
        title = f'Top-{top_k} Probability Distribution'
    elif plot_type == 'cdf':
        values = np.cumsum(top_probs.cpu().numpy())
        title = f'Top-{top_k} Cumulative Distribution'
    else:
        raise ValueError("Invalid plot_type. Choose 'pdf' or 'cdf'.")
    
    if token_axis == 'vertical':
        # Vertical plot (tokens on y-axis)
        ax.barh(top_tokens, values, color='skyblue')
        ax.set_xlabel('Probability' if plot_type == 'pdf' else 'Cumulative Probability')
        ax.set_title(title)
        plt.gca().invert_yaxis()  # Highest probability on top
        
    elif token_axis == 'horizontal':
        # Horizontal plot (tokens on x-axis)
        ax.bar(top_tokens, values, color='skyblue')
        ax.set_ylabel('Probability' if plot_type == 'pdf' else 'Cumulative Probability')
        ax.set_title(title)
        ax.set_xticks(np.arange(len(top_tokens)))
        ax.set_xticklabels(top_tokens, rotation=45, ha='right')
        
    else:
        raise ValueError("Invalid token_axis. Choose 'horizontal' or 'vertical'.")
    
    plt.show()


def visualize_conversation(conversation):
    """
    Display the conversation history using Markdown formatting.
    """
    md_output = "# Conversation Flow\n"
    for message in conversation:
        if message["role"] == "system":
            md_output += f"### **System:** {message['content']}\n\n"
        elif message["role"] == "user":
            md_output += f"**User:** {message['content']}\n\n"
        elif message["role"] == "assistant":
            md_output += f"**Assistant:** {message['content']}\n\n"
    clear_output(wait=True)
    display(Markdown(md_output))

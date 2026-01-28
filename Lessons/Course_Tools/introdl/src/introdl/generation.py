import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    raise ImportError(
        "Failed to import transformers. This is usually caused by incompatible "
        "versions of transformers and accelerate. Fix with:\n"
        "    pip install --upgrade transformers accelerate"
    ) from e
import numpy as np
import pandas as pd
from IPython.display import display, HTML, clear_output, Markdown
import matplotlib.pyplot as plt


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

# Note: visualize_conversation has been moved to nlp.py (new signature with show_recent_only parameter)
# The old version here has been removed to avoid import conflicts
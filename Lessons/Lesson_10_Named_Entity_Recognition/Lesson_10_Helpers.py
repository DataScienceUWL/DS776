
from collections import Counter
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import re
import spacy
from spacy.tokens import Doc, Span
from spacy.util import filter_spans
from spacy import displacy
import torch


def generate_entity_colors(full_label_list, cmap_name="Set2", special_labels=None):
    """
    Generates a dictionary mapping entity labels (e.g., "B-PER", "I-LOC") to unique colors for visualization.
    This helps highlight different entity types using consistent, colorblind-friendly colors.

    Args:
        full_label_list (list of str): All possible entity labels (e.g., ["O", "B-PER", "I-PER", "B-LOC", ...]).
        cmap_name (str): Name of a matplotlib colormap to use for generating the colors (default: "Set2").
        special_labels (dict): Optional dictionary of labels to manually assign specific hex colors (e.g., {"IGNORE": "#cccccc"}).

    Returns:
        dict: A dictionary mapping each label to a hex color code (e.g., {"B-PER": "#56B4E9"}).
    """
    cmap = plt.get_cmap(cmap_name)
    colors = {}

    if special_labels is None:
        special_labels = {}

    for i, label in enumerate(sorted(full_label_list)):
        if label == "O":
            continue  # Skip "O" which means "Outside" any named entity
        rgba = cmap(i % cmap.N)
        rgb = tuple(int(255 * x) for x in rgba[:3])
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
        colors[label] = hex_color

    # Add any special manually specified colors
    for label, hex_color in special_labels.items():
        colors[label] = hex_color

    return colors

def display_ner_html(tokens, ner_tags, label_list,
                     entity_colors=None, cmap_name="Set2",
                     subword_label="IGNORE", subword_color="#cccccc",
                     aggregate=False):
    """
    Visualizes NER predictions using spaCy's displacy tool.

    Args:
        tokens (list of str): List of input tokens (e.g., words in a sentence).
        ner_tags (list of int): Corresponding label indices for each token. Subword tokens should have -100.
        label_list (list of str): Mapping from label indices to BIO tag strings.
        entity_colors (dict): Optional custom color mapping for entity labels.
        cmap_name (str): Name of colormap used to generate entity colors if none provided.
        subword_label (str): Label name used to mark subword tokens (default: "IGNORE").
        subword_color (str): Color used to highlight subword tokens (default: light gray).
        aggregate (bool): If True, aggregates B- and I- tags into single spans. If False (default),
                         displays B- and I- tags separately to show the BIO tagging structure.

    Returns:
        None: Displays the rendered HTML visualization inside a Jupyter notebook or IPython environment.
    """
    nlp = spacy.blank("en")
    spaces = [True] * (len(tokens) - 1) + [False]
    doc = Doc(nlp.vocab, words=tokens, spaces=spaces)

    # Convert integer tags to BIO tag strings, replacing -100 with a placeholder label
    bio_tags = [label_list[tag] if tag != -100 else subword_label for tag in ner_tags]

    spans = []

    if aggregate:
        # Original behavior: aggregate B- and I- tags into single spans
        current_entity = []
        current_label = None

        for i, tag in enumerate(bio_tags):
            if tag == subword_label:
                spans.append(Span(doc, i, i + 1, label=subword_label))  # treat subwords as individual spans
            elif tag.startswith("B-"):
                if current_entity:
                    spans.append(Span(doc, current_entity[0], current_entity[1], label=current_label))
                current_entity = [i, i + 1]
                current_label = tag
            elif tag.startswith("I-") and current_label and tag[2:] == current_label[2:]:
                current_entity[1] = i + 1
            else:
                if current_entity:
                    spans.append(Span(doc, current_entity[0], current_entity[1], label=current_label))
                current_entity = []
                current_label = None

        if current_entity:
            spans.append(Span(doc, current_entity[0], current_entity[1], label=current_label))
    else:
        # New default behavior: show B- and I- tags separately, skip O tags
        for i, tag in enumerate(bio_tags):
            if tag == subword_label:
                spans.append(Span(doc, i, i + 1, label=subword_label))
            elif tag.startswith("B-") or tag.startswith("I-"):
                # Display each B- and I- tag separately
                spans.append(Span(doc, i, i + 1, label=tag))
            # Skip "O" tags (background tokens)

    # Remove overlapping spans and assign to the Doc
    doc.ents = filter_spans(spans)

    # Generate consistent colors for entities if not provided
    if entity_colors is None:
        entity_colors = generate_entity_colors(
            full_label_list=label_list,
            cmap_name=cmap_name,
            special_labels={subword_label: subword_color}
        )

    options = {"colors": entity_colors}

    # Render the entities using displacy and display in notebook
    html = displacy.render(doc, style="ent", jupyter=False, options=options)

    wrapped_html = f"""
    <div style="
        line-height: 1.6;
        max-width: 120ch;
        white-space: normal;
        word-wrap: break-word;
        font-family: 'Segoe UI', sans-serif;
    ">{html}</div>
    """
    display(HTML(wrapped_html))

def display_pipeline_ner_html(text, pipeline_results, label_list, cmap_name="Set2"):
    """
    Visualizes NER predictions from HuggingFace pipeline output using spaCy's displacy tool.

    Args:
        text (str): The original input text.
        pipeline_results (list): Output from pipeline("token-classification") with aggregation_strategy.
                                Each item should have 'entity_group', 'start', 'end' keys.
        label_list (list of str): List mapping label indices to BIO tag names (e.g., BIO_tags_list).
        cmap_name (str): Name of colormap for entity colors (default: "Set2").

    Returns:
        None: Displays the rendered HTML visualization.
    """
    nlp = spacy.blank("en")
    doc = nlp(text)

    # Extract entity spans from pipeline results
    spans = []
    entity_labels = set()

    for entity in pipeline_results:
        # Get character positions
        start_char = entity['start']
        end_char = entity['end']
        entity_group = entity['entity_group']

        # Skip "O" (outside) entities
        if entity_group == "LABEL_0" or entity_group == "O":
            continue

        # Convert LABEL_X to actual BIO tag if needed
        if entity_group.startswith("LABEL_"):
            try:
                label_idx = int(entity_group.split("_")[1])
                label = label_list[label_idx]
            except (ValueError, IndexError):
                label = entity_group
        else:
            label = entity_group

        # Find tokens that overlap with this character span
        span = doc.char_span(start_char, end_char, label=label, alignment_mode="expand")
        if span is not None:
            spans.append(span)
            entity_labels.add(label)

    # Remove overlapping spans and assign to doc
    doc.ents = filter_spans(spans)

    # Generate colors for all entity types found
    entity_colors = generate_entity_colors(
        full_label_list=sorted(entity_labels),
        cmap_name=cmap_name
    )

    options = {"colors": entity_colors}

    # Render the entities using displacy
    html = displacy.render(doc, style="ent", jupyter=False, options=options)

    wrapped_html = f"""
    <div style="
        line-height: 1.6;
        max-width: 120ch;
        white-space: normal;
        word-wrap: break-word;
        font-family: 'Segoe UI', sans-serif;
    ">{html}</div>
    """
    display(HTML(wrapped_html))

def predict_ner_tags(text, model, tokenizer):
    """
    Tokenizes and predicts NER tags for the given text using a Hugging Face model.

    Args:
        text (str): Input sentence (e.g., "Barack Obama was born in Hawaii").
        model: A Hugging Face token classification model (e.g., DistilBERT).
        tokenizer: The tokenizer corresponding to the model.

    Returns:
        tokens (List[str]): Original word tokens from the input text.
        predicted_tag_ids (List[int]): One predicted tag index per word (subwords/specials skipped).
    """

    # Step 1: Split the input text into whitespace-separated words
    words = text.split()

    # Step 2: Tokenize the list of words and retain word alignment
    inputs = tokenizer(words, return_tensors="pt", is_split_into_words=True).to(model.device)

    # Step 3: Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Step 4: Convert logits to predicted label indices
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

    # Step 5: Get word IDs for each token
    word_ids = inputs.word_ids(batch_index=0)

    # Step 6: Extract one prediction per word (first subword only)
    predicted_tag_ids = []
    seen_words = set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx not in seen_words:
            predicted_tag_ids.append(int(predictions[token_idx]))
            seen_words.add(word_idx)
        # skip subwords and special tokens

    # Step 7: Return the original words and corresponding predicted tags
    return words, predicted_tag_ids

def format_ner_eval_results(results):
    """
    Formats the evaluation results of a Named Entity Recognition (NER) model into a pandas DataFrame.
    This function processes a dictionary of evaluation metrics, extracting entity-specific metrics
    (precision, recall, F1-score, etc.) and overall metrics, and organizes them into a tabular format.

    Args:
        results (dict): A dictionary containing evaluation metrics. Can have two formats:
                        1. Keys with "eval_" prefix (e.g., "eval_LOC", "eval_overall_precision")
                        2. Keys without prefix (e.g., "LOC", "overall_precision")
                        Entity-specific keys should map to dicts with 'precision', 'recall', 'f1', 'number'.

    Returns:
        pandas.DataFrame: A DataFrame where each row corresponds to an entity's evaluation metrics, and
                          an additional row summarizes the overall metrics. Columns include "Entity",
                          "Precision", "Recall", "F1", "Number", and optionally "Accuracy".
    """
    entity_rows = []

    # Check if keys have "eval_" prefix or not
    has_eval_prefix = any(k.startswith("eval_") for k in results.keys() if isinstance(results[k], dict))

    for key, value in results.items():
        # Handle both formats: with and without "eval_" prefix
        if isinstance(value, dict):
            if has_eval_prefix and not key.startswith("eval_"):
                continue  # Skip non-eval keys when we're in eval format
            if not has_eval_prefix and key.startswith("overall"):
                continue  # Skip overall keys at this stage

            # Extract the label
            label = key.replace("eval_", "") if key.startswith("eval_") else key

            # Skip "overall" keys (they'll be handled separately)
            if label.startswith("overall"):
                continue

            # Build the row
            row = {"Entity": label}
            row.update({k.capitalize(): round(v, 4) for k, v in value.items()})
            entity_rows.append(row)

    df = pd.DataFrame(entity_rows)

    # Add overall row
    # Try both with and without "eval_" prefix
    overall_precision = results.get("eval_overall_precision", results.get("overall_precision", 0))
    overall_recall = results.get("eval_overall_recall", results.get("overall_recall", 0))
    overall_f1 = results.get("eval_overall_f1", results.get("overall_f1", 0))
    overall_accuracy = results.get("eval_overall_accuracy", results.get("overall_accuracy", None))

    overall = {
        "Entity": "Overall",
        "Precision": round(overall_precision, 4),
        "Recall": round(overall_recall, 4),
        "F1": round(overall_f1, 4),
    }

    # Only add accuracy if it's available (not all NER metrics include it)
    if overall_accuracy is not None:
        overall["Accuracy"] = round(overall_accuracy, 4)

    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)
    return df

def json_extractor(text):
    # Extract the JSON object from the LLM response
    try:
        text = text.strip("```json").strip("```").strip()
        json_object = json.loads(text)
    except json.JSONDecodeError:
        json_object = {"error": "Could not parse JSON"}
    return json_object

def normalize(entity):
    # Lowercase and remove punctuation including apostrophes
    return re.sub(r"[^\w\s]", "", entity.lower()).strip()

def fuzzy_match(pred_set, gold_set, threshold=90):
    """
    Perform fuzzy matching between two sets of strings and calculate true positives (TP), 
    false positives (FP), and false negatives (FN) based on a similarity threshold.

    This function uses a similarity ratio to determine matches between predicted and 
    gold standard strings. A match is considered valid if the similarity ratio is 
    greater than or equal to the specified threshold.

    Args:
        pred_set (set): A set of predicted strings.
        gold_set (set): A set of gold standard (true) strings.
        threshold (int, optional): The similarity threshold (0-100) for matching. 
            Defaults to 90.

    Returns:
        tuple: A tuple containing:
            - tp (int): The number of true positives (correct matches).
            - fp (int): The number of false positives (incorrect predictions).
            - fn (int): The number of false negatives (missed gold standard strings).

    Note:
        This function requires the `fuzz` module from the `rapidfuzz` library. 
        For more details on `fuzz.ratio`, refer to the RapidFuzz documentation:
        https://rapidfuzz.readthedocs.io/en/latest/Ratio.html
    """
    matched_gold = set()
    tp = 0
    for pred in pred_set:
        for gold in gold_set:
            if gold in matched_gold:
                continue
            if fuzz.ratio(pred, gold) >= threshold:
                matched_gold.add(gold)
                tp += 1
                break
    fp = len(pred_set) - tp
    fn = len(gold_set) - tp
    return tp, fp, fn


def extract_gold_entities(example, labels_list):
    """
    Extracts named entities from a given example based on token-level NER tags.
    This function processes a sequence of tokens and their corresponding NER tags
    to identify and group named entities. It uses the BIO tagging scheme, where:
      - "B-" indicates the beginning of an entity,
      - "I-" indicates a continuation of the same entity,
      - "O" indicates no entity.
    
    Args:
        example (dict): A dictionary containing:
            - "tokens" (list of str): The list of tokens in the input text.
            - "ner_tags" (list of int): The list of NER tag indices corresponding to the tokens.
        labels_list (list of str): A list mapping tag indices to their string labels (e.g., "B-PER", "I-LOC").

    Returns:
        dict: A dictionary where keys are entity types (e.g., "PER", "LOC") and values
              are lists of entity strings extracted from the input.
    Example:
        Input:
            example = {
                "tokens": ["John", "Doe", "is", "from", "New", "York", "."],
                "ner_tags": [1, 2, 0, 0, 3, 4, 0]
            }
            labels_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
        Output:
            {
                "PER": ["John Doe"],
                "LOC": ["New York"]
            }
    """

    # Get the list of tokens and their corresponding tag indices
    tokens = example["tokens"]
    tags = example["ner_tags"]

    # Dictionary to store final entity results (e.g., {"PER": ["John Doe"]})
    entities = {}

    # These track the current entity being built
    current_entity = []   # A list of tokens that belong to the current entity
    current_type = None   # The type of the current entity (e.g., "PER", "LOC")

    # Go through each token and its corresponding tag index
    for token, tag_idx in zip(tokens, tags):
        tag = labels_list[tag_idx]  # Convert tag index to actual tag string, e.g., "B-PER"

        if tag.startswith("B-"):
            # Beginning of a new entity
            # Save the previous entity, if we were building one
            if current_entity:
                entity_text = " ".join(current_entity)
                if current_type in entities:
                    entities[current_type].append(entity_text)
                else:
                    entities[current_type] = [entity_text]

            # Start a new entity
            current_entity = [token]
            current_type = tag[2:]  # Extract the type, e.g., "PER" from "B-PER"

        elif tag.startswith("I-") and current_type == tag[2:]:
            # Continuation of the current entity (same type)
            current_entity.append(token)

        else:
            # Either an "O" tag or a mismatched "I-" tag
            # If we were building an entity, save it
            if current_entity:
                entity_text = " ".join(current_entity)
                if current_type in entities:
                    entities[current_type].append(entity_text)
                else:
                    entities[current_type] = [entity_text]

            # Reset — not currently building an entity
            current_entity = []
            current_type = None

    # If there is an entity left at the end, save it
    if current_entity:
        entity_text = " ".join(current_entity)
        if current_type in entities:
            entities[current_type].append(entity_text)
        else:
            entities[current_type] = [entity_text]

    return entities

def extract_entities_dict(pipeline_results, label_list):
    """
    Convert pipeline results to dictionary (or list of dictionaries) organized by entity type.

    This function works with HuggingFace token classification pipelines to extract named entities
    and organize them by type (PER, ORG, LOC, MISC). It properly merges multi-token entities
    (e.g., "Elon Musk") by combining consecutive B- and I- tags.

    Args:
        pipeline_results: Either:
            - Single result: List of dicts from pipeline (one text)
            - Batch results: List of lists of dicts from pipeline (multiple texts)
        label_list: List of BIO tags (e.g., ['O', 'B-PER', 'I-PER', 'B-LOC', ...])

    Returns:
        If single text input:
            dict: {'PER': ['Elon Musk', 'Sam Altman'], 'ORG': ['OpenAI'], ...}
        If batch input:
            list of dict: [{'PER': [...], 'ORG': [...]}, {'PER': [...], 'LOC': [...]}, ...]

    Example:
        >>> # Single text
        >>> results = pipeline("Elon Musk founded OpenAI.")
        >>> extract_entities_dict(results, BIO_tags_list)
        {'PER': ['Elon Musk'], 'ORG': ['OpenAI'], 'LOC': [], 'MISC': []}

        >>> # Batch of texts
        >>> results = pipeline(["Elon Musk lives in Texas.", "OpenAI is in San Francisco."])
        >>> extract_entities_dict(results, BIO_tags_list)
        [{'PER': ['Elon Musk'], 'LOC': ['Texas'], ...},
         {'ORG': ['OpenAI'], 'LOC': ['San Francisco'], ...}]
    """

    # Check if input is batched (list of lists) or single (list of dicts)
    # Batched: [[{result1}, {result2}], [{result3}, {result4}]]
    # Single:  [{result1}, {result2}, {result3}]
    is_batched = isinstance(pipeline_results[0], list) if pipeline_results else False

    # If batched, recursively process each text's results
    if is_batched:
        return [extract_entities_dict(single_result, label_list)
                for single_result in pipeline_results]

    # ============================================================================
    # Single text processing starts here
    # ============================================================================

    # Step 1: Initialize the output dictionary with all entity types
    # This ensures every entity type has a key, even if no entities are found
    entities = {}
    for label in label_list:
        if label != 'O':  # Skip 'O' which means "Outside" (no entity)
            # Extract entity type from BIO tag: 'B-PER' -> 'PER'
            entity_type = label.split('-')[-1]
            if entity_type not in entities:
                entities[entity_type] = []

    # Step 2: Track the current entity being built across multiple tokens
    # Example: "Elon" (B-PER) + "Musk" (I-PER) = "Elon Musk" (complete entity)
    current_entity_tokens = []  # Accumulates tokens for current entity
    current_entity_type = None  # Tracks which entity type we're building

    # Step 3: Process each token from the pipeline results sequentially
    # The pipeline returns results in text order, which is crucial for merging
    for result in pipeline_results:
        # Pipeline outputs entity_group as 'LABEL_X' where X is the label index
        # Example: 'LABEL_3' means index 3 in label_list
        entity_label = result['entity_group']

        # Convert 'LABEL_X' to the actual BIO tag string
        if entity_label.startswith('LABEL_'):
            label_idx = int(entity_label.replace('LABEL_', ''))

            # Look up the BIO tag (e.g., 'B-PER', 'I-LOC', 'O')
            if label_idx < len(label_list):
                bio_label = label_list[label_idx]

                # ----------------------------------------------------------------
                # Handle 'O' (Outside) tags - marks end of entity
                # ----------------------------------------------------------------
                if bio_label == 'O':
                    # If we were building an entity, save it now
                    if current_entity_tokens:
                        entity_text = ' '.join(current_entity_tokens).strip()
                        # Avoid adding duplicates or empty strings
                        if entity_text and entity_text not in entities[current_entity_type]:
                            entities[current_entity_type].append(entity_text)
                        # Reset state for next entity
                        current_entity_tokens = []
                        current_entity_type = None
                    continue  # Move to next token

                # ----------------------------------------------------------------
                # Extract entity information from BIO tag
                # ----------------------------------------------------------------
                entity_type = bio_label.split('-')[-1]  # 'B-PER' -> 'PER'
                entity_text = result['word'].strip()    # Token text

                # ----------------------------------------------------------------
                # Handle 'B-' (Beginning) tags - starts new entity
                # ----------------------------------------------------------------
                if bio_label.startswith('B-'):
                    # Save the previous entity if we were building one
                    if current_entity_tokens:
                        complete_entity = ' '.join(current_entity_tokens).strip()
                        if complete_entity and complete_entity not in entities[current_entity_type]:
                            entities[current_entity_type].append(complete_entity)

                    # Start building a new entity
                    current_entity_tokens = [entity_text]
                    current_entity_type = entity_type

                # ----------------------------------------------------------------
                # Handle 'I-' (Inside) tags - continues current entity
                # ----------------------------------------------------------------
                elif bio_label.startswith('I-'):
                    # Check if this I- tag matches the current entity type
                    if current_entity_type == entity_type:
                        # Add token to current entity
                        # Example: current=['Elon'], adding 'Musk' -> ['Elon', 'Musk']
                        current_entity_tokens.append(entity_text)
                    else:
                        # Mismatched I- tag (tagging error or special case)
                        # Treat it as starting a new entity
                        if current_entity_tokens:
                            complete_entity = ' '.join(current_entity_tokens).strip()
                            if complete_entity and complete_entity not in entities[current_entity_type]:
                                entities[current_entity_type].append(complete_entity)
                        # Start new entity with this token
                        current_entity_tokens = [entity_text]
                        current_entity_type = entity_type

    # Step 4: Don't forget the last entity if text ends while building one
    # Example: "... lives in New York" - need to save "New York" at the end
    if current_entity_tokens:
        complete_entity = ' '.join(current_entity_tokens).strip()
        if complete_entity and complete_entity not in entities[current_entity_type]:
            entities[current_entity_type].append(complete_entity)

    return entities

def llm_ner_extractor(model_name,
                      texts,
                      system_prompt,
                      prompt_template,
                      temperature=0):
    """
    Extract named entities using a Large Language Model (LLM) in zero-shot fashion.

    Args:
        model_name (str): Name of the LLM model to use (e.g., 'gemini-flash-lite').
        texts (list of str): List of input texts to process.
        system_prompt (str): System prompt guiding the LLM behavior.
        prompt_template (str): Template to construct the user prompt for each text.
        temperature (float, optional): Temperature for generation (0 = deterministic). Defaults to 0.

    Returns:
        list of dict: List of JSON objects containing extracted entities for each input text.
    """
    from introdl import llm_generate

    # Step 1: Create user prompts by formatting the prompt template with each input text.
    # This ensures that each text is passed to the LLM with the same structure.
    user_prompts = [prompt_template.format(text=text) for text in texts]

    # Step 2: Generate json outputs from the LLM using the provided model name and prompts.
    # The `llm_generate` function sends the prompts to the LLM and retrieves the responses.
    json_outputs = llm_generate(model_name,
                               user_prompts,
                               system_prompt=system_prompt,
                               mode='json',
                               temperature=temperature)

    return json_outputs

def evaluate_ner(pred_dicts, gold_dicts, labels=["PER", "LOC", "ORG", "MISC"], threshold=0.9):
    """
    Evaluate named entity recognition (NER) predictions using fuzzy string matching.

    This function compares predicted entity strings to gold (true) entity strings
    by entity type (e.g., PER, LOC, ORG) and computes precision, recall, F1 score,
    and accuracy using fuzzy matching.

    Args:
        pred_dicts (list of dict): A list of model predictions where each dict maps
                                   entity types (e.g., "PER") to lists of entity strings.
        gold_dicts (list of dict): A list of ground truth annotations in the same format.
        labels (list of str): The entity types to evaluate (default = ["PER", "LOC", "ORG", "MISC"]).
        threshold (float): Similarity threshold for fuzzy matching (0.0 to 1.0). Default is 0.9.

    Returns:
        dict: A dictionary of evaluation metrics for each entity type and overall, with keys like:
              "eval_PER", "eval_LOC", ..., "eval_overall_precision", "eval_overall_accuracy", etc.
    """

    # Count true positives, false positives, and false negatives per entity type
    tp_counts = Counter()
    fp_counts = Counter()
    fn_counts = Counter()

    # Compare predictions and gold labels sentence by sentence
    for pred, gold in zip(pred_dicts, gold_dicts):
        for label in labels:
            # Normalize and collect predicted and gold entities for this label
            pred_set = {normalize(e) for e in pred.get(label, [])}
            gold_set = {normalize(e) for e in gold.get(label, [])}

            # Use fuzzy matching to count TP, FP, and FN
            tp, fp, fn = fuzzy_match(pred_set, gold_set, threshold=threshold * 100)

            # Accumulate totals for this label
            tp_counts[label] += tp
            fp_counts[label] += fp
            fn_counts[label] += fn

    # Store final metric results
    results = {}
    total_tp = total_fp = total_fn = 0

    for label in labels:
        # Retrieve counts for this entity type
        tp, fp, fn = tp_counts[label], fp_counts[label], fn_counts[label]

        # Update overall totals
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Calculate precision, recall, F1 for this label
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        # Store in results under key like "eval_PER"
        results[f"eval_{label}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "number": tp + fn  # total number of gold entities for this type
        }

    # Compute overall (micro-averaged) metrics across all entity types
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) else 0.0
    overall_accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) else 0.0

    # Add overall scores to results
    results["eval_overall_precision"] = overall_precision
    results["eval_overall_recall"] = overall_recall
    results["eval_overall_f1"] = overall_f1
    results["eval_overall_accuracy"] = overall_accuracy

    return results


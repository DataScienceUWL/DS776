import numpy as np
from evaluate import load
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

# Load metrics
bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

def format_for_rougeLsum(texts):
    """Adds newlines between sentences as expected by ROUGE-Lsum."""
    if isinstance(texts, str):
        return "\n".join(sent_tokenize(texts.strip()))
    return ["\n".join(sent_tokenize(t.strip())) for t in texts]

def compute_all_metrics(predictions, references):
    """
    Compute BLEU, ROUGE, and BERTScore for a batch or single example.

    Inputs can be strings or lists of strings.
    Returns a dictionary of raw metric scores.
    """
    # Wrap single examples in lists
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]

    # BLEU (expects raw strings)
    bleu_result = bleu.compute(predictions=predictions, references=references)

    # ROUGE (add sentence breaks for Lsum)
    rouge_preds = format_for_rougeLsum(predictions)
    rouge_refs = format_for_rougeLsum(references)
    rouge_result = rouge.compute(predictions=rouge_preds, references=rouge_refs, use_stemmer=True)

    # BERTScore
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")

    # Aggregate metrics
    results = {
        "bleu": bleu_result["bleu"] * 100,
        "rouge1": rouge_result["rouge1"] * 100,
        "rouge2": rouge_result["rouge2"] * 100,
        "rougeL": rouge_result["rougeL"] * 100,
        "rougeLsum": rouge_result["rougeLsum"] * 100,
        "bertscore_f1": np.mean(bertscore_result["f1"]) * 100,
    }
    return results

def print_metrics(metrics):
    """Print nicely formatted metrics dictionary."""
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

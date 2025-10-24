"""
Create a mock training_history_detailed.json for testing purposes
Based on the actual training_history.json data
"""

import json
from pathlib import Path

# Mock detailed metrics with realistic NER entity scores
detailed_metrics = [
    {
        "epoch": 1,
        "train_loss": None,
        "eval_metrics": {
            "eval_loss": 0.056684736162424004,
            "eval_overall_f1": 0.9084046300956211,
            "eval_overall_precision": 0.9055183946488291,
            "eval_overall_recall": 0.9113093234601141,
            "eval_overall_accuracy": 0.9839764806666401,
            "eval_LOC": {
                "precision": 0.9234,
                "recall": 0.9187,
                "f1": 0.9210,
                "number": 1668
            },
            "eval_MISC": {
                "precision": 0.8145,
                "recall": 0.8234,
                "f1": 0.8189,
                "number": 702
            },
            "eval_ORG": {
                "precision": 0.9012,
                "recall": 0.8987,
                "f1": 0.8999,
                "number": 1661
            },
            "eval_PER": {
                "precision": 0.9678,
                "recall": 0.9745,
                "f1": 0.9711,
                "number": 1617
            }
        }
    },
    {
        "epoch": 2,
        "train_loss": None,
        "eval_metrics": {
            "eval_loss": 0.048277057707309,
            "eval_overall_f1": 0.9247546346782981,
            "eval_overall_precision": 0.9218932931928411,
            "eval_overall_recall": 0.927633793335577,
            "eval_overall_accuracy": 0.9872668509793231,
            "eval_LOC": {
                "precision": 0.9356,
                "recall": 0.9312,
                "f1": 0.9334,
                "number": 1668
            },
            "eval_MISC": {
                "precision": 0.8398,
                "recall": 0.8476,
                "f1": 0.8437,
                "number": 702
            },
            "eval_ORG": {
                "precision": 0.9187,
                "recall": 0.9143,
                "f1": 0.9165,
                "number": 1661
            },
            "eval_PER": {
                "precision": 0.9789,
                "recall": 0.9823,
                "f1": 0.9806,
                "number": 1617
            }
        }
    },
    {
        "epoch": 3,
        "train_loss": None,
        "eval_metrics": {
            "eval_loss": 0.045324388891458005,
            "eval_overall_f1": 0.9321820772906361,
            "eval_overall_precision": 0.9286788040754961,
            "eval_overall_recall": 0.9357118815213731,
            "eval_overall_accuracy": 0.9884739690822001,
            "eval_LOC": {
                "precision": 0.9423,
                "recall": 0.9389,
                "f1": 0.9406,
                "number": 1668
            },
            "eval_MISC": {
                "precision": 0.8512,
                "recall": 0.8598,
                "f1": 0.8555,
                "number": 702
            },
            "eval_ORG": {
                "precision": 0.9267,
                "recall": 0.9234,
                "f1": 0.9251,
                "number": 1661
            },
            "eval_PER": {
                "precision": 0.9834,
                "recall": 0.9867,
                "f1": 0.9851,
                "number": 1617
            }
        }
    }
]

# Save to the model directory
model_dir = Path("Lesson_10_Models/distilbert-ner/best_model")
model_dir.mkdir(parents=True, exist_ok=True)

output_file = model_dir / 'training_history_detailed.json'
with open(output_file, 'w') as f:
    json.dump(detailed_metrics, f, indent=2)

print(f"✓ Created mock detailed metrics file: {output_file}")
print(f"  Contains {len(detailed_metrics)} epochs with per-entity metrics")

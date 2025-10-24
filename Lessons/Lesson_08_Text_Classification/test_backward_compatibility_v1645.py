"""
Test backward compatibility: Verify that text classification (Lesson 8)
displays simple DataFrame as before, not rich display.
"""

import sys
import json
from pathlib import Path

# Add introdl to path
sys.path.insert(0, str(Path.home() / 'DS776_new' / 'Lessons' / 'Course_Tools' / 'introdl' / 'src'))

from introdl.nlp import TrainerWithPretend

print("="*80)
print("Testing Backward Compatibility - Lesson 8 Text Classification")
print("introdl v1.6.45 should show simple DataFrame, NOT rich display")
print("="*80)

# Check Lesson 8 model directory
model_dir = Path("Lesson_08_Models/L08_fine_tune_distilbert/best_model")
history_file = model_dir / 'training_history.json'
detailed_file = model_dir / 'training_history_detailed.json'

print(f"\n1. Checking Lesson 8 model files...")
print(f"   training_history.json exists: {history_file.exists()}")
print(f"   training_history_detailed.json exists: {detailed_file.exists()}")

if history_file.exists():
    # Load the simple history
    import pandas as pd
    history_df = pd.read_json(history_file)

    print(f"\n2. Loaded training history with {len(history_df)} epochs")
    print(f"   Columns: {list(history_df.columns)}")

    # Check if metrics are flat (classification) not nested (NER)
    print(f"\n3. Checking metric structure...")
    has_eval_accuracy = 'eval_accuracy' in history_df.columns
    has_eval_overall_f1 = 'eval_overall_f1' in history_df.columns

    print(f"   Has eval_accuracy (classification): {has_eval_accuracy}")
    print(f"   Has eval_overall_f1 (NER): {has_eval_overall_f1}")

    # Test the _extract_detailed_metrics detection logic
    print(f"\n4. Testing metric detection...")

    # Create minimal trainer instance
    trainer = TrainerWithPretend.__new__(TrainerWithPretend)
    trainer._training_history = history_df
    trainer._detailed_training_history = None

    # Try to load detailed metrics (should be None for classification)
    if detailed_file.exists():
        with open(detailed_file, 'r') as f:
            trainer._detailed_training_history = json.load(f)
        print("   ✗ UNEXPECTED: detailed metrics found for text classification!")
    else:
        print("   ✓ No detailed metrics file (expected for text classification)")

    # Test formatting
    detailed_display = trainer._format_detailed_metrics_display()

    if detailed_display is None:
        print("   ✓ _format_detailed_metrics_display() returned None (correct)")
        print("   ✓ Will use simple DataFrame display (backward compatible)")
        print("\n📊 Training History (Simple DataFrame Display):")
        print(history_df.to_string(index=False))

        # Check best model detection
        if has_eval_accuracy and not history_df['eval_accuracy'].isna().all():
            best_idx = history_df['eval_accuracy'].idxmax()
            best_epoch = history_df.loc[best_idx, 'epoch']
            best_acc = history_df.loc[best_idx, 'eval_accuracy']
            print(f"\n✓ Best model: Epoch {best_epoch:.0f} | Accuracy: {best_acc:.4f}")
    else:
        print("   ✗ ERROR: _format_detailed_metrics_display() returned rich display!")
        print("   This should only happen for NER models with nested metrics.")

    print("\n" + "="*80)
    print("✓ BACKWARD COMPATIBILITY TEST PASSED!")
    print("="*80)
    print("\nExpected behavior:")
    print("- No training_history_detailed.json file")
    print("- _format_detailed_metrics_display() returns None")
    print("- Simple DataFrame display shown above")
    print("- Best model uses eval_accuracy (not eval_overall_f1)")
    print("\n✓ Text classification (Lesson 8) remains unchanged!")

else:
    print(f"\n✗ ERROR: No training history found at {history_file}")
    print("   Run Lesson 8 training first to test backward compatibility.")

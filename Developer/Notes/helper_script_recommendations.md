# Helper Script Recommendations

## Current State Analysis

### Helper Scripts by Category

#### 1. **Visualization Helpers** (Good candidates for package)
- `conv_size_widget.py` (L02) - Interactive widget for convolution size calculation
- `viz_functions.py` (L09) - Transformer attention visualization
- NER display functions in L10 helpers

#### 2. **Model Training Helpers** (Mixed - some generic, some specific)
- Object detection training (L06/HW06) - Very specific to RCNN/YOLO
- LLM classifier wrapper (HW08) - Simple API wrapper

#### 3. **Data Preparation** (Task-specific, keep local)
- Spiral data generation (HW01) - Homework-specific
- Penn-Fudan/Nuclei dataset prep (L06/HW06) - Dataset-specific

#### 4. **Text Generation Utilities** (Good candidates for package)
- L11/HW11 helpers - Identical, used for decoding strategies visualization
- L12/HW12 helpers - Identical, ROUGE metrics computation

#### 5. **NER Utilities** (Good candidate for package)
- L10 helpers - Entity visualization and evaluation

## Recommendations

### 1. **Move to introdl Package** ✅

Create new modules in the introdl package:

```python
introdl/
├── viz/
│   ├── __init__.py
│   ├── conv_widget.py      # From L02 conv_size_widget.py
│   ├── attention.py         # From L09 viz_functions.py
│   └── ner.py              # NER visualization from L10
├── nlp/
│   ├── generation.py       # From L11/HW11 helpers.py
│   ├── summarization.py    # From L12/HW12 helpers.py
│   └── ner_utils.py        # NER utilities from L10
└── data/
    └── spiral.py           # From HW01 make_spirals
```

**Benefits:**
- No duplicate files (L11/HW11, L12/HW12)
- Consistent API across lessons
- Students learn to import from package
- Easier maintenance and updates

### 2. **Keep Local (in homework/lesson folders)** ⚠️

Keep these as local scripts:
- `graphics_and_data_prep.py` (L06/HW06) - Too specific to object detection tasks
- Dataset-specific preparation functions
- One-off utility functions

**Benefits:**
- Students see example of local helper scripts
- Task-specific code stays with the task
- Reduces package complexity

### 3. **Hybrid Approach (Recommended)** 🎯

**Phase 1 - Immediate:**
1. Move identical duplicates to package (L11/HW11, L12/HW12 helpers)
2. Move generic visualization tools (conv widget, attention viz)
3. Keep dataset-specific code local

**Phase 2 - After testing:**
1. Evaluate student confusion with imports
2. Consider moving more utilities if beneficial

**Implementation:**
```python
# In homework notebooks:
from introdl.nlp.generation import generate_greedy_decoding_table
from introdl.viz.attention import plot_attention_weights

# For local helpers (if needed):
from graphics_and_data_prep import prepare_penn_fudan_yolo
```

## Educational Considerations

### Pros of Package Approach:
1. **Professional practice** - Students learn package-based development
2. **Cleaner notebooks** - Focus on concepts, not utility code
3. **Consistency** - Same functions work everywhere
4. **No file copying** - Reduces errors and confusion

### Pros of Local Scripts:
1. **Transparency** - Students can see all code
2. **Customization** - Easy to modify for experiments
3. **Learning opportunity** - Shows code organization patterns

## Recommended Action Plan

### Immediate Actions:
1. **Create introdl submodules:**
   - `introdl.nlp.generation` (L11/HW11 helpers)
   - `introdl.nlp.summarization` (L12/HW12 helpers)
   - `introdl.viz.conv` (L02 conv widget)

2. **Update imports in affected notebooks:**
   - HW11: Change from `from helpers import *` to `from introdl.nlp.generation import *`
   - HW12: Similar update

3. **Add docstring to each module explaining:**
   - Purpose of the module
   - Which lessons/homeworks use it
   - Example usage

### Keep Local:
1. **HW01** - `make_spirals` (consider moving to introdl.data.spiral later)
2. **L06/HW06** - Object detection helpers (too specific)
3. **HW08** - `llm_classifier` (simple enough to inline)

### Documentation:
Add to course materials:
- "When to use package imports vs local scripts"
- "Understanding the introdl package structure"
- Examples of both approaches

## File Renaming While We're At It

If keeping local scripts, standardize names:
- `Homework_XX_Helpers.py` (capital H, consistent naming)
- Copy missing helpers from lessons to homework where needed
- Document at top of file if it's a copy from lesson

## Conclusion

**Recommended approach: Hybrid**
- Move reusable, generic utilities to package
- Keep task-specific code local
- Document the reasoning for students

This balances:
- **Convenience** (no duplicate files)
- **Education** (students see both patterns)
- **Maintainability** (clear separation of concerns)
# Documentation Regeneration Instructions

## Overview

This guide explains how to regenerate introdl documentation after updating
L07-L12 NLP functions or making other package changes.

## Quick Regeneration

### Method 1: Automated Script
```bash
cd /mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs
bash build_docs.sh both
```

### Method 2: Make Commands
```bash
cd /mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs
make rebuild
```

### Method 3: Manual Sphinx
```bash
cd /mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs
sphinx-build -b html source build/html
sphinx-build -b markdown source build/markdown
```

## When to Regenerate

### **Always regenerate after:**
- Adding new public functions to any module
- Updating docstrings with new examples
- Adding L07-L12 NLP function documentation
- Changing function signatures
- Adding new modules to the package

### **Consider regenerating after:**
- Updating existing examples for accuracy
- Fixing typos in docstrings
- Adding new lesson examples (L07-L12)
- Updating version numbers

## Adding L07-L12 NLP Functions

### Step 1: Update NLP Module Documentation
Edit `/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs/source/api/nlp_functions.rst` (create if needed):

```rst
NLP Functions
=============

.. automodule:: introdl.nlp
   :members:
   :undoc-members:
   :show-inheritance:

Text Generation
--------------

.. automodule:: introdl.generation
   :members:
   :undoc-members:
   :show-inheritance:

Text Summarization
-----------------

.. automodule:: introdl.summarization
   :members:
   :undoc-members:
   :show-inheritance:
```

### Step 2: Update Main Index
Add to `/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs/source/index.rst` toctree:

```rst
.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/core_utilities
   api/training_functions
   api/visualization_functions
   api/nlp_functions        # <-- Add this line
   api/storage_utilities
```

### Step 3: Add L07-L12 Examples
Create `/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs/source/examples/nlp_examples.rst`:

```rst
NLP Examples (L07-L12)
=====================

Examples from NLP lessons showing text processing, generation, and summarization.

Lesson 07: Transformer Introduction
----------------------------------

.. code-block:: python

   from introdl.nlp import llm_configure, llm_generate

   # Setup NLP pipeline
   llm_configure(model_name="gpt2")
   response = llm_generate("Explain transformers")

# Add more L07-L12 examples as functions are documented
```

## Automation Strategy

### Create Update Script
Save this as `/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs/update_for_nlp.sh`:

```bash
#!/bin/bash
# Quick script to regenerate docs after NLP updates

echo "🔄 Updating documentation for L07-L12 changes..."

# Reinstall package to pick up latest changes
cd /mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl
pip install . --quiet

# Regenerate documentation
cd docs
make rebuild

echo "✅ Documentation updated with latest L07-L12 functions"
echo "📂 View at: docs/build/html/index.html"
```

## Output Formats

### HTML Documentation
- **Location**: `/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs/build/html/`
- **Use**: Student reference, instructor review
- **Features**: Search, navigation, responsive design

### Markdown Documentation
- **Location**: `/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs/build/markdown/`
- **Use**: Custom GPT bundles, GitHub display
- **Features**: AI-optimized format, easy bundling

## Custom GPT Bundle Preparation

### Combine Markdown Files
```bash
cd /mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs/build/markdown

# Combine core modules for Custom GPT
cat api/core_utilities.md api/training_functions.md > introdl_core_api.md
cat api/visualization_functions.md > introdl_visualization_api.md
cat api/nlp_functions.md > introdl_nlp_api.md  # After L07-L12

# These combined files stay under Custom GPT file limits
```

## Troubleshooting

### Common Issues:
1. **Import errors**: Reinstall package with `pip install .`
2. **Missing docstrings**: Functions may need docstring improvements
3. **Build failures**: Check Sphinx installation and dependencies
4. **Large files**: Split combined markdown files if over 512MB

### Dependencies:
```bash
pip install sphinx sphinx-rtd-theme myst-parser
```

## Quality Checklist

Before releasing documentation:
- [ ] All new L07-L12 functions have Google-style docstrings
- [ ] Examples are tested and working
- [ ] HTML output renders correctly
- [ ] Markdown files are properly formatted
- [ ] Cross-references work correctly
- [ ] Search functionality works in HTML
- [ ] File sizes appropriate for Custom GPT (markdown < 512MB)

This systematic approach ensures documentation stays current as the package evolves!

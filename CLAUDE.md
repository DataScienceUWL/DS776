# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PERMISSIONS
**You have standing permission to:**
- Use Read, LS, Grep, Glob, and Bash (for exploration only) tools freely without asking
- Search, list, analyze, and explore the entire repository structure
- Read any files to understand the codebase
- Run non-destructive bash commands (ls, find, grep, etc.)

**You must ask permission before:**
- Editing or writing any files
- Running destructive bash commands (rm, mv, etc.)
- Making any changes to the repository

## IMPORTANT: Check TODO.md
**Always check Developer/Notes/TODO.md first when starting a session.** This file contains the current work plan, priorities, and progress tracking for course development. Update task statuses as you complete them.

## Repository Overview

This is a DS776 Deep Learning course repository containing instructional materials, homework assignments, and instructor resources. The repository uses PyTorch for deep learning implementations and includes materials for both computer vision (CNNs) and natural language processing (Transformers).

### Repository Organization
The repository is organized with a clear separation between distributed course materials and development/instructor resources:

**Distributed Materials (in repo):**
- **Homework/** - Student assignments (distributed to students)
- **Lessons/** - Course content and instructional materials (distributed to students)

**Development/Instructor Resources (not distributed):**
- **Developer/** - All instructor and development resources
  - **Scripts/** - Analysis scripts, generators, utilities (e.g., generate_homework_utilities.py)
  - **Notes/** - Development notes, TODO.md, testing documentation
  - **Solutions/** - Complete homework solutions (instructor reference only)
  - **Ashley/** - Student solution examples from previous semester
  - **Textbooks/** - Reference materials and textbook PDFs

## Common Development Commands

### Python Environment Setup
```bash
# Install the introdl package (course utilities)
cd Lessons/Course_Tools/introdl
pip install .

# Or use the installation script
python Lessons/Course_Tools/install_introdl.py
```

### Running Jupyter Notebooks
```bash
# Start Jupyter Lab or Notebook
jupyter lab
# or
jupyter notebook
```

### Testing Code in Notebooks
When working with notebook cells, test code execution with:
- Run individual cells to verify functionality
- Check GPU availability: `torch.cuda.is_available()`
- Verify introdl package imports: `from introdl.utils import *`

## High-Level Architecture

### Course Structure
The repository follows a lesson-based structure:
- **Lessons/** - Main course content organized by topic (L01-L13)
  - Each lesson contains numbered notebooks with embedded videos
  - Helper modules for specific lessons (e.g., `helpers.py`, `viz_functions.py`)
  
- **Homework/** - Student assignments corresponding to lessons
  - Each homework folder contains Jupyter notebooks and supporting scripts
  - Each folder includes Homework_XX_Utilities.ipynb for storage management

### Key Dependencies
The course relies on the following main frameworks (from `introdl/pyproject.toml`):
- PyTorch ecosystem: `torch`, `torchvision`, `torchmetrics`, `torchinfo`
- Transformers: `transformers>=4.49.0`, `datasets`, `evaluate`
- Computer Vision: `timm`, `segmentation_models_pytorch`, `ultralytics` (YOLO)
- NLP: `spacy`, `bert_score`, `bertviz`
- LLM APIs: `openai>=1.65.5`, `google-genai`
- Visualization: `matplotlib`, `seaborn`, `ipywidgets`

### Custom Package: introdl
Located in `Lessons/Course_Tools/introdl/`, this package provides:
- `idlmam` - Course-specific implementations
- `utils` - Utility functions for the course
- `visul` - Visualization helpers
- `nlp` - NLP-specific utilities

### Lesson Progression
1. **L01-L06**: Computer Vision fundamentals
   - Neural Networks basics
   - CNNs, training techniques
   - Transfer learning, object detection
   
2. **L07-L12**: NLP and Transformers
   - Transformer introduction
   - Text classification, NER
   - Text generation, summarization
   
3. **L13**: Final project

### Working with Course Materials
- Notebooks often include embedded videos accessed through the introdl package
- Many notebooks depend on helper modules in the same directory
- Environment files (`.env`) contain API keys and configuration - handle with care

## Workspace Structure (NEW)
The repository now uses a workspace structure that mirrors student CoCalc environments:
- `home_workspace/` - Synced storage (10GB limit in CoCalc)
  - `data/` - Datasets
  - `downloads/` - Cached pretrained models
  - `models/` - Student-trained model checkpoints
- `cs_workspace/` - Compute server local storage (not synced)
- For local development, DS776_ROOT_DIR uses `home_workspace/` to simulate student experience

## Important Notes
- This is an educational repository - prioritize clarity and learning objectives
- When modifying notebooks, preserve markdown cells with instructional content
- Test GPU/CPU compatibility when working with deep learning code
- Respect the lesson numbering system when adding new content
- Follow Developer/Notes/TODO.md for current development priorities and tasks

## File Storage Guidelines
**When creating new files, follow this structure:**
- **Homework/** and **Lessons/** - Only course materials that students will use
- **Developer/Scripts/** - Any analysis scripts, generators, or utility scripts
- **Developer/Notes/** - Documentation, TODO lists, testing notes, CSVs from analysis
- **Developer/Solutions/** - Complete homework solutions (instructor reference)
- **Developer/Textbooks/** - Reference PDFs and textbook materials
- Never place development files in the root directory

## Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive context window. Use `gemini -p` to leverage Google Gemini's large context capacity.

### Best Practices for Gemini CLI

**Preferred Use Cases:**
- Targeted codebase analysis with specific questions
- Finding implementations across large directories
- Understanding complex interactions between modules
- Verifying presence of features or patterns
- Analyzing code that would exceed Claude's context window

**Important Considerations:**
- **Response Time**: Full codebase analysis (`@./`) can take 10+ minutes
- **Use Background Execution**: For large analyses, always use `run_in_background: true`
- **Response Format**: Gemini responds conversationally, not as structured documents
- **Targeted Questions**: Ask specific questions rather than requesting document generation
- **Context Preservation**: Use Gemini for analysis to keep Claude's context free for implementation

**When to Use Gemini Instead of Claude's Tools:**
- When searching would require reading many large files
- When you need to understand project-wide patterns
- Before making architectural decisions that affect multiple modules
- To verify if a feature exists before implementing it
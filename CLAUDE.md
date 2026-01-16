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

## GIT WORKFLOW - CRITICAL REQUIREMENTS

**🔴 MANDATORY: Always commit changes BEFORE making edits to important files**

Before editing any lesson notebooks, homework assignments, or other critical files:
1. **Check git status** to see what's currently modified
2. **Commit any uncommitted changes** with a descriptive message
3. **Then proceed** with your edits

**Why this is critical:**
- Prevents loss of work if edits go wrong
- Creates restore points during long editing sessions
- Makes it easy to revert problematic changes
- Preserves history of incremental improvements

**🔴 MANDATORY: Push to remote frequently**

After completing a logical unit of work (e.g., fixing a section, adding a feature):
1. **Commit your changes** with a clear, descriptive message
2. **Push to remote immediately** with `git push`
3. **Don't wait** until "everything is perfect" - push incremental progress

**Why this is critical:**
- Backs up work to remote repository (safety net)
- Ensures work is not lost if local system fails
- Makes work accessible from other machines
- Creates cloud backup of all progress

**Example workflow:**
```bash
# Before starting edits
git status
git add .
git commit -m "State before editing notebook inference section"

# Make your edits...
# After completing a section
git add .
git commit -m "Update inference section with pipeline examples"
git push

# Continue with next section...
```

**When in doubt:**
- **Commit early, commit often** - it's better to have too many commits than too few
- **Push after every significant change** - don't accumulate unpushed commits
- **Use descriptive commit messages** - explain what changed and why

## IMPORTANT: Check Session Planning Documents

**At the start of EVERY session, check these files in order:**

1. **Developer/NEXT_SESSION.md** - Contains context for continuing work from previous sessions
   - Current task status and what we're working on
   - Known blockers and how to resolve them
   - Quick start instructions
   - Files to review for context

2. **Developer/TODO.md** - Master task list and project roadmap
   - Current priorities in priority order
   - Completed tasks (for reference)
   - Known issues to watch for
   - Future enhancements

**Throughout the session:**
- Update task statuses in TODO.md as you complete them (check off boxes)
- Update NEXT_SESSION.md if you discover important context for the next session
- Keep both files current so future sessions can continue seamlessly

## Repository Overview

This is a DS776 Deep Learning course repository containing instructional materials, homework assignments, and instructor resources. The repository uses PyTorch for deep learning implementations and includes materials for both computer vision (CNNs) and natural language processing (Transformers).

### Repository Organization
The repository is organized with a clear separation between distributed course materials and development/instructor resources:

**Distributed Materials (in repo):**
- **Homework/** - Student assignments (distributed to students)
- **Lessons/** - Course content and instructional materials (distributed to students)

**Development/Instructor Resources (not distributed):**
- **Developer/** - All instructor and development resources
  - **TODO.md** - Master task list and project roadmap (root of Developer/)
  - **NEXT_SESSION.md** - Context for continuing work between sessions (root of Developer/)
  - **Scripts/** - Analysis scripts, generators, utilities (e.g., generate_homework_utilities.py)
  - **Notes/** - Development notes, testing documentation, analysis results
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
- Verify introdl package imports: `from introdl import config_paths_keys`

### Import Conventions for Course Notebooks
**IMPORTANT: All course notebooks should use flattened imports from introdl:**

**Correct (Flattened Import):**
```python
from introdl import (
    config_paths_keys, get_device, wrap_print_text,
    llm_generate, init_cost_tracking, display_markdown
)
```

**Incorrect (Submodule Imports - DON'T USE):**
```python
# Don't do this:
from introdl.utils import config_paths_keys
from introdl.nlp import llm_generate
```

**Why flattened imports?**
- Simpler for students - one import line instead of multiple
- Easier to maintain - don't need to remember which submodule each function belongs to
- Consistent with the flat package structure in `__init__.py`

When creating or updating notebooks:
- Always use flattened imports: `from introdl import ...`
- List all needed functions in one import statement
- Group related functions on the same line for readability

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

## Workspace Structure

### WSL Native Filesystem Setup (Current)
The repository is stored in **WSL native filesystem** for optimal performance and proper file system event propagation:
- **Location**: `~/DS776_new/` (real files on ext4 filesystem)
- **Symlink**: `~/DS776` → `~/DS776_new` (for compatibility)
- **Windows Access**: `\\wsl$\Ubuntu\home\jbaggett\DS776\` (or create Windows symlink)

### Google Drive Backup
Use the automated rsync script to backup from WSL to Google Drive:
```bash
# Run backup (from anywhere)
bash ~/DS776/backup_to_gdrive.sh

# Manual rsync command
rsync -a --info=progress2 --delete ~/DS776/ /mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/
```

**Why WSL Native?**
- **File System Events**: Changes properly trigger filesystem watchers (git, VS Code, etc.)
- **Performance**: Much faster than DrvFS-mounted Google Drive (10x+ speed improvement)
- **Reliability**: No sync conflicts between WSL, Windows, and Google Drive
- **Git Operations**: Git works correctly without phantom file changes

### Student Environment Mirroring
The workspace structure mirrors student CoCalc environments:
- `home_workspace/` - Synced storage (10GB limit in CoCalc)
  - `data/` - Datasets
  - `downloads/` - Cached pretrained models
  - `models/` - Student-trained model checkpoints
- `cs_workspace/` - Compute server local storage (not synced)
- For local development, DS776_ROOT_DIR uses `home_workspace/` to simulate student experience

## Generating Homework Solutions

### Quick Command
When instructor requests homework solutions, use:
```
Create solutions for Homework_XX
```

### Automatic Workflow
Claude will automatically:
1. **Find assignment**: `Homework/Homework_XX/Homework_XX_Assignment.ipynb`
2. **Locate student example**: `Developer/Ashley/HWXX/Homework_XX_GRADE_THIS_ONE.ipynb`
3. **Check lesson overview**: `Lessons/Lesson_XX_[Topic]/L_XX_0_Overview.ipynb` (for reading assignments)
4. **Find textbook PDF**:
   - Computer Vision (HW01-04): `Developer/Textbooks/idlmam/Chapter_XX_*.pdf`
   - NLP (HW05-08): `Developer/Textbooks/nlpwt/Chapter_XX_*.pdf`
5. **Generate solution**: `Developer/Solutions/Homework_XX/Solutions_XX.ipynb`

### Homework-Lesson-Textbook Mapping
| HW | Lesson | Topic | Textbook Reading |
|----|--------|-------|------------------|
| 01 | L01 | Neural Networks | IDL Ch. 1 (focus 1.2, 1.4, 1.5), Ch. 2 |
| 02 | L02 | CNNs | IDL Ch. 3 (through 3.5) |
| 03 | L03 | Better Training | IDL Sections 3.6, 5.1-5.3 |
| 04 | L04 | Better Networks | IDL Sections 6.1-6.5 |
| 05 | L05 | Transfer Learning | IDL Sections 13.1-13.3 |
| 06 | L06 | Object Detection | IDL Chapter 8 |
| 07 | L07 | Transformers Intro | NLPWT Ch. 1 |
| 08 | L08 | Text Classification | NLPWT Ch. 2 |
| 09 | L09 | Transformer Details | Report (no solution) |
| 10 | L10 | Named Entity Recognition | NLPWT Ch. 4 |
| 11 | L11 | Text Generation | NLPWT Ch. 5 |
| 12 | L12 | Summarization | NLPWT Ch. 6 |

**Notes:**
- HW09 is a report assignment (no solution needed)
- L13 is final project (no homework)
- Readings in `Lessons/Lesson_XX_[Topic]/L_XX_0_Overview.ipynb`

**Full instructions**: `Developer/Solutions/Solution_Generation_Instructions.md`

## Generating Canvas Reading Quizzes

### Overview
Reading quizzes are Canvas multiple choice quizzes (10 pts each) created from textbook readings. They are separate from homework notebooks and imported into Canvas as QTI packages.

### Tool: text-to-qti
- **Install:** `pip install text-to-qti`
- **Docs:** https://pypi.org/project/text-to-qti/
- **Convert:** `text-to-qti convert quiz_XX.md -o quiz_XX.zip`
- **Import to Canvas:** Settings → Import Course Content → QTI .zip file

### Quiz File Location
- **Source files:** `Developer/Quizzes/quiz_XX.md`
- **Generated ZIPs:** `Developer/Quizzes/quiz_XX.zip`

### Quiz Markdown Format
```markdown
---
title: Lesson X Reading Quiz
description: Quiz on readings for Lesson X
points_per_question: 1
shuffle_answers: true
---

## Question 1
[Type: multiple_choice]

Question text here?

a) Wrong answer
*b) Correct answer (marked with asterisk)
c) Wrong answer
d) Wrong answer

Feedback: Explanation with textbook reference (e.g., "See Section 2.3").
```

### Quiz Creation Workflow
1. **Read assigned sections** from textbook (see Homework-Lesson-Textbook Mapping above)
2. **Create quiz_XX.md** with 10 multiple choice questions (1 pt each)
3. **Align questions** with specific sections - include page/section references in feedback
4. **Convert to QTI:** `text-to-qti convert Developer/Quizzes/quiz_XX.md -o Developer/Quizzes/quiz_XX.zip`
5. **Import to Canvas:** Course Settings → Import Course Content → QTI .zip file
6. **Configure in Canvas:** Set due date, attempts, time limit as needed

### Quiz Alignment Requirements
**CRITICAL:** Each quiz question MUST be answerable from the assigned reading only. When creating quizzes:
- Reference specific textbook sections
- Use terminology from the textbook
- Avoid questions that require external knowledge
- Include feedback that points students to relevant sections

### Spring 2026 Plan
See `Developer/Spring_2026_Launch_Plan.md` for the full launch plan including:
- Quiz creation checklist for all 11 lessons (L09 excluded - report assignment)
- Homework update tasks (remove embedded reading questions, add storage reminders)
- Issues from `Developer/Spring_2026_updates.md` to address

## Important Notes
- This is an educational repository - prioritize clarity and learning objectives
- When modifying notebooks, preserve markdown cells with instructional content
- Test GPU/CPU compatibility when working with deep learning code
- Respect the lesson numbering system when adding new content
- Follow Developer/TODO.md and Developer/NEXT_SESSION.md for current development priorities and tasks

## File Storage Guidelines
**When creating new files, follow this structure:**
- **Homework/** and **Lessons/** - Only course materials that students will use
- **Developer/** - Session planning documents (TODO.md, NEXT_SESSION.md)
- **Developer/Scripts/** - Any analysis scripts, generators, or utility scripts
- **Developer/Notes/** - Documentation, testing notes, CSVs from analysis
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

## introdl Package Version Management

**CRITICAL: Always increment the introdl package version when making changes that affect functionality.**

### When to Bump Version
**ALWAYS increment the version** in `Lessons/Course_Tools/introdl/src/introdl/__init__.py` when you:
- Add new functions to introdl.utils, introdl.idlmam, introdl.visul, or introdl.nlp
- Modify existing function signatures or behavior
- Fix bugs that change how functions work
- Add or change dependencies in pyproject.toml
- Modify path handling, device detection, or core utilities
- Change any functionality students might use in notebooks

### How to Bump Version
1. **Open** `Lessons/Course_Tools/introdl/src/introdl/__init__.py`
2. **Increment** the version number (e.g., "1.4.1" → "1.4.2")
3. **Add entry** to version history comment
4. **Commit changes** - the auto-update system will handle student updates

### Why This Matters
The auto-update system compares installed vs source versions to determine if students need package updates. Without version bumps, students won't get critical fixes and may encounter import errors or function failures.

**Example version bump:**
```python
__version__ = "1.4.2"  # Bump this version on each update
# Version history:
# 1.4.2 - Fixed config_paths_keys Windows compatibility, added new visualization utils
# 1.4.1 - Fixed API key priority (DS776_ROOT_DIR first), cs_workspace for data on compute servers
```
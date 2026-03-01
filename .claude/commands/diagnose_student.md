---
description: Diagnose student notebook issues and generate a friendly response
---

# Diagnose Student Notebook Issue

## Description
Analyzes a student's homework notebook to diagnose issues, compares against the instructor solution, and generates a friendly diagnostic response markdown that the instructor can send to the student. Supports three invocation modes: local file, CoCalc fetch, and question-only.

## Usage
```
/diagnose-student <path_to_notebook.ipynb>              # Mode 1: Local file
/diagnose-student <StudentName> HW<XX>                   # Mode 2: Fetch from CoCalc
/diagnose-student <StudentName> HW<XX> - <question>      # Mode 2: Fetch + question context
/diagnose-student no notebook, question from HW<XX> - <question>  # Mode 3: No notebook
```

### Examples
- `/diagnose-student tmp/Homework_04_Assignment_Winny.ipynb` — local file
- `/diagnose-student Sarah HW05` — fetch Sarah Koch's HW05 notebook from CoCalc
- `/diagnose-student Kyle HW03 - my model accuracy is only 60%` — fetch + context
- `/diagnose-student no notebook, question from HW05 - how do I evaluate on test set?` — question only

## Instructions
Please diagnose the student's notebook issue following this workflow:

### Step 0: Detect Invocation Mode

Parse the arguments to determine which mode to use:

1. **Mode 1 (Local file)**: Argument contains a file path (has `/` or ends in `.ipynb`) AND the file exists locally. Proceed to Step 1.
2. **Mode 2 (CoCalc fetch)**: Argument contains a recognizable student name (matches the class roster) AND a homework number (`HW05`, `hw 5`, `Homework_05`, `HW5`, etc.). Proceed to Step 0a.
3. **Mode 3 (Question only)**: Argument contains a homework reference but no local file path and no roster match (e.g., starts with "no notebook" or contains only a homework number and question text). Skip to Step 2 (infer HW# from arguments), then Step 4, then Step 5 (the question is in the arguments — still ask for additional context), then Step 6 (limited diagnosis from solution + question), then Step 7.

### Step 0a: Fetch Notebook from CoCalc (Mode 2 only)

#### Roster Lookup
Read the class roster from: `/home/jbaggett/DS776_new/Developer/OpenRouter/OpenRouter_CoCalc/names.csv`
- Columns: `First Name`, `Last Name`, `Email`, `Project ID`
- **Skip the instructor row** (Jeff B)
- Match the student name from the arguments using this priority:
  1. Exact first name match (case-insensitive)
  2. Exact last name match (case-insensitive)
  3. Full name match (first + last, case-insensitive)
  4. Prefix match (e.g., "Sar" matches "Sarah")
  5. Token match (any word in the input matches first or last name)
- **If ambiguous** (multiple matches): list the matching students and use AskUserQuestion to let the instructor pick
- **If no match**: list all students and use AskUserQuestion to let the instructor choose

#### Parse Homework Number
Extract the homework number from the arguments. Accept formats: `HW05`, `HW5`, `hw 5`, `Homework_05`, `homework 5`. Zero-pad to 2 digits.

#### CoCalc API Calls
Use `curl` via the Bash tool for all API calls. The API key is in `$COCALC_API_KEY`.

**1. Start the student's project** (always do this first — projects may be stopped):
```bash
curl -s -X POST "https://cocalc.com/api/v2/projects/start" \
  -H "Content-Type: application/json" \
  -u "$COCALC_API_KEY:" \
  -d '{"project_id": "<PROJECT_ID>"}'
```
Wait a few seconds after starting for the project to initialize.

**2. List the homework directory** to find the notebook:
```bash
curl -s -X POST "https://cocalc.com/api/v2/exec" \
  -H "Content-Type: application/json" \
  -u "$COCALC_API_KEY:" \
  -d '{"project_id":"<PROJECT_ID>","command":"ls","args":["-la","Homework/Homework_<XX>/"],"bash":false,"timeout":15,"err_on_exit":false}'
```

**3. Fetch the notebook content:**
```bash
curl -s -X POST "https://cocalc.com/api/v2/exec" \
  -H "Content-Type: application/json" \
  -u "$COCALC_API_KEY:" \
  -d '{"project_id":"<PROJECT_ID>","command":"cat","args":["Homework/Homework_<XX>/Homework_<XX>_Assignment.ipynb"],"bash":false,"timeout":30,"err_on_exit":false}'
```

**4. Save locally** to `tmp/<FirstName>_Homework_<XX>_Assignment.ipynb`
- Create the `tmp/` directory if it doesn't exist
- Write the fetched JSON content to the file
- Report success: "Fetched {FirstName}'s HW{XX} notebook from CoCalc, saved to tmp/{filename}"

**5. Then continue** with Steps 1-7 using the locally saved file.

#### Error Handling
- **API key not set**: Tell the instructor to add `COCALC_API_KEY` to `~/.bashrc` and source it
- **Student not in roster**: List all students from the CSV and ask the instructor to pick
- **Project won't start**: Report the error and suggest the student may need to log into CoCalc
- **Homework directory missing**: Tell the instructor the student hasn't created the homework directory yet
- **Notebook not found**: List what IS in the homework directory so the instructor can see what the student has
- **Content too large or truncated**: Try fetching with gzip+base64:
  ```bash
  curl -s -X POST "https://cocalc.com/api/v2/exec" \
    -H "Content-Type: application/json" \
    -u "$COCALC_API_KEY:" \
    -d '{"project_id":"<PROJECT_ID>","command":"bash","args":["-c","gzip -c Homework/Homework_<XX>/Homework_<XX>_Assignment.ipynb | base64"],"bash":false,"timeout":30,"err_on_exit":false}'
  ```
  Then decode locally: `echo '<base64_content>' | base64 -d | gunzip > tmp/<filename>`

### Step 1: Read the Student Notebook
- **Mode 1**: Read the notebook at the provided path completely
- **Mode 2**: Read the notebook from `tmp/<FirstName>_Homework_<XX>_Assignment.ipynb` (saved in Step 0a)
- **Mode 3**: Skip this step (no notebook available)

For Modes 1 and 2:
- Note all code cells, their outputs, and any error messages
- Pay attention to model architectures, training loops, hyperparameters, and results

### Step 2: Infer the Homework Number
- **Mode 2**: Already known from arguments (parsed in Step 0a)
- **Mode 3**: Already known from arguments (parsed in Step 0)
- **Mode 1**: Determine using these strategies in order:
  1. **From filename**: Look for `Homework_XX` or `Homework_X` pattern in the filename (e.g., `Homework_03_Assignment-Listra.ipynb` -> HW 03)
  2. **From content**: If filename doesn't contain it, scan the notebook's markdown cells for title patterns like "# Homework XX", "Homework XX Assignment", or references to specific homework numbers
  3. **Zero-pad** the number to 2 digits (e.g., 3 -> 03)

If the homework number cannot be determined, ask the instructor.

### Step 3: Infer the Student Name
- **Mode 2**: Already known from roster lookup in Step 0a (use first name)
- **Mode 3**: Use "student" as default, or extract from arguments if provided
- **Mode 1**: Extract the student name from the filename:
  - Look for text after `Assignment[-_]` and before `.ipynb` or other suffixes
  - Examples: `Homework_03_Assignment-Listra.ipynb` -> "Listra", `Homework_04_Assignment_Winny.ipynb` -> "Winny"
  - Strip suffixes like `_WIP`, `_v2`, `_GRADE_THIS_ONE`, `_final`
  - If no name can be extracted, use "student" as the default

### Step 4: Load Reference Materials
Using the inferred homework number (HW), read these files:
- **Instructor solution**: `Developer/Solutions/Homework_{HW}/Solutions_{HW}.ipynb`
- **Assignment**: `Homework/Homework_{HW}/Homework_{HW}_Assignment.ipynb`
- **Helpers** (if they exist): `Homework/Homework_{HW}/Homework_{HW}_Helpers.py`
- **Lesson notebooks** (skim if needed): `Lessons/Lesson_{HW}_*/` for relevant patterns

### Step 5: Ask the Instructor for Context
**IMPORTANT**: Before diagnosing, ask the instructor (using AskUserQuestion or conversational prompt):
- "What did the student say? Please paste their forum question, email, or describe their issue."
- "Any additional observations or hints about where the problem might be?"

**Mode-aware behavior:**
- **Mode 2 with question in arguments** (e.g., `Sarah HW05 - how do I evaluate on test set?`): The question is already provided. Still ask: "You mentioned: '{question}'. Any additional context or observations?"
- **Mode 3**: The question is in the arguments. Still ask: "You mentioned: '{question}'. Any additional context?"
- **Modes 1 & 2 without question**: Ask both questions as written above.

Wait for the instructor's response before proceeding to diagnosis. The student's own description of their problem is essential for a targeted, helpful response.

### Step 6: Diagnose Issues
Compare the student's code against the instructor solution systematically:

**For Modes 1 and 2 (notebook available):**

*Code Analysis:*
- Compare model architectures (layers, dimensions, activation functions)
- Check for layer reuse bugs (same nn.Module instance called multiple times)
- Verify training loops, loss functions, optimizers, and hyperparameters
- Check data loading and preprocessing steps
- Look for missing or incorrect function calls
- Examine import statements and setup code

*Output Analysis:*
- Compare accuracy/loss metrics against expected values from the solution
- Check parameter counts (use torchinfo/summary output if present)
- Look for `(recursive)` markers in model summaries (indicates layer reuse)
- Note any error tracebacks or warnings

**For Mode 3 (no notebook):**
- Base diagnosis on the instructor solution and the student's question
- Identify the most likely issues based on common patterns for that homework
- Provide general guidance referencing the solution approach

**Common Issue Patterns to Watch For:**
- Layer reuse (same nn.Linear/nn.BatchNorm called multiple times)
- Wrong model architecture (missing layers, wrong dimensions)
- Incorrect loss function for the task
- Missing data augmentation or preprocessing
- Wrong learning rate or optimizer settings
- Not using pretrained weights when required
- Forgetting to freeze/unfreeze layers in transfer learning
- Incorrect tokenizer usage in NLP tasks
- Missing .to(device) calls
- Training in eval mode or evaluating in train mode

### Step 7: Generate the Response
**Output path by mode:**
- **Mode 1**: `{notebook_dir}/{student_name}_hw{HW}_response.md` (same directory as the student's notebook)
- **Mode 2**: `tmp/{student_name}_hw{HW}_response.md`
- **Mode 3**: `tmp/student_hw{HW}_response.md`

**Tone and Style Guidelines (CRITICAL - follow these exactly):**
- **Friendly but succinct** — don't over-explain or pad with filler. Get to the point quickly.
- **Brief acknowledgment** — a sentence or two max, then straight into the diagnosis
- **Explain the WHY** — explain why the bug causes what they're seeing, but keep it tight. Use an analogy only if it genuinely clarifies.
- **Provide corrected code** — include copy-paste-ready code snippets with the fix
- **Set expectations** — tell them what results to expect after fixing (accuracy ranges, parameter counts, etc.)
- **Keep it focused** — address the specific issue they asked about. Don't do a full code review unless asked.
- **Sign off as "Dr. B"** — end with an invitation to follow up at office hours or on Piazza
- **Avoid verbose paragraphs** — prefer short paragraphs, bold headers, and code blocks. Students skim; make it scannable.

**Response Structure:**
```markdown
Hi {StudentName},

[1-2 sentence acknowledgment]

**[Bold summary of the core issue]**

[Concise explanation of what's wrong and why, with their problematic code snippet]

**The fix:**

[Corrected code snippet]

[Brief note on what to expect after fixing — accuracy, params, etc.]

[Any other minor issues, kept brief]

Let me know if you have questions at office hours or on Piazza.

Dr. B
```

### Processing Steps Summary:
1. **Step 0**: Detect mode from arguments (local file / CoCalc fetch / question only)
2. **Step 0a** (Mode 2 only): Look up student in roster, fetch notebook from CoCalc, save to `tmp/`
3. **Step 1**: Read student notebook (Modes 1 & 2) or skip (Mode 3)
4. **Step 2**: Infer homework number (may already be known from arguments)
5. **Step 3**: Infer student name (may already be known from roster)
6. **Step 4**: Load instructor solution, assignment, and helpers
7. **Step 5**: Ask instructor for student's question and additional context
8. **Step 6**: Compare student code to solution — identify bugs and issues
9. **Step 7**: Write friendly diagnostic response to appropriate output path
10. Display the output path and a brief summary of issues found

### Quality Checks:
- Response addresses the specific issue the student asked about
- Code snippets are correct and match the course's coding style
- Expected results (accuracy, params) are realistic and match the solution
- Tone is friendly, encouraging, and educational
- No condescending language — treat mistakes as learning opportunities
- Response is concise and focused (not a full code review unless warranted)
- Output file is saved to the correct location based on mode

Please execute this diagnostic workflow for the student notebook provided.

ARGUMENTS: see Usage above for the three invocation modes
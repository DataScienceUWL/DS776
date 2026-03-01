# Review Student Submissions

## Description
Reviews a batch of student homework submissions (HTML notebook exports from Canvas) against the instructor solution. Generates a structured markdown report with per-student findings, issues, highlights, and pasteable LMS feedback remarks.

## Usage
```
/review_submissions <path_to_zip_or_folder>
```
Examples:
- `/review_submissions Developer/Submissions/submissions.zip`
- `/review_submissions /tmp/hw03_submissions/`
- `/review_submissions ~/Downloads/homework_submissions.zip`

## Instructions
Please review the student submissions following this workflow:

### Step 1: Extract Submissions
1. Run the extraction script on the provided zip file or folder:
   ```bash
   python3 Developer/Scripts/extract_notebook_html.py <input_path> --output /tmp/hw_extracted.json
   ```
2. Note the homework number inferred by the script
3. Note the student text files directory (output alongside the JSON)

### Step 2: Load Reference Materials
Using the inferred homework number (HW):
- **Instructor solution**: Read `Developer/Solutions/Homework_{HW}/Solutions_{HW}.ipynb`
- **Assignment**: Read `Homework/Homework_{HW}/Homework_{HW}_Assignment.ipynb` (for requirements and point values)
- Understand the expected structure: what parts are there, what code should look like, what accuracy/results are expected

### Step 2b: Load Prior Reviews (Longitudinal Context)
Look in `Developer/Submissions/` for up to 2 previous review files:
- Check for `Homework_{HW-1}_Review.md` and `Homework_{HW-2}_Review.md` (zero-padded)
- If found, read them and extract each student's **Findings** and **LMS Remarks** sections
- Build a quick lookup of prior feedback keyed by student name/ID
- If no prior reviews exist, skip this step

This prior context will be used in Step 3 to:
- Identify recurring issues across assignments (same mistake repeated)
- Avoid giving the same feedback verbatim
- Acknowledge improvement when a student fixed a previously flagged issue
- Escalate tone slightly if a student continues making the same mistake after feedback

### Step 3: Review Each Student
For each student text file in the extracted output directory, read their file and compare their work against the solution. Use the Task tool with subagent_type="general-purpose" to process students in parallel batches of 3-4 to speed things up.

**For each student, evaluate:**

#### Correctness
- Does their code produce correct results?
- Are model architectures correct (right layers, dimensions, activations)?
- Are training parameters appropriate (optimizer, lr, epochs)?
- Do their outputs show reasonable metrics (accuracy, loss)?
- Are there error tracebacks in their outputs?
- Did they complete all required parts?

#### Course Pattern Compliance
- Are they using `introdl` functions? Look for:
  - `train_network()` for training (not custom training loops)
  - `plot_training_metrics()` for plotting (not manual matplotlib)
  - `load_results()` / `load_model()` for checkpoint handling
  - `config_paths_keys()` for path setup
  - `evaluate_classifier()` for model evaluation
  - `get_device()` for device selection
- Do they follow the notebook structure from lessons?

#### AI-Generated Code Indicators
Flag code that appears likely AI-generated. Most students in this course are NOT advanced programmers, so watch for:
- `try`/`except` blocks (especially elaborate error handling)
- Type hints and docstrings on student-written functions
- Logging module usage
- Context managers beyond basic `with open()`
- Overly defensive validation/assertion patterns
- Unusually sophisticated list comprehensions or generators
- Perfect PEP-8 style with comprehensive comments that sound like documentation
- Code patterns that don't match the course's established style
- Functions or classes not requested by the assignment

**Important:** Don't flag standard patterns taught in the course (e.g., using nn.Module, DataLoader). Only flag things that go beyond what lessons demonstrate.

#### Longitudinal Patterns (if prior reviews available)
- Compare this student's current work against their prior feedback
- **Recurring issues**: If the student was told about a problem before and it reappears, flag it clearly (e.g., "Same ReLU-after-output issue flagged in HW03")
- **Improvement**: If a previously flagged issue is now fixed, acknowledge it briefly
- **Don't repeat yourself**: If prior feedback covered a topic thoroughly, reference it rather than re-explaining (e.g., "See my HW03 comments on DataLoader ordering" rather than a full re-explanation)
- **Escalation**: If the same fundamental mistake appears a third time, suggest office hours directly

#### Notable Work
- Did the student do anything particularly well?
- Creative approaches or insightful analysis?
- Strong reflection answers?
- Going above and beyond the requirements?

#### Completeness
- Any unanswered questions (look for `YOUR ANSWER HERE` still present)?
- Any empty code cells or template-only cells?
- Did they export and save all outputs?
- Did they include the reflection section?

### Step 4: Generate the Report
Write the report to `Developer/Submissions/Homework_{HW}_Review.md`

**Report structure:**

```markdown
# Homework {HW} Submission Review

**Date:** {today's date}
**Students Reviewed:** {count}
**Solution Reference:** Developer/Solutions/Homework_{HW}/Solutions_{HW}.ipynb

## Summary
- X students with clean submissions
- X students with issues needing attention
- X late submissions
- Common patterns observed: [brief notes]

---

## {Student Name} ({student_id})
{🕐 LATE SUBMISSION if applicable}

### Findings
- [Bulleted list of specific observations]
- [Issues found, deviations from solution, errors]
- [Things done well]
- [AI-generated code concerns if any]

### LMS Remarks
{Pasteable first-person feedback paragraph - friendly, constructive, specific}

---
```

### Tone and Style for LMS Remarks (CRITICAL)
Write as if you are Dr. B speaking directly to the student:
- First person: "I noticed...", "Great work on...", "I'd suggest..."
- Friendly and encouraging — acknowledge effort and what they did well
- Specific about issues — reference exact parts or code
- Constructive — frame problems as learning opportunities
- Concise — 2-4 sentences for clean submissions, more for students with issues
- Diplomatic about AI usage — "I noticed some advanced patterns like try/except blocks — make sure you understand each construct and could explain it"

### Example LMS Remarks

**Clean submission:**
> Great work on this assignment! Your model architectures look solid and your training results are right in the expected range. I especially liked your analysis in Part 5 comparing the different optimizers. Keep it up!

**Submission with issues:**
> Good effort on this assignment. I noticed a couple of things: your model in Part 2 is using a custom training loop instead of the `train_network()` function we provide — using our function will save you time and give you consistent plots. Also, your OneCycleLR scheduler wasn't quite set up correctly (the max_lr should match your optimizer's lr). Take a look at the L03_4 notebook for the pattern. Overall your analysis was thoughtful though!

**AI concerns:**
> Nice job completing all parts. I did notice some code patterns that go beyond what we've covered in class — things like the try/except error handling and type annotations. While these are good practices in general, for this course I want to make sure you understand the core PyTorch patterns first. Try writing the training code using just the patterns from our lesson notebooks. Let me know if you have questions about any of the concepts!

**Recurring issue (longitudinal):**
> Good work completing all parts. One thing — I noticed you still have ReLU after the output layer, which I flagged on HW03 as well. With CrossEntropyLoss the final layer should produce raw logits with no activation. Please revisit my HW03 feedback on this and let me know if the explanation didn't click — happy to go over it at office hours.

**Improvement acknowledged (longitudinal):**
> Nice work here! I can see you fixed the DataLoader ordering issue from last time — data pipeline looks clean now. Your model architectures and training results are solid across all parts.

### Processing Steps Summary:
1. Run extraction script on the provided input
2. Note homework number and student text file directory
3. Read instructor solution and assignment for reference
4. Load up to 2 prior reviews from Developer/Submissions/ for longitudinal context
5. For each student: read their extracted text, compare to solution, check against prior feedback, note findings
6. Generate structured report with findings and LMS remarks per student (weaving in longitudinal observations)
7. Save report to Developer/Submissions/Homework_{HW}_Review.md
8. Display summary of findings

### Quality Checks:
- Every student has a section in the report
- Findings are specific (reference exact code/parts, not vague)
- LMS remarks are pasteable without editing
- Late submissions are flagged
- Report is organized alphabetically by student name
- Tone is consistently friendly and constructive
- AI-generated code patterns are noted diplomatically

Please execute this review workflow for the provided submissions.

ARGUMENTS: path to zip file or folder
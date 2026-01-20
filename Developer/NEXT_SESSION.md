# DS776 - Next Session Context

**Last Updated:** 2026-01-20
**Session Focus:** Spring 2026 Launch Preparation - Repo Cleanup Complete

---

## 🎯 Current Status: Repository Cleaned and Synced

### What Was Accomplished This Session (2026-01-20)

1. **Placeholder standardization** across all homework notebooks (HW01-HW12)
   - Code: `# YOUR CODE HERE` (uppercase, no END marker)
   - Imports: `# YOUR IMPORTS HERE`
   - Text: `📝 **YOUR ANSWER HERE:**`
   - Analysis: `📝 **YOUR ANALYSIS HERE:**`

2. **Removed pip install instructions** from HW05, HW06, HW07
   - All packages now in introdl pyproject.toml

3. **Added beautifulsoup4** to introdl dependencies (v1.6.63)

4. **Removed Colab notebooks** from HW11 and HW12

5. **Repository cleanup:**
   - Added `Homework/Homework_Tests/` to .gitignore and removed from repo
   - Removed `Lessons/Lesson_12_Summarization_v1/` (duplicate)
   - Removed `Lessons/Lesson_13_Project/` (renamed to Lesson_13_14_Project)

6. **CoCalc sync verified** - Fresh pull tested and working

---

## ✅ Quiz Specifications (Confirmed)

| Setting | Value |
|---------|-------|
| Points per quiz | 10 pts (1 pt per question) |
| Homework points | 40 pts (reduced from 50) |
| Quiz timing | Same deadline as homework |
| Attempts | 2 allowed |
| Feedback | Show correct answers after due date |
| L07 scope | NLPWT Chapters 1 AND 2 |

---

## ⚠️ Large PDF Strategy

**Problem:** IDL Chapter 2 PDF (~4.5MB) is too large for context.

**Solution:** Parse chapter PDFs in chunks:
1. Read PDF page by page or section by section
2. Extract key concepts for quiz questions
3. Never try to load entire chapter at once

This applies to any large chapter PDFs.

---

## 📋 Next Steps (After Questions Answered)

### Step 1: Install text-to-qti
```bash
pip install text-to-qti
mkdir -p Developer/Quizzes
```

### Step 2: Create First Quiz (L01)
1. Read IDL Chapter 1 (focus 1.2, 1.4, 1.5) and Chapter 2
2. Create `Developer/Quizzes/quiz_01.md` with 10 MC questions
3. Test conversion: `text-to-qti convert quiz_01.md -o quiz_01.zip`
4. Test import in Canvas sandbox

### Step 3: Process Lessons in Order
For each lesson (L01 → L12):
1. Create quiz aligned with assigned reading
2. Update homework notebook (remove reading Qs if present, add storage reminder)
3. Address any Spring_2026_updates.md issues for that lesson
4. Commit and push after each lesson complete

---

## 📁 Key Files for This Work

### Planning Documents
- `Developer/Spring_2026_Launch_Plan.md` - Master plan with all checklists
- `Developer/Spring_2026_updates.md` - Issues to address (reference)
- `Developer/TODO.md` - Progress tracking

### Textbooks (for quiz creation)
- `Developer/Textbooks/idlmam/` - Inside Deep Learning (L01-L06)
- `Developer/Textbooks/nlpwt/` - NLP with Transformers (L07-L12)

### Lesson Overviews (verify readings)
- `Lessons/Lesson_XX_*/L_XX_0_Overview.ipynb`

### Homeworks to Update
- `Homework/Homework_XX/Homework_XX_Assignment*.ipynb`

---

## 🗂️ Reading Assignments Reference

| Lesson | Textbook | Sections |
|--------|----------|----------|
| L01 | IDL | Ch 1 (1.2, 1.4, 1.5), Ch 2 |
| L02 | IDL | Ch 3 (through 3.5) |
| L03 | IDL | 3.6, 5.1-5.3 |
| L04 | IDL | 6.1-6.5 |
| L05 | IDL | 13.1-13.3 |
| L06 | IDL | Ch 8 |
| L07 | NLPWT | Ch 1 AND Ch 2 |
| L08 | NLPWT | Ch 2 |
| L09 | - | Report (no quiz) |
| L10 | NLPWT | Ch 4 |
| L11 | NLPWT | Ch 5 |
| L12 | NLPWT | Ch 6 |

---

## ⚠️ Important Reminders

1. **Commit before editing** - Always commit current state before making changes
2. **Push frequently** - Don't accumulate unpushed commits
3. **Quiz alignment** - Every question must be answerable from assigned reading only
4. **Storage reminders** - Add to ALL homework notebooks (template in launch plan)

---

## 📚 Previous Session Summary (Fall 2025)

The previous session completed Lesson 11 v3 with class-based API approach:
- LLMClient and ChatbotClient classes
- 9 well-structured sections
- Updated homework Part 2 to class-based
- See git history for details

---

**CURRENT SESSION:** Creating Quiz 01 from IDL Chapters 1-2 (parsing in chunks)

# DS776 - Next Session Context

**Last Updated:** 2026-01-15
**Session Focus:** Spring 2026 Launch Preparation

---

## 🎯 Current Status: Planning Complete, Ready to Execute

### What Was Accomplished This Session

1. **Created comprehensive launch plan:** `Developer/Spring_2026_Launch_Plan.md`
   - Detailed checklist for all lessons (L01-L12)
   - Quiz creation workflow using text-to-qti
   - Homework update requirements
   - Issues from Spring_2026_updates.md prioritized

2. **Updated TODO.md** with Spring 2026 priorities
   - New checklist format for tracking progress
   - Organized by phase (quizzes, homework, issues)

3. **Updated CLAUDE.md** with quiz workflow
   - Added "Generating Canvas Reading Quizzes" section
   - Documented text-to-qti format and usage
   - Added quiz alignment requirements

---

## ❓ Questions Awaiting Instructor Answers

Before proceeding, need decisions on:

1. **Point structure:** Should each week have 50 pts homework + 10 pts quiz = 60 total? Or redistribute?

2. **Quiz timing:** Should reading quizzes be due before the homework deadline, or at the same time?

3. **Quiz attempts:** How many attempts should students get on reading quizzes?

4. **Quiz feedback:** Should students see correct answers immediately, or after the due date?

5. **L07 reading scope:** L07 overview mentions "Chapters 1-2" but HW07 questions only cover Chapter 1. Should the quiz cover just Ch 1, or both Ch 1-2?

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
| L07 | NLPWT | Ch 1 |
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

**NEXT SESSION PRIORITY:** Get answers to instructor questions, then begin L01 quiz + homework updates

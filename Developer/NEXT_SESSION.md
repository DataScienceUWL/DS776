# DS776 - Next Session Context

**Last Updated:** 2025-10-28
**Session Focus:** Review Lesson 11 v2 & Homework 11 v2

---

## 🔥 NEXT SESSION: Review and Test Lesson 11 v2 Materials (2025-10-28+)

### WHAT WAS COMPLETED THIS SESSION:

**Lesson 11 Text Generation v2** - ✅ COMPLETE
- Created new version in `Lessons/Lesson_11_Text_Generation_v2/`
- Updated Section 1 chronology with 2025 developments (agentic AI, reasoning models)
- Updated Section 2 model table with o3-mini, Qwen 2.5, DeepSeek-V3
- Condensed Section 4 (Training Pipeline) to high-level overview
- Enhanced Section 5 with 4 new subsections:
  - 5.1 Environment Variables and API Keys
  - 5.2 OpenRouter: Multi-Provider API Access
  - 5.3 Building Custom API Helper Functions (complete code example)
  - 5.4 LangChain Framework (brief mention)
- Condensed Section 8 (Adapting LLMs) to brief overview
- Enhanced Section 7 with OpenRouter API examples and 70B model demonstrations
- Created `L11_2_Background_Supplement.ipynb` with stubs for condensed content
- Updated `L11_0_Overview.ipynb` with new learning objectives

**Homework 11 v2** - ✅ COMPLETE
- Created new version in `Homework/Homework_11_v2/`
- 5 Reading Questions (10 points) from NLPWT Chapter 5:
  1. Autoregressive models and conditional text generation (2 pts)
  2. Log probabilities vs. regular probabilities (2 pts)
  3. Greedy search vs. beam search (2 pts)
  4. Temperature parameter in sampling (2 pts)
  5. Top-k and nucleus sampling methods (2 pts)
- 6 Technical Parts (40 points):
  - Part 1: Decoding strategies comparison (10 pts)
  - Part 2: Building API helper functions (8 pts)
  - Part 3: Model size comparison (3B vs 8B vs 70B) (8 pts)
  - Part 4: Creative text generation application (8 pts)
  - Part 5: Analysis and comparison (4 pts)
  - Part 6: Reflection (2 pts)
- Added `Storage_Cleanup.ipynb` utility

---

## 🎯 NEXT SESSION PRIMARY GOALS:

### 1. Review Lesson 11 v2 Materials
**Location:** `Lessons/Lesson_11_Text_Generation_v2/`

**Review checklist:**
- [ ] Read through updated L11_1_Text_Generation.ipynb
  - Section 1: Is chronology accurate and up-to-date?
  - Section 2: Are model listings current?
  - Section 5: Does API content flow well? Are code examples clear?
  - Section 7: Do OpenRouter examples work? Are 70B models appropriate?
- [ ] Review L11_2_Background_Supplement.ipynb
  - Check stub content is appropriate
  - Decide if/when to expand stubs
- [ ] Review L11_0_Overview.ipynb
  - Verify learning objectives align with updated content
  - Check homework ideas match new assignment

### 2. Review Homework 11 v2 Assignment
**Location:** `Homework/Homework_11_v2/Homework_11_Assignment_v2.ipynb`

**Review checklist:**
- [ ] Reading questions: Are they appropriate for Chapter 5 content?
- [ ] Part 1 (Decoding): Is the task clear and achievable?
- [ ] Part 2 (API Helpers): Will students understand the requirements?
- [ ] Part 3 (Model Sizes): Is model selection reasonable for students?
- [ ] Part 4 (Creative Application): Are the three options clear?
- [ ] Part 5 (Analysis): Are the questions thought-provoking?
- [ ] Part 6 (Reflection): Standard format verified?

### 3. Decide on Solutions and Testing
**Two options:**

**Option A: Create Full Solutions**
- Create `Developer/Solutions/Homework_11_v2/Solutions_11_v2.ipynb`
- Include complete working code for all 6 parts
- Test on compute server with 48GB GPU (70B models)
- Verify API calls work with OpenRouter

**Option B: Defer Solutions**
- Test individual code snippets as needed
- Create solutions when homework is assigned to students
- Focus on completing Lesson 10 and Lesson 12 instead

---

## 📋 Files Created This Session

### Lesson 11 v2:
- `Lessons/Lesson_11_Text_Generation_v2/L11_0_Overview.ipynb` (updated)
- `Lessons/Lesson_11_Text_Generation_v2/L11_1_Text_Generation.ipynb` (major updates)
- `Lessons/Lesson_11_Text_Generation_v2/L11_2_Background_Supplement.ipynb` (new)

### Homework 11 v2:
- `Homework/Homework_11_v2/Homework_11_Assignment_v2.ipynb` (new)
- `Homework/Homework_11_v2/Storage_Cleanup.ipynb` (copied from HW10)

### Documentation:
- `Developer/TODO.md` (updated with Lesson 11 completion status)

---

## 💡 Key Design Decisions Made

### Content Condensation Strategy
**Problem:** Original lesson was very long with extensive background material.

**Solution:**
1. Condensed pre-2017 history in Section 1 to 2 paragraphs
2. Moved detailed training pipeline to supplement (Section 4 → brief overview)
3. Moved RAG and fine-tuning details to supplement (Section 8 → brief overview)
4. Created L11_2_Background_Supplement.ipynb for students wanting deeper knowledge

**Rationale:** Focus on practical skills (decoding, APIs, models) over historical/theoretical detail.

### API Content Addition
**What was added:**
- Environment variable management and API key security
- OpenRouter as multi-provider gateway
- Complete code example for building custom API helper function
- Brief LangChain mention (no deep dive)

**Rationale:** Students need practical API skills; these are more immediately useful than deep training knowledge.

### Model Size Progression
**Models demonstrated:**
- 3B: Llama-3.2-3B-Instruct (4-bit, ~2GB VRAM)
- 8B: Llama-3.1-8B-Instruct (4-bit, ~5GB VRAM)
- 70B: Llama-3.3-70B-Instruct (4-bit, ~35GB VRAM)

**Rationale:** Show full spectrum from fast/lightweight to slow/accurate. 70B fits on RTX A6000 (48GB).

### Homework Structure
**Reading Questions:** Based directly on NLPWT Chapter 5 (follows established pattern from HW 7-10)

**Technical Parts:** Progression from basic to advanced:
1. Compare decoding methods (foundational understanding)
2. Build API helpers (practical coding skill)
3. Compare model sizes (resource trade-offs)
4. Creative application (synthesis and creativity)
5. Analysis (critical thinking)
6. Reflection (metacognition)

---

## 🔍 Review Focus Areas

### Content Accuracy
- [ ] Are 2025 model releases accurately represented?
- [ ] Is agentic AI description current and accurate?
- [ ] Are reasoning model capabilities correctly stated?
- [ ] Are code examples syntactically correct?

### Pedagogical Flow
- [ ] Does lesson build logically from concepts to application?
- [ ] Are API examples clear for students new to API usage?
- [ ] Is homework difficulty appropriate for this stage of course?
- [ ] Do learning objectives match actual lesson content?

### Technical Feasibility
- [ ] Can students load 70B models on compute servers?
- [ ] Will API examples work with student OpenRouter credits?
- [ ] Are decoding strategy examples achievable in reasonable time?
- [ ] Is homework workload reasonable (estimate 8-10 hours)?

---

## 📝 Questions for Instructor

### Content Questions:
1. **Agentic AI coverage**: Is the treatment of agentic AI at the right depth? Too much? Too little?
2. **API vs Local balance**: Does Section 5 (APIs) vs Section 7 (local models) strike the right balance?
3. **Background supplement**: Should stubs be expanded now, or leave as optional future work?
4. **70B models**: Are 70B models appropriate for students, or should we stick to 8B maximum?

### Homework Questions:
1. **Creative application options**: Are the 3 options (Story, Dialogue, Style) all equally good?
2. **Reading question difficulty**: Are questions too easy? Too hard?
3. **Point allocation**: Is 10 pts for decoding comparison appropriate, or too much?
4. **Solutions needed**: Should we create full solutions now, or wait until assignment?

### Process Questions:
1. **Should we test Lesson 11 before completing Lesson 10?** (Currently Lesson 10 is incomplete)
2. **Priority**: Finish Lesson 10 first, or test Lesson 11 thoroughly first?
3. **Custom GPT Bundle**: When should we start packaging Lessons 7-12 for Custom GPT?

---

## 📚 Reference Materials

### Textbook:
- `Developer/Textbooks/nlpwt/Chapter_05_Text_Generation.pdf` - Used for reading questions

### Previous Lessons:
- `Lessons/Lesson_11_Text_Generation/` - Original lesson (for comparison)
- `Homework/Homework_11/` - Original homework (for comparison)

### Similar Homework:
- `Homework/Homework_10/Homework_10_Assignment.ipynb` - Most recent format with reading questions
- `Homework/Homework_08/Homework_08_Assignment.ipynb` - Reference for structure

### Documentation:
- `Developer/TODO.md` - Overall project status
- `Developer/NEXT_SESSION.md` - This file

---

## 🚀 Quick Start for Next Session

### If Reviewing Lesson 11:
1. Open `Lessons/Lesson_11_Text_Generation_v2/L11_1_Text_Generation.ipynb`
2. Read through updated sections (1, 2, 4, 5, 7, 8)
3. Check code examples for correctness
4. Review learning objectives in L11_0_Overview.ipynb
5. Provide feedback on content, clarity, and difficulty

### If Reviewing Homework 11:
1. Open `Homework/Homework_11_v2/Homework_11_Assignment_v2.ipynb`
2. Read through all 6 parts
3. Assess difficulty and time requirements
4. Check reading questions against Chapter 5
5. Provide feedback on clarity and workload

### If Creating Solutions:
1. Create new notebook in `Developer/Solutions/Homework_11_v2/`
2. Work through all 5 reading questions
3. Implement all 6 technical parts
4. Test code on compute server
5. Document any issues or improvements needed

---

## 🎯 Remaining Work (If Needed)

### Must Do:
- [ ] Instructor review of Lesson 11 v2 materials
- [ ] Instructor review of Homework 11 v2 assignment
- [ ] Decision on solutions (create now vs. later)

### Should Do:
- [ ] Test API examples with OpenRouter
- [ ] Test 70B model loading on compute server
- [ ] Verify all code examples execute without errors

### Could Do:
- [ ] Expand background supplement stubs
- [ ] Add more API provider examples
- [ ] Create video walkthroughs for new content

---

**STATUS: Ready for Instructor Review**

All Lesson 11 v2 and Homework 11 v2 materials are complete and pushed to repository. Instructor should review materials and provide feedback before proceeding with solutions or testing.

**Next Major Milestone:** Complete Lesson 10 updates (Helpers + Homework)
**Future Milestone:** Complete Lesson 12 (Summarization)
**High Priority:** Custom GPT Bundle for Lessons 7-12

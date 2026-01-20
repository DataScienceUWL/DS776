# DS776 Course Development TODO

**Last Updated:** 2026-01-20

---

## 🔥 Current Priority: Spring 2026 Launch Preparation

**Critical Path:**
1. **Canvas Reading Quizzes** - Convert reading questions to Canvas multiple choice quizzes
2. **Homework Updates** - Remove embedded reading questions, add storage reminders
3. **Consistency Pass** - Address issues from Spring_2026_updates.md
4. **Testing** - Test all materials before semester start

**See:** `Developer/Spring_2026_Launch_Plan.md` for detailed plan

### Quiz Specifications (Confirmed 2026-01-16)
- **Points:** 40 pts homework + 10 pts quiz per lesson
- **Quiz timing:** Same deadline as homework
- **Attempts:** 2 attempts allowed
- **Feedback:** Correct answers shown after due date
- **L07 scope:** Both NLPWT Chapters 1 and 2

---

## 🎯 Spring 2026 Launch Checklist

### Phase 1: Reading Quiz Creation (11 quizzes)
- [ ] Install text-to-qti tool: `pip install text-to-qti`
- [ ] Create `Developer/Quizzes/` directory
- [ ] L01: Create quiz_01.md (IDL Ch 1-2)
- [ ] L02: Create quiz_02.md (IDL Ch 3)
- [ ] L03: Create quiz_03.md (IDL 3.6, 5.1-5.3)
- [ ] L04: Create quiz_04.md (IDL 6.1-6.5)
- [ ] L05: Create quiz_05.md (IDL 13.1-13.3)
- [ ] L06: Create quiz_06.md (IDL Ch 8)
- [ ] L07: Create quiz_07.md (NLPWT Ch 1)
- [ ] L08: Create quiz_08.md (NLPWT Ch 2)
- [ ] L10: Create quiz_10.md (NLPWT Ch 4)
- [ ] L11: Create quiz_11.md (NLPWT Ch 5)
- [ ] L12: Create quiz_12.md (NLPWT Ch 6)

### Phase 2: Homework Updates
- [x] Standardize placeholder formats across all HW notebooks (2026-01-20)
- [x] Remove pip install instructions from HW05, HW06, HW07 (2026-01-20)
- [x] Remove Colab notebooks from HW11, HW12 (2026-01-20)
- [x] Add beautifulsoup4 to introdl dependencies (v1.6.63) (2026-01-20)
- [ ] Add storage reminder template to all HW notebooks
- [ ] HW07: Remove embedded reading questions
- [ ] HW08: Remove embedded reading questions
- [ ] Update point allocations in all homework headers
- [ ] Verify reflection sections present (2 pts each)

### Phase 2b: Repository Cleanup (2026-01-20) ✅ COMPLETE
- [x] Add Homework/Homework_Tests/ to .gitignore and remove from repo
- [x] Remove Lessons/Lesson_12_Summarization_v1/ (duplicate folder)
- [x] Remove Lessons/Lesson_13_Project/ (renamed to Lesson_13_14_Project)
- [x] Sync and verify CoCalc can pull clean copy

### Phase 3: Spring_2026_updates.md Issues
**Priority 1 (Must Fix):**
- [ ] L01: Use train/validation terminology consistently
- [x] L05: Verify timm in introdl package ✅ (already in pyproject.toml)
- [x] L06: Verify ultralytics in introdl package ✅ (already in pyproject.toml)
- [ ] L07: Document max_tokens for llm_generate
- [ ] L08: Update HW to reference OpenRouter models
- [ ] L10: Ensure best_model saved correctly
- [ ] L11: Verify reading questions align with Ch 5

**Priority 2 (Should Fix):**
- [ ] L01: Add PyTorch pipeline overview
- [ ] L02: Improve train_network documentation
- [ ] L03: Better scheduler explanation
- [ ] L04: More residual connection scaffolding
- [ ] L05: Layer indexing/freezing examples
- [ ] L07: Add f-string prompt examples

---

## ✅ Fall 2025 Completed Work

### Lesson 10 Completion + Custom GPT Bundle (Previous Priority)
1. ✅ **Lessons 7-9 Complete** - Lesson 7, 8, 9 updated and tested
2. ✅ **Homework 7-8 Complete** - Reading questions + new API
3. ✅ **Lesson 11 v2 Complete** - Text Generation updated with 2025 content, APIs, 70B models
4. ✅ **Homework 11 v2 Complete** - Ready for review (solutions & testing needed)
5. ✅ **Lesson 12 Complete** - Summarization updated with auto-update, flat imports, 2025 models
6. ✅ **Homework 12 Complete** - Reading questions, hint cells, production pipeline task

---

## Phase 0: Infrastructure ✅ COMPLETE (v1.6.7-1.6.8)

### ✅ Module Structure Cleanup (v1.6.6)
- [x] Removed duplicated visualization functions from nlp.py
- [x] Restored __init__.py to import viz functions from generation.py
- [x] Deleted nlp_orig.py (all useful code migrated)
- [x] Bumped version to 1.6.6

### ✅ JSON Capability System (v1.6.7)
- [x] Created automated test script for JSON capabilities
- [x] Enhanced openrouter_models.json with JSON metadata
- [x] Updated llm_generate() to intelligently use JSON modes
- [x] Added get_model_metadata() and enhanced llm_list_models()
- [x] Tested all features - verified working
- [x] Bumped version to 1.6.7

### ✅ New llm_generate API Developed (v1.6.8)
- [x] Simplified API: `llm_generate('model', prompt, mode='text'/'json')`
- [x] Built-in cost tracking and JSON schema support
- [x] Automatic fallbacks and error handling
- [x] Tested in `Developer/openrouter_json_generation_master_v2.ipynb`

### ✅ OpenRouter Models Curated
- [x] 16 models selected (commercial + open-source)
- [x] FREE tier options included (gpt-oss-20b, llama-3.2-3b, deepseek-v3-free)
- [x] Default: `gemini-flash-lite` (fast, cheap, full JSON support)
- [x] File: `Lessons/Course_Tools/introdl/src/introdl/openrouter_models.json`

---

## Phase 1a: OpenRouter Encrypted Key Deployment ✅ **COMPLETE**

**Status:** All 28 students have working OpenRouter API keys deployed

**Completed:** 2025-10-08

### Deployment Checklist

#### ✅ Completed
- [x] New API keys generated from OpenRouter (28-31 keys)
- [x] Excel file updated: `Developer/OpenRouter/2025_Fall_DS776_Openrouter_API_Keys.xlsx`
- [x] Encryption script ready: `Developer/OpenRouter/OpenRouter_CoCalc/generate_encrypted_mapping.py`
- [x] Student distribution script ready: `Lessons/Course_Tools/distribute_openrouter_key.py`
- [x] Deployment guide created: `Developer/OpenRouter/ENCRYPTED_DEPLOYMENT_GUIDE.md`

#### ✅ All Deployment Steps Complete

- [x] **Export student roster from CoCalc with Project IDs**
  - All 28 students have complete Project IDs (UUIDs)
  - Saved as: `Developer/OpenRouter/OpenRouter_CoCalc/names.csv`

- [x] **Run encryption script**
  - Generated: `encrypted_key_mapping.json` (28 entries)
  - Each key encrypted with its own project ID (AES-256-GCM)
  - Verified correct student count

- [x] **Deploy to GitHub Pages**
  - Deployed to: https://datascienceuwl.github.io/ds776-keys/encrypted_key_mapping.json
  - GitHub Pages live and accessible
  - JSON file verified and accessible

- [x] **Test with one student project ID**
  - Tested with Teena Alexander's project ID
  - Verified successful decryption
  - api_keys.env updated correctly
  - Backup file created successfully

- [x] **Push distribution script to all student projects**
  - Script location: `Lessons/Course_Tools/distribute_openrouter_key.py`
  - Already up-to-date with encrypted key system
  - All student projects have the script

- [x] **Run distribution script remotely via CoCalc**
  - All 28 students have keys distributed
  - All keys working in student projects

- [x] **Verify key distribution**
  - All students confirmed with working keys
  - Keys in `~/home_workspace/api_keys.env`
  - Backup files created

### Important Notes
- **Distribution script location**: `Lessons/Course_Tools/distribute_openrouter_key.py`
  - This is the CORRECT location for remote execution
  - Already configured for encrypted keys
  - Fetches from: https://datascienceuwl.github.io/ds776-keys/encrypted_key_mapping.json

- **Reproducible process**: See `Developer/OpenRouter/ENCRYPTED_DEPLOYMENT_GUIDE.md` for complete workflow

- **Security**: AES-256-GCM encryption, each key uses its project ID, safe for public hosting

---

## Phase 1b: Lesson 7 Updates ✅ **COMPLETE**

**Status:** All Lesson 7 notebooks updated successfully
**Completed:** 2025-10-09
**Dependencies:** Phase 1a complete ✅

### Old API → New API Changes

**Old:**
```python
config = llm_configure('mistral-7B')
response = llm_generate(config, prompt, system_prompt=system_prompt)
```

**New:**
```python
response = llm_generate('gemini-flash-lite', prompt,
                        system_prompt=system_prompt,
                        mode='text',  # or 'json'
                        estimate_cost=True)
```

### L07_0_Overview.ipynb - ✅ **COMPLETE**
- [x] Created lesson overview notebook
- [x] Topics, learning outcomes, readings, assessment
- [x] Matches format of other lesson overviews

### L07_1_Getting_Started.ipynb - ✅ **COMPLETE**
- [x] Remove all `llm_configure()` calls
- [x] Update to new `llm_generate()` signature
- [x] Change default model to `gemini-flash-lite`
- [x] Add `llm_list_models()` example
- [x] Add cost tracking examples with `llm_get_credits()`
- [x] Add OpenRouter-focused content ($15 credit per student)
- [x] Add comprehensive privacy discussion (DPAs, HIPAA, SOC 2)
- [x] Explain pre-configured API keys in `~/home_workspace/api_keys.env`
- [x] Show how to use any OpenRouter model (full model IDs)
- [x] Add examples: `gemini-flash-lite` (default), `openai/gpt-5-nano` (custom)

### L07_2_NLP_Tasks.ipynb - ✅ **COMPLETE**
- [x] Remove all `llm_configure()` calls
- [x] Update all NLP task examples (sentiment, NER, QA, translation, summarization)
- [x] Show both string parsing and JSON mode for NER (dual approach comparison)
- [x] Fixed JSON parsing error (LLM returns array not object)
- [x] Updated to new llm_generate() API throughout

### L07_Other_APIs.ipynb - ✅ **COMPLETE**
- [x] Removed JSON support section (deferred to Lesson 8)
- [x] Kept optional content for students wanting to use other API providers directly

---

## Phase 2: Homework 7 Updates ✅ **COMPLETE**

**Status:** Homework 7 fully updated and ready for students
**Completed:** 2025-10-09
**Dependencies:** Lesson 7 updates complete ✅

### Homework_07_Assignment.ipynb Updates

#### API Updates - ✅ **COMPLETE**
- [x] Remove all `llm_configure()` calls
- [x] Update to new `llm_generate()` API
- [x] Recommend `gemini-flash-lite` as default
- [x] Add cost tracking with `init_cost_tracking()`
- [x] Require at least one other model per task for comparison
- [x] Update all imports (removed llm_configure, added wrap_print_text)

#### Reading Questions (8 pts) - ✅ **COMPLETE**
Based on NLPWT Chapter 1: Hello Transformers:
- [x] Question 1: RNN encoder-decoder limitations and attention mechanisms (2 pts)
- [x] Question 2: ULMFiT's three steps (pretraining, domain adaptation, fine-tuning) (2 pts)
- [x] Question 3: GPT vs BERT differences (encoder vs decoder) (2 pts)
- [x] Question 4: Three transformer challenges (2 pts)

#### Reflection Question (2 pts) - ✅ **COMPLETE**
- [x] Standard reflection format matching Homework 1-6:
  - What was difficult to understand?
  - What resources supported learning most/least?

#### Tasks Preserved with Updates (40 pts total)
- [x] Task 1: Sentiment Analysis (6 pts) - HuggingFace pipeline + 2 LLM models
- [x] Task 2: Named Entity Recognition (6 pts) - HuggingFace pipeline + mode='json' with 2 LLMs
- [x] Task 3: Text Generation (6 pts) - HuggingFace pipeline + 2 LLM models
- [x] Task 4: Translation (6 pts) - HuggingFace pipeline + 2 LLM models
- [x] Task 5: Summarization (8 pts) - The Bitter Lesson with HuggingFace + 2 LLMs
- [x] Task 6: Sarcasm Detection (8 pts) - LLMs only + optional few-shot extra credit

**Total Points:** 50 pts (8 reading + 40 tasks + 2 reflection)

---

## Phase 2b: introdl Package Enhancements ✅ **COMPLETE**

**Status:** Cost tracking and usability improvements complete
**Completed:** 2025-10-10
**Dependencies:** Homework 7 complete ✅

### introdl v1.6.19 - Dictionary Return for llm_list_models()
- [x] Changed `llm_list_models()` to return dictionary instead of list
- [x] Structure: `{model_name: {model_id, size, costs, json_schema, ...}}`
- [x] Enables easy lookups: `models['gemini-flash-lite']['cost_in_per_m']`
- [x] Added examples in L07_1_Getting_Started.ipynb demonstrating usage
- [x] Backward compatible - verbose table display still works

### introdl v1.6.20 - Simplified Session Spending Display
- [x] Simplified `show_session_spending()` output format
- [x] Shows "Total Spent this session" instead of cumulative
- [x] Fetches live credit from OpenRouter API
- [x] Shows "Approximate Credit remaining" with disclaimer about delay
- [x] Removed "Overall Credit Status" section

### introdl v1.6.21 - Automatic Cost Tracking Initialization
- [x] Modified `config_paths_keys()` to automatically call `init_cost_tracking()`
- [x] Cost tracking now "just works" when OpenRouter API key is present
- [x] Students no longer need to manually import/call `init_cost_tracking()`
- [x] Updated L07_1_Getting_Started.ipynb to remove manual initialization
- [x] Session spending tracking works automatically in all notebooks

### Documentation Improvements
- [x] Added comprehensive **Hugging Face Token Setup** section to L07_1
  - Step-by-step account creation and token generation
  - Instructions for adding token to `api_keys.env`
  - Verification and troubleshooting guidance
  - Positioned before OpenRouter API section

---

## Phase 3: Lessons 8-12 Updates ✅ **Lessons 7-9 COMPLETE**

**Status:** Lessons 7, 8, 9 complete. Ready for Lessons 10-12
**Dependencies:** Homework 7 complete and tested ✅

### Update Priority Order
1. ✅ **Lesson 8**: Text Classification - COMPLETE (OpenRouter + TrainerWithPretend)
2. ✅ **Lesson 9**: Transformer Details - Report assignment (no changes needed)
3. **Lesson 10**: Named Entity Recognition - NEXT PRIORITY 🔥
4. **Lesson 11**: Text Generation (uses transformers directly)
5. **Lesson 12**: Summarization

### Completed Lessons

- [x] **Lesson 8: Text Classification** ✅ COMPLETE (2025-10-14)
  - [x] Update to new llm_generate API
  - [x] Remove local LLM models, use OpenRouter exclusively
  - [x] Update Lesson_08_Helpers.py to accept model_name + provider params
  - [x] Implement TrainerWithPretend
  - [x] Updated L08_1_Text_Classification.ipynb
  - [x] Added JSON Schema Section
  - [x] Tested and cleaned notebook

- [x] **Lesson 9: Transformer Details** ✅ COMPLETE
  - Report assignment - no LLM usage, no changes needed

### Remaining Lessons (Priority Order)

- [~] **Lesson 10: Named Entity Recognition** 🔄 IN PROGRESS (2025-10-23)
  - [x] Updated L10_1 to use new llm_generate API
  - [x] Updated to TrainerWithPretend with `pretend_train=True`
  - [x] Fixed trainer.evaluate() → trainer.predict() for pretend_train compatibility
  - [x] Fixed introdl v1.6.39 NER metrics display bug (eval_overall_f1 check)
  - [x] Fixed batch size mismatch in trainer.predict() ✅
  - [x] Updated `display_ner_html` with aggregate parameter
  - [x] Fixed `extract_entities_dict` to properly merge B-/I- tags
  - [x] Added batch processing support to entity extraction
  - [x] Recovered aggregation strategy demonstration examples
  - [x] Updated CLAUDE.md with Git workflow requirements
  - [ ] **NEXT: Update Lesson_10_Helpers.py**
    - [ ] Add `extract_entities_dict()` function from notebook cell 46
    - [ ] Add `llm_ner_extractor()` function from notebook cell 65
    - [ ] Remove deprecated/unused functions
  - [ ] **NEXT: Update Homework 10**
    - [ ] Add 8 reading questions from NLPWT Chapter 4 (1 pt each)
    - [ ] Update reflection to standard format (2 pts)
    - [ ] Verify point allocation totals 50 points
  - [ ] Test all notebooks end-to-end

- [x] **Lesson 11: Text Generation** ✅ COMPLETE (2025-10-28)
  - [x] Created Lesson_11_Text_Generation_v2 with updated content
  - [x] Condensed Section 1 chronology with 2025 developments
  - [x] Updated Section 2 model table with reasoning models (o3-mini, DeepSeek-R1)
  - [x] Added Chinese/international models (Qwen 2.5, DeepSeek-V3)
  - [x] Condensed Section 4 (training pipeline) to high-level overview
  - [x] Enhanced Section 5 with API content:
    - Environment variables and API key management
    - OpenRouter multi-provider access
    - Building custom API helper functions (complete code example)
    - LangChain framework mention
  - [x] Condensed Section 8 (adapting LLMs) to brief overview
  - [x] Enhanced Section 7 with OpenRouter examples and 70B model demos
  - [x] Created L11_2_Background_Supplement.ipynb with stubs
  - [x] Updated L11_0_Overview.ipynb with new learning objectives
  - [x] Uses transformers directly (AutoModelForCausalLM, AutoTokenizer)
  - [x] Kept local model examples (3B, 8B, 70B quantized models)
  - [x] Added substantial API usage section via OpenRouter
  - **Note**: Generation typically doesn't use Trainer for fine-tuning

- [ ] **Lesson 12: Summarization**
  - [ ] Update to new llm_generate API
  - [ ] Compare specialized models vs LLMs
  - [ ] **Add TrainerWithPretend** (if using HuggingFace Trainer)
    - [ ] Update imports: `from introdl.nlp import Trainer`
    - [ ] Add `pretend_train=True` parameter
    - [ ] Add section: "Accessing Training Metrics Programmatically"
    - [ ] Add section: "Loading Pretrained Model for Inference"
  - [ ] Update Homework 12 with pretend_train pattern
  - [ ] Test end-to-end

---

## Phase 4: Homework 8-12 Updates 🔄 **Homeworks 7-8 COMPLETE**

**Status:** Homeworks 7-8 complete. Ready for Homeworks 10-12
**Dependencies:** Lessons 8-12 updated

### Homework 8: Text Classification ✅ **COMPLETE** (2025-10-12)
- [x] Added 8 reading questions (2 pts each) from NLPWT Chapter 2: Text Classification
  - Question 1: DistilBERT advantages over BERT
  - Question 2: Tokenization strategies comparison (character, word, subword)
  - Question 3: Feature extraction vs fine-tuning trade-offs
  - Question 4: Loss-based sorting for error analysis
- [x] Updated reflection section to match Homework 7 format (2 pts)
- [x] Fixed approach numbering consistency (was 1, 3, 4 → now 1, 2, 3)
  - Approach 1: TF-IDF + ML Model (7 pts)
  - Approach 2: Fine-tune DistilBERT (7 pts) - was "Approach 3"
  - Approach 3 Part 1: LLM Zero-Shot (7 pts) - was "Approach 4 Part 1"
  - Approach 3 Part 2: LLM Few-Shot (7 pts) - was "Approach 4 Part 2"
- [x] Removed duplicate reflection section
- [x] Updated point allocation in header (total: 50 pts)
- [x] Verified consistency with simplified Lesson 8 (3 approaches)

### Remaining Homework Updates
- [ ] **Homework 9**: Transformer Details
  - Report assignment - may not need reading questions
  - Minimal changes needed

- [ ] **Homework 10**: Named Entity Recognition
  - [ ] Update to new `llm_generate()` API
  - [ ] Add reading questions from NLPWT Chapter 4
  - [ ] Update reflection to standard format
  - [ ] Test with student perspective

- [x] **Homework 11**: Text Generation ✅ COMPLETE (2025-10-28)
  - [x] Created Homework_11_v2 with complete assignment
  - [x] Added 5 reading questions (10 points) from NLPWT Chapter 5:
    - Autoregressive models and conditional text generation
    - Log probabilities vs. regular probabilities
    - Greedy search vs. beam search
    - Temperature parameter in sampling
    - Top-k and nucleus sampling methods
  - [x] Updated to new `llm_generate()` API throughout
  - [x] Updated reflection to standard format (2 points)
  - [x] Created 6 technical parts (40 points total):
    - Part 1: Decoding strategies comparison (10 pts)
    - Part 2: Building API helper functions (8 pts)
    - Part 3: Model size comparison (3B vs 8B vs 70B) (8 pts)
    - Part 4: Creative text generation application (8 pts)
    - Part 5: Analysis and comparison (4 pts)
    - Part 6: Reflection (2 pts)
  - [x] Added Storage_Cleanup.ipynb utility
  - [ ] **TODO: Create complete solutions notebook**
  - [ ] **TODO: Test all code end-to-end**

- [ ] **Homework 12**: Summarization
  - [ ] Update to new `llm_generate()` API
  - [ ] Add reading questions from NLPWT Chapter 6
  - [ ] Update reflection to standard format
  - [ ] Test with student perspective

---

## Phase 4b: Custom GPT Bundle for Lessons 7-12 🔥 **HIGH PRIORITY**

**Status:** Not started
**Goal:** Package Lessons 7-12 learning materials into Custom GPT knowledge base

### Planning Documents
- [ ] Review `Developer/Notes/Custom_GPT_Bundle_Plan.md` for bundle strategy
- [ ] Check `Developer/Custom_GPT_Bundles/` directory structure

### Bundle Creation
- [ ] Identify key materials from Lessons 7-12
  - Lesson overview notebooks
  - Main instructional notebooks
  - Helper modules and utilities
  - Reading assignments references
- [ ] Export/convert notebooks to appropriate format for GPT ingestion
- [ ] Create structured knowledge base document(s)
- [ ] Test GPT bundle with typical student queries
- [ ] Deploy to course Custom GPT

### Expected Use Cases
- Student reference for NLP concepts covered in Lessons 7-12
- Homework help and conceptual clarification
- Code debugging assistance for transformer-based assignments
- Explanation of OpenRouter API and LLM usage patterns

---

## Phase 5: Package Version & Student Rollout ⏭️ **FINAL**

**Status:** Not started

### introdl Package Version Bump
- [ ] Bump version when Lesson 7 updates complete
  - Current: v1.6.8
  - Next: v1.7.0 (major API change)
- [ ] Update `Lessons/Course_Tools/introdl/src/introdl/__init__.py`
- [ ] Document changes in version history

### Student Communication
- [ ] Test Lesson 7 with fresh student account simulation
- [ ] Release updated Lesson 7 materials to all students
- [ ] Release Homework 7 with reading questions
- [ ] Monitor first student usage for issues

---

## 📋 Completed Tasks

### ✅ Infrastructure (2025-09-30 - 2025-10-08)
- [x] Module structure cleanup (v1.6.6)
- [x] JSON capability system (v1.6.7)
- [x] New llm_generate API developed (v1.6.8)
- [x] OpenRouter models curated (16 models)
- [x] Encryption scripts created
- [x] Student distribution script configured
- [x] Deployment guide documented

### ✅ Homework Utilities Updates (2025-09-30)
- [x] Updated Homework 01-12 Utilities notebooks
- [x] Removed notebook states functionality
- [x] Updated to flat import structure

### ✅ Lesson 7 Complete Update (2025-10-09)
- [x] Created L07_0_Overview.ipynb
- [x] Updated L07_1_Getting_Started.ipynb to new API
- [x] Updated L07_2_NLP_Tasks.ipynb with dual NER approach (string + JSON mode)
- [x] Fixed JSON parsing error in NER example
- [x] Removed JSON section from L07_Other_APIs.ipynb (deferred to Lesson 8)
- [x] Created LangChain instructor note for Lesson 11

### ✅ Homework 7 Complete Update (2025-10-09)
- [x] Complete rewrite with new llm_generate() API
- [x] Added 8 pts reading questions from NLPWT Chapter 1
- [x] Added standard reflection question (2 pts)
- [x] Updated all 6 tasks to use gemini-flash-lite + model comparison
- [x] Task 2 (NER) uses mode='json' for structured output
- [x] Removed all local model references
- [x] Total: 50 points (8 reading + 40 tasks + 2 reflection)

### ✅ introdl Package Enhancements (2025-10-10)
- [x] v1.6.19: Changed llm_list_models() to return dictionary for easy lookups
- [x] v1.6.20: Simplified show_session_spending() with live OpenRouter credit
- [x] v1.6.21: Automatic cost tracking initialization in config_paths_keys()
- [x] Added Hugging Face Token Setup documentation to L07_1_Getting_Started.ipynb
- [x] Updated L07_1 to remove manual init_cost_tracking import
- [x] Students now get automatic cost tracking without extra setup

---

## 🐛 Known Issues

### None Currently Blocking
All infrastructure is ready. Proceed with key deployment.

### Watch Items
- Cost tracking file corruption (backup system implemented)
- JSON schema validation failures (fallback strategies implemented)
- Rate limiting from OpenRouter (no built-in throttling)

---

## 📝 Notes

### OpenRouter Models Deployed
**FREE Tier:**
- gpt-oss-20b (free)
- llama-3.2-3b (free)
- deepseek-v3-free (free)

**Recommended Default:**
- gemini-flash-lite ($0.075/M input, $0.30/M output)
  - Fast, cheap, full JSON support
  - Good balance of quality and cost

**Commercial Options:**
- gpt-4o-mini, gpt-4o
- claude-haiku, claude-sonnet

**Open-Source Options:**
- llama-3.3-70b, mistral-nemo, qwen3-32b, deepseek-v3

### Student Credit Limit
- Budget: $5-10 per student per semester (set in OpenRouter)
- Cost tracking: `~/home_workspace/openrouter_costs.json`
- Monitor via OpenRouter dashboard

### File Locations
**OpenRouter Deployment:**
- Script: `Developer/OpenRouter/OpenRouter_CoCalc/generate_encrypted_mapping.py`
- Keys source: `Developer/OpenRouter/2025_Fall_DS776_Openrouter_API_Keys.xlsx`
- Student roster: `Developer/OpenRouter/OpenRouter_CoCalc/names.csv`
- Output: `Developer/OpenRouter/OpenRouter_CoCalc/encrypted_key_mapping.json`
- Guide: `Developer/OpenRouter/ENCRYPTED_DEPLOYMENT_GUIDE.md`

**Student Distribution:**
- Script: `Lessons/Course_Tools/distribute_openrouter_key.py` ← PUSH TO STUDENTS HERE
- URL: https://datascienceuwl.github.io/ds776-keys/encrypted_key_mapping.json

**Development:**
- Master notebook: `Developer/openrouter_json_generation_master_v2.ipynb`
- Model config: `Lessons/Course_Tools/introdl/src/introdl/openrouter_models.json`

---

## 📐 TrainerWithPretend Implementation Specification

**Purpose:** Extend HuggingFace `Trainer` with `pretend_train` mode for classroom efficiency, similar to `train_network` for PyTorch models.

### Core Behavior

**Drop-in replacement pattern:**
```python
# Students import from introdl instead of transformers
from introdl.nlp import Trainer  # Actually TrainerWithPretend

# Use exactly like HuggingFace Trainer, just add one parameter
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    pretend_train=True  # Only extra parameter
)

# Standard HuggingFace API
trainer.train()
history = trainer.get_training_history()  # Access metrics as DataFrame
```

### Loading Priority (Like train_network)

1. **Local directory first:** `{output_dir}/best_model/`
   - If complete (config.json + weights + training_history.json)
   - Load model and metrics, skip training

2. **HuggingFace Hub (optional):** `hf_repo_id` parameter
   - If local not found and `hf_repo_id` provided
   - Download model and cache locally

3. **Train from scratch:** If neither exists
   - Proceed with actual training
   - Save to `best_model/` for next time

### Key Features

**Training History Management:**
- Save: `{output_dir}/best_model/training_history.json`
- Format: DataFrame columns `['epoch', 'train_loss', 'eval_loss', 'eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']`
- Extract from `Trainer.state.log_history` after training
- Display automatically when loading with `pretend_train=True`

**User Experience:**
```
Session 1 (first run):
  - Full HuggingFace training output
  - "✓ Model saved to: best_model/"

Session 2+ (re-runs):
  - "✓ Loading pre-trained model from: best_model/"
  - "Model already trained. Loading checkpoint..."
  - Loading: [=====>] 100%
  - 📊 Training History (DataFrame displayed)
  - ✓ Best model: Epoch 2 | Accuracy: 0.8932
```

**Methods:**
- `train()` - Returns standard TrainOutput (maintains HF compatibility)
- `get_training_history()` - Returns pandas DataFrame with metrics

**Directory Structure:**
```
{output_dir}/
├── checkpoint-191/          # HF's intermediate checkpoints
├── best_model/              # Final model (our convention)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── training_history.json  ← Our addition
└── logs/                    # TensorBoard logs
```

### Implementation Location

- **File:** `Lessons/Course_Tools/introdl/src/introdl/nlp.py`
- **Class:** `TrainerWithPretend(Trainer)` extends HuggingFace Trainer
- **Export:** `Trainer = TrainerWithPretend` for drop-in usage

### Design Principles

1. **100% HuggingFace compatible** - Students learn the real API
2. **Minimal interface change** - Just one `pretend_train` parameter
3. **Matches train_network** - Same loading priority and behavior
4. **Works from first run** - `pretend_train=True` can be set once and left
5. **Graceful fallbacks** - Always proceeds if loading fails

### Dependencies

- Standard imports: `transformers`, `pandas`, `tqdm`, `pathlib`
- For HF Hub: `huggingface_hub.hf_hub_download` (optional)
- Existing introdl utilities: None (self-contained)

---

## 🔮 Future Enhancements (Low Priority)

- [ ] Add automatic cost reporting dashboard
- [ ] Create student usage monitoring
- [ ] Develop prompt engineering mini-lesson
- [ ] Add JSON schema builder tool
- [ ] Implement rate limiting/throttling
- [ ] Add streaming response support
- [ ] TrainerWithPretend: Push trained models to HF Hub automatically
- [ ] TrainerWithPretend: Support for multi-GPU training

---

## 📚 Reference Documents

- **OpenRouter Deployment**: `Developer/OpenRouter/ENCRYPTED_DEPLOYMENT_GUIDE.md`
- **Deployment Status**: `Developer/OpenRouter/OpenRouter_CoCalc/DEPLOYMENT_COMPLETE.md` (outdated)
- **Next Session Context**: `Developer/NEXT_SESSION.md`
- **Master Notebook**: `Developer/openrouter_json_generation_master_v2.ipynb`
- **Model Config**: `Lessons/Course_Tools/introdl/src/introdl/openrouter_models.json`

---

## 🖥️ Infrastructure: WSL Native Filesystem Migration ✅ **COMPLETE**

**Status:** Migration complete - repository now on WSL native filesystem
**Completed:** 2025-10-11
**Benefits:** 10x+ performance, proper file events, no sync conflicts

### Migration Accomplishments
- [x] Copied 79GB from Google Drive DrvFS to WSL native filesystem
  - Source: `/mnt/e/.../DS776/` (DrvFS mount)
  - Destination: `~/DS776_new/` (WSL ext4 filesystem)
  - Verified: 34,844 files, git repository intact, sizes match
- [x] Updated symlink structure
  - Old: `~/DS776` → Google Drive path
  - New: `~/DS776` → `~/DS776_new/` (WSL native)
- [x] Created automated backup script: `~/DS776/backup_to_gdrive.sh`
  - Syncs WSL → Google Drive with progress reporting
  - Archive mode, size verification, deletion of removed files
- [x] Updated CLAUDE.md documentation
  - Added "WSL Native Filesystem Setup" section
  - Documented backup automation workflow
  - Explained benefits and usage

### Next Steps After Migration
- [ ] Test VSCode with new WSL-native location
- [ ] Optional: Clean up original Google Drive copy (keep as backup for now)
- [ ] Run backup script periodically: `bash ~/DS776/backup_to_gdrive.sh`

---

**IMMEDIATE NEXT STEPS:**
1. ✅ OpenRouter key deployment COMPLETE
2. ✅ Lessons 7-9 COMPLETE
3. ✅ Homeworks 7-8 COMPLETE
4. ✅ introdl package v1.6.37 COMPLETE
5. ✅ WSL Native Filesystem Migration COMPLETE
6. ✅ Homework 06 Reflections Summary COMPLETE (2025-10-20)
7. ✅ Lesson 11 v2 - Text Generation COMPLETE (2025-10-28)
   - Updated with 2025 models, agentic AI, reasoning models
   - Added API usage section (OpenRouter, helper functions)
   - Enhanced with 70B model demonstrations
   - Created background supplement notebook
8. ✅ Homework 11 v2 - Text Generation COMPLETE (2025-10-28)
   - Reading questions from NLPWT Chapter 5
   - 6 technical parts covering decoding, APIs, model sizes
   - **TODO: Create solutions notebook**
   - **TODO: Test all code end-to-end**
9. 🔥 **NEXT: Lesson 10 - Named Entity Recognition**
   - Update to new llm_generate API
   - Compare specialized NER models vs LLM zero-shot
   - Add TrainerWithPretend if using HuggingFace Trainer
   - Update Homework 10 with reading questions + reflection
10. 🔥 **HIGH PRIORITY: Custom GPT Bundle for Lessons 7-12**
    - Review `Developer/Notes/Custom_GPT_Bundle_Plan.md`
    - Package lesson materials into Custom GPT knowledge base
    - Test with typical student queries
11. ⏭️ **THEN: Lesson 12 - Summarization**
    - Update to new llm_generate API
    - Compare specialized models vs LLMs
    - Update corresponding homework assignment


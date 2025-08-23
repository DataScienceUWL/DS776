# DS776 Course Update Master Plan
**Course Start: Next Week**
**Priority: Launch-ready utilities + Lessons/HW 1-3**

## Session Progress (Most Recent)
### Completed Tasks ✅
- [x] Fixed file path issues in HW5 (Flowers102 dataset)
- [x] Fixed file path issues in HW6 (nuclei dataset paths)
- [x] Configured YOLO to use appropriate directories (simplified approach)
- [x] Created storage analysis script and analyzed all 12 homeworks
- [x] Generated homework_storage_analysis.csv with detailed breakdown
- [x] Fixed path configuration for home_workspace in local development
- [x] Created HW##_Utilities.ipynb notebooks for all 12 homework folders
- [x] Moved utility functions to introdl.utils.storage_utils module
- [x] Added environment-aware storage reporting (10GB CoCalc, ~50GB compute, no limits local)
- [x] Fixed numbering consistency (Lesson_01 format, Homework_01_models pattern)
- [x] Fixed HTML export to work in Jupyter notebooks (no input())
- [x] Added zip models functionality for download/backup
- [x] Renamed all utility notebooks to Homework_##_Utilities.ipynb pattern
- [x] Enhanced storage report with model vs notebook breakdown
- [x] Added clear explanations of cached/pretrained vs user-trained models
- [x] Fixed zip naming to use Homework_##_models pattern

### Session Completed (Aug 23, 2025)
- [x] Created Course Tools notebooks (Course_Setup.ipynb, Clean_and_Free_Space.ipynb)
- [x] Implemented API key priority system (env vars > ~/api_keys.env > home_workspace/api_keys.env)
- [x] Fixed TORCH_HUB deprecation and cs_workspace data path
- [x] Added Reflection questions to Homework_02
- [x] Analyzed helper scripts and created recommendations (see Developer/Notes/helper_script_recommendations.md)

### Next Session Priority - HOMEWORK STANDARDIZATION

#### 1. HELPER SCRIPTS - Move duplicates to introdl package
**Create new modules:**
```bash
# Create directories
mkdir -p Lessons/Course_Tools/introdl/src/introdl/nlp/generation
mkdir -p Lessons/Course_Tools/introdl/src/introdl/nlp/summarization
```

**Move files:**
1. Copy `Lessons/Lesson_11_Text_Generation/helpers.py` → `introdl/src/introdl/nlp/generation.py`
2. Copy `Lessons/Lesson_12_Summarization/helpers.py` → `introdl/src/introdl/nlp/summarization.py`
3. Update `introdl/src/introdl/nlp/__init__.py` to import from these modules
4. Delete `Homework/Homework_11/helpers.py` and `Homework/Homework_12/helpers.py`
5. Update HW11 and HW12 notebooks to import from `introdl.nlp.generation` and `introdl.nlp.summarization`

#### 2. NOTEBOOK RENAMING - Standardize all homework notebooks
**Rename main notebooks to Homework_XX_Assignment.ipynb:**
```bash
cd Homework
mv Homework_01/Homework_01_Classify_Spiral_Points.ipynb Homework_01/Homework_01_Assignment.ipynb
mv Homework_02/Homework_02.ipynb Homework_02/Homework_02_Assignment.ipynb
mv Homework_03/Homework_03.ipynb Homework_03/Homework_03_Assignment.ipynb
# Continue for all 12 homeworks
```

**Handle special cases:**
- HW07: Delete `Homework_07.ipynb`, rename `Homework_07_USE_THIS_ONE.ipynb` → `Homework_07_Assignment.ipynb`
- HW09: Rename `Homework_09_Prompt_START_HERE.ipynb` → `Homework_09_Assignment.ipynb`
- HW11/HW12: Remove Solutions notebooks from Homework folders

#### 3. FOLDER STANDARDIZATION
**Rename model folders to use capital M:**
```bash
mv Homework_01/Homework_01_models Homework_01/Homework_01_Models
mv Homework_02/Homework_02_models Homework_02/Homework_02_Models
# Continue for all that have model folders
```

**Standardize script names:**
```bash
mv Homework_01/Homework_01_scripts.py Homework_01/Homework_01_Helpers.py
mv Homework_06/graphics_and_data_prep.py Homework_06/Homework_06_Helpers.py
mv Homework_08/llm_helpers.py Homework_08/Homework_08_Helpers.py
```

#### 4. UPDATE IMPORTS in notebooks after renaming
- Search and replace old filenames in notebook imports
- Update from `Homework_01_scripts` to `Homework_01_Helpers`
- Update from `from helpers import` to `from introdl.nlp.generation import` (HW11)
- Update from `from helpers import` to `from introdl.nlp.summarization import` (HW12)

#### 5. COPY MISSING HELPERS
- Check if HW10 needs helpers.py from Lesson_10 (NER utilities)
- If yes, copy and rename to Homework_10_Helpers.py

### Storage Analysis Results
- Total storage across all HW: 34.9 GB
- Largest: HW7 (9.3 GB), HW11 (7.9 GB), HW10 (6.3 GB)
- Average per homework: 2.9 GB
- Downloads folder dominates (25.2 GB / 72% of total)

## PHASE 0: Immediate Setup (Today)
### Storage Environment Setup
- [x] Create home_workspace folder in project root (using local_workspace=True)
- [x] Update .gitignore with all workspace folders
- [x] Update config_paths_keys() to use home_workspace for local development
- [x] Test local setup mirrors CoCalc student experience

## PHASE 1: Storage Analysis (Day 1-2) - COMPLETED ✅
### Run Complete Student Solutions
- [x] Obtain student solution notebooks from last semester (Ashley's solutions)
- [x] Create analyze_storage.py script with local workspace mode
- [x] Run storage analysis: `python analyze_homework_storage.py`
- [x] Document storage requirements per assignment (homework_storage_analysis.csv)
- [x] Identify storage bottlenecks and peak usage (HW7, HW10, HW11)
- [x] Determine if 10GB home_workspace is viable (NO - need cache management)
- [ ] Create storage_analysis.md with recommendations and management strategy

## PHASE 2: Launch-Critical Updates (Days 2-4) - MUST COMPLETE
### 2.1 Storage Architecture Redesign - COMPLETED ✅
- [x] **Homework-Aware Model Storage**:
  - [x] Make each HW notebook aware of its title (automatic detection)
  - [x] Create HW#_models folder inside each HW directory (synced via home_workspace)
  - [x] Update config_paths_keys() to detect and use local HW#_models path
  - [x] Test in Lesson_02_CNNs and Homework_02 - working perfectly
  - [ ] Test sync between base CoCalc and compute servers (needs CoCalc environment)

- [x] **Homework Utility Notebooks** (Homework_##_Utilities.ipynb):
  - [x] Create template utility notebook with standard cells:
    - Storage status check (red/yellow/green indicators)
    - Time-based cache cleanup (delete models older than 7 days)
    - **Option to delete corresponding Lesson_X_models folder**
    - HTML export for submission (with HW# and timestamp)
    - Emergency cleanup option (delete all cache with warnings)
  - [x] Deploy to each homework folder
  - [x] Progressive features:
    - HW01-05: Basic utilities only
    - HW06+: Add cache cleanup reminders, lesson models cleanup
    - HW11: Extra warnings about large model downloads
  - [x] Use simple time-based deletion (no complex model tracking)
  - [x] Clear feedback on space freed after operations

- [ ] **Cache Management Strategy**:
  - [ ] In base CoCalc: all folders in home_workspace (synced, 10GB limit)
  - [ ] On compute servers: use cs_workspace for cache/downloads (not synced)
  - [ ] Implement time-based cleanup (7-day default)
  - [ ] No homework-specific model preservation (too complex)
  
- [ ] **Student Submission Tools**:
  - [ ] Add HTML export to utility notebooks (not main homework)
  - [ ] Integrate existing HTML generation code
  - [ ] Include HW number and timestamp in filename
  - [ ] Test LMS compatibility
  
- [ ] **Notebook State Management**:
  - [ ] Research and design state save/load functionality
  - [ ] Add to utility notebooks (not main homework)
  - [ ] Document usage for students

### 2.2 Course Tools Notebooks & Setup
- [x] **Course_Setup.ipynb** - Initial student setup notebook:
  - [x] Install introdl package
  - [x] Create home_workspace folder structure
  - [x] Copy api_keys.env template to home_workspace
  - [x] Verify installation and show success message
  - [x] Place in Lessons/Course_Tools/

- [x] **setup_course.sh** - Bash script for batch setup:
  - [x] Install introdl package
  - [x] Create workspace directories
  - [x] Set up API keys template
  - [x] Verify installation
  
- [x] **Clean_and_Free_Space.ipynb** - Global storage management:
  - [x] Similar to homework utilities but course-wide
  - [x] Show storage across all lessons/homework
  - [x] Clean old cache files globally
  - [x] Option to delete all lesson models
  - [x] Emergency cleanup options
  
- [x] **API Key Management**:
  - [x] Create api_keys.env template in Course_Tools
  - [x] Update config_paths_keys() priority:
    1. Environment variables (highest priority)
    2. ~/api_keys.env
    3. home_workspace/api_keys.env (student editable, synced)
  - [x] Ignore blank values and "abcdefg" placeholder values
  - [x] Document for students to edit home_workspace/api_keys.env

- [ ] Update introdl package:
  - [x] **Condense config_paths_keys() output to reduce clutter**
  - [x] Implement new API key priority system
  - [x] Version bump to 1.4.1
  - [x] Remove deprecated TORCH_HUB variable
  - [x] Fix cs_workspace data path for compute servers
  - [ ] Add nlp.generation module (from L11/HW11 helpers)
  - [ ] Add nlp.summarization module (from L12/HW12 helpers)
  - [ ] Audit and remove unused functions from idlmam.py
  - [ ] Add checkpoint_management.py module
  - [ ] Implement smart checkpoint rotation in train_network()
  - [ ] Add homework detection utilities

### 2.2 Lessons 1-3 Updates
- [ ] **Lesson 1**: 
  - [ ] Add "Getting Started" section with utility notebooks
  - [ ] Update import pattern (remove installation cell)
  - [ ] Add storage management overview
- [ ] **Lesson 2**:
  - [ ] Update import pattern
  - [ ] Add checkpoint management instructions
  - [ ] Test with new storage paths
- [ ] **Lesson 3**:
  - [ ] Update import pattern
  - [ ] Verify augmentation examples work
  - [ ] Check scheduler implementations

### 2.3 Homework 1-3 Updates
- [ ] **HW01**:
  - [ ] Rename to Homework_01_Assignment.ipynb
  - [ ] Update import pattern (remove installation cell)
  - [ ] Rename Homework_01_scripts.py to Homework_01_Helpers.py
  - [ ] Update imports in notebook
  - [ ] Adjust point distribution
  - [ ] Complete solution using student reference
- [x] **HW02**:
  - [x] Added Reflection questions
  - [ ] Rename to Homework_02_Assignment.ipynb
  - [ ] Update import pattern (remove installation cell)
  - [ ] Adjust point distribution
  - [ ] Complete solution using student reference
- [ ] **HW03**:
  - [ ] Rename to Homework_03_Assignment.ipynb
  - [ ] Add Reflection questions (copy from HW01)
  - [ ] Update import pattern (remove installation cell)
  - [ ] Adjust point distribution
  - [ ] Complete solution using student reference

### 2.4 Documentation for Launch
- [ ] Create STUDENT_QUICKSTART.md with:
  - [ ] First-time setup instructions
  - [ ] Storage management basics
  - [ ] Utility notebook guide
- [ ] Update CLAUDE.md with storage architecture
- [ ] Create storage_tips.md for students

## PHASE 3: Week 1 Updates (During First Week of Course)
### Lessons 4-6 & Homework
- [ ] Update import patterns
- [ ] Complete solutions using student references
- [ ] Adjust point distributions
- [ ] Test storage impact

## PHASE 4: Week 2 Updates (Week 2 of Course)
### OpenRouter Migration (Lessons 7-12)
- [ ] **Lesson 7**: Convert to OpenRouter API
- [ ] **Lesson 8**: Convert to OpenRouter API
- [ ] **Lesson 10**: Convert to OpenRouter API
- [ ] **Lesson 11**: Keep local LLMs (educational purpose)
- [ ] **Lesson 12**: Convert to OpenRouter API
- [ ] Add OpenRouter setup instructions
- [ ] Document $15 credit and usage expectations
- [ ] Update API key management

### Corresponding Homework Updates
- [ ] HW07-HW12: Update for OpenRouter
- [ ] Complete solutions using student references
- [ ] Adjust point distributions

## PHASE 5: Week 3 Updates
### Final Assignments
- [ ] HW13: Project Draft assignment
- [ ] HW14: Final Project assignment
- [ ] Create project rubrics
- [ ] Update grading guidelines

## Ongoing During Course
### Testing & Refinement
- [ ] Monitor student storage usage
- [ ] Gather feedback on utility notebooks
- [ ] Fix issues as they arise
- [ ] Update documentation based on questions

## Storage Management Implementation Details
### checkpoint_management.py features:
```python
- get_homework_context()  # Detect current HW
- smart_checkpoint_save()  # Rotation, compression
- sync_to_home()          # Selective sync
- cleanup_old_homework()  # Archive completed work
- estimate_storage_needs() # Predict requirements
```

### Checkpoint_Manager.ipynb cells:
1. Check current storage usage
2. Sync checkpoints between servers
3. Compress old checkpoints
4. Archive completed homework
5. Clear specific homework data
6. Setup external storage (optional)

## Critical Success Metrics for Launch
✅ Students can:
1. Install course package easily
2. Understand storage limitations
3. Manage checkpoints across compute servers
4. Complete HW1-3 without storage issues
5. Find help when needed

## Post-Launch Priority Order
1. Complete remaining solutions (HW4-14)
2. OpenRouter migration (L7-12)
3. Remove unused introdl functions
4. Consistent formatting across all notebooks
5. Enhanced storage tools based on feedback

## Future Enhancements (Not Urgent - Don't Lose Track)
### Improved Pretend Training Experience
**Current State**: pretend_train flag loads pretrained models but shows minimal output

**Goal**: Students see realistic training output while using pretrained models

**Implementation Ideas**:
1. **Cached Training Logs Approach**:
   - Save actual training logs alongside pretrained models on HuggingFace
   - Replay logs with realistic timing during pretend training
   - Store as JSON/pickle: training_logs, metrics, sample_outputs, timing_info

2. **Decorator-Based Solution**:
   ```python
   @pretend_trainable("jeffbaggett/ds776-lesson3-cnn", log_file="training_logs.json")
   def train_network(model, loss_func, ...):
       # Original code unchanged
   ```
   - Clean separation of concerns
   - Reusable across different training functions
   - Works for both train_network and Transformers Trainer

3. **Hybrid Approach**:
   - Run real training for 1-2 steps to show actual output
   - Then load pretrained model
   - Shows students real metrics before fast-forwarding

4. **Universal Training Cache Decorator**:
   - Cache any training function's results
   - Replay on demand with educational output
   - Useful for both lessons and homework solutions

**Benefits**:
- Students see realistic training progression
- Understand expected metrics and timing
- Can follow lessons without GPU
- Maintains educational value while saving time/resources

**Implementation Notes**:
- Create artifact collection script for existing models
- Store logs on HuggingFace with models
- Update train_network for Lessons 1-6
- Create PretendTrainer wrapper for Lessons 7-12
- Add small delays during replay for realism

## Questions Resolved
- ✅ Course starts next week
- ✅ 14 total assignments (HW1-12 + 2 project)
- ✅ Keep local LLMs only in Lesson 11
- ✅ $15 OpenRouter credits per student
- ✅ Use student solutions as reference for completion

## Next Immediate Actions
1. Get student solution notebooks
2. Run storage analysis
3. Update Install_and_Clean.ipynb
4. Create Checkpoint_Manager.ipynb
5. Update Lessons/HW 1-3
6. Deploy and test in CoCalc
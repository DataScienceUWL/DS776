# DS776 Deep Learning - Course Repository

This repository contains all course materials for DS776 Deep Learning. These materials are identical to what you'll find in CoCalc but are provided here for reference and for students who wish to work locally.

## 📚 Course Structure

```
DS776/
├── Homework/           # Homework assignments (HW01-HW12)
├── Lessons/            # Course lessons and instructional notebooks
│   └── Course_Tools/   # Setup scripts and utilities
└── home_workspace/     # Your data, models, and downloads (created automatically)
```

## 🚀 Local Setup (Optional)

**Note:** Local installations are not officially supported. CoCalc with GPU compute servers is the recommended environment. Only attempt local setup if you have:
- A capable NVIDIA GPU (or Apple Silicon for some tasks)
- Experience with Python environment management
- Willingness to troubleshoot on your own

### Quick Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/DataScienceUWL/DS776.git
   cd DS776
   ```

2. **Set the environment variable:**
   ```bash
   export DS776_ROOT_DIR=/absolute/path/to/DS776
   ```
   Add this to your `.bashrc` or `.zshrc` to make it permanent. The path should point to the folder containing `Lessons` and `Homework`.

3. **Choose ONE of these setup methods:**

   **Option A - Run the setup script:**
   ```bash
   cd Lessons/Course_Tools
   bash setup_course.sh
   ```

   **Option B - Use the setup notebook:**
   ```bash
   jupyter notebook Lessons/Course_Tools/Course_Setup.ipynb
   # Run all cells in the notebook
   ```

   **Option C - Just install the package:**
   ```bash
   pip install Lessons/Course_Tools/introdl/
   ```

All three methods install the `introdl` package. The notebooks will create necessary folders (`home_workspace`) automatically when you run `config_paths_keys()` in your first code cell.

### Testing Your Setup

```python
from introdl.utils import config_paths_keys

# This creates folders if needed and shows your configured paths
paths = config_paths_keys()
```

If this runs without errors, you're all set!

## ⚠️ Important Notes

- **API Keys:** Some later lessons require API keys (OpenAI, HuggingFace, etc.). When needed, add these to `home_workspace/api_keys.env` (the file will be created when you first run config_paths_keys)
- **Storage:** Deep learning models are large. Ensure you have adequate disk space (50+ GB recommended)
- **GPU:** Many notebooks require GPU. Without one, you'll be limited to CPU-only operations or using pretrained models
- **Support:** Local installations are not supported. Use CoCalc for the official course experience

## 🔗 Resources

- **Course Syllabus:** Available in Canvas
- **Piazza:** For all course-related questions
- **CoCalc:** Primary development environment with GPU support

## 📝 License

These materials are provided for educational purposes as part of the DS776 course at the University of Wisconsin-La Crosse.
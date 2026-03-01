# Generate Homework Solution

## Description
Creates complete homework solutions for DS776 assignments by analyzing lesson patterns, textbook readings, and reference materials to produce professional-quality Solutions_XX.ipynb files.

## Usage
```
/generate_solution <homework_number>
```
Examples:
- `/generate_solution 01`
- `/generate_solution 3`
- `/generate_solution 12`

## Instructions
Please generate a complete homework solution following these systematic requirements:

### Pre-Analysis Phase:
1. **Validate homework exists**: Check that `Homework/Homework_{HW}/Homework_{HW}_Assignment.ipynb` exists (where HW is zero-padded, e.g., 01, 02, 12)

2. **Extract reading assignments**: 
   - Read `Lessons/Lesson_{HW}_*/L{HW}_0_Overview.ipynb` 
   - Identify assigned textbook chapters from "Readings" section
   - Note specific sections (e.g., "Chapter 1, pay attention to 1.2, 1.4, 1.5")

3. **Gather reference materials**:
   - Previous solution: `Developer/Solutions_Old/Solutions_{HW}/Solutions_{HW}.ipynb`
   - Student reference: `Developer/Ashley/HW{int(HW)}/` (e.g., Ashley/HW1/ for homework 01)
   - Lesson notebooks: `Lessons/Lesson_{HW}_*/` (all numbered notebooks)
   - Textbook chapters: `Developer/Textbooks/idlmam/` and `Developer/Textbooks/nlpwt/`

### Analysis Phase:
4. **Understand assignment structure**:
   - Read the homework assignment completely
   - Identify code problems, reading questions, and reflection questions
   - Note point distributions and requirements
   - Check for helper files (e.g., `Homework_{HW}_Helpers.py`)

5. **Extract lesson patterns**:
   - Review corresponding lesson notebooks for:
     - Import patterns and library usage
     - Code style and function signatures
     - Data handling approaches
     - Model architectures and training patterns
   - Note custom utilities from `introdl` package

6. **Review reference solutions**:
   - Examine Ashley's student approach for problem-solving patterns
   - Check previous solution structure and methodology
   - Note any differences between old and new assignment versions

### Solution Generation Phase:
7. **Create solution structure**:
   - Copy homework assignment notebook structure
   - Keep all instruction cells and problem descriptions
   - Replace `# === YOUR CODE HERE ===` sections with complete solutions
   - Replace `📝 **YOUR ANSWER HERE:**` with complete answers

8. **Code implementation requirements**:
   - Use imports and patterns established in corresponding lesson
   - Follow lesson-style model architectures and training approaches
   - Include proper documentation and comments
   - Ensure all code executes correctly with proper outputs
   - Use established helper functions from lesson or homework helpers
   - Match expected performance metrics (e.g., "achieve 95% accuracy")

9. **Reading question requirements**:
   - Answer all reading questions using assigned textbook material
   - Reference specific sections, chapters, and page numbers when possible
   - Provide comprehensive answers that demonstrate textbook understanding
   - Connect textbook concepts to practical implementation

10. **Quality assurance**:
    - Verify all code cells execute without errors
    - Check that outputs match expected results
    - Ensure reading answers are substantive and textbook-based
    - Maintain consistent formatting and style

### Output Requirements:
- **File location**: `Developer/Solutions/Solutions_{HW}.ipynb`
- **File naming**: Zero-padded homework numbers (e.g., Solutions_01.ipynb, Solutions_12.ipynb)
- **Content quality**: Professional instructor-level solutions, not student-level work
- **Completeness**: All problems solved, all questions answered
- **Code style**: Consistent with lesson patterns and course standards

### Textbook Reference Guidelines:
- **idlmam chapters**: Inside Deep Learning textbook (Lessons 1-6 typically)
- **nlpwt chapters**: Natural Language Processing with Transformers (Lessons 7-12 typically)
- Include specific chapter and section references in answers
- Explain concepts from textbook in context of homework problems

### Processing Steps:
1. **Validate input**: Ensure homework number is valid and assignment exists
2. **Read assignment**: Completely analyze the homework requirements
3. **Extract readings**: Get assigned textbook chapters from lesson overview
4. **Review references**: Examine Ashley's work and previous solutions
5. **Study lesson patterns**: Understand code style and approaches from lessons
6. **Generate solution**: Create complete Solutions_{HW}.ipynb with all problems solved
7. **Quality check**: Verify code execution and answer completeness
8. **Save output**: Place in Developer/Solutions/ directory

### Quality Checks:
- All code sections are complete and executable
- All reading questions reference assigned textbook material
- Solution follows lesson-established patterns and imports
- Point values and requirements are addressed
- File is saved with correct naming convention in correct location

Please execute this comprehensive solution generation workflow for the requested homework number.
# DS776 - Next Session Context

**Last Updated:** 2026-03-02
**Session Focus:** Course Dashboard & Platform Integrations (Canvas API, Piazza, CoCalc)

---

## What Was Accomplished This Session (2026-03-01 / 2026-03-02)

### CoCalc Remote Management (Complete)
1. **SSH access** to CoCalc project via `ssh cocalc`
2. **CoCalc API keys** configured in `~/.bashrc` (global + project-specific)
3. **Spring 2026 roster** built from course file + name resolution API (19 students)
4. **`/diagnose-student` enhanced** with 3 modes (local file, CoCalc fetch, question only)
5. **Course_Management** copied from CoCalc to `Developer/Course_Management/`
6. **LOCAL_WORKFLOW.md** created documenting SSH, API, and skill workflows
7. **All committed and pushed** — security audit verified no API keys in repo

### Student Diagnostics
- Diagnosed Ameya's HW05 (test evaluation, variable name mismatch, ResNet class bugs)
- Generated responses in `Developer/Feedback/`

---

## Next Priority: Course Dashboard & Platform Integrations

The instructor wants to build a unified course management workflow integrating:
1. **Canvas LMS** — download student submissions, upload grades/feedback
2. **Piazza** — monitor questions, draft responses, alert on new posts
3. **CoCalc** — already working (fetch notebooks, push content, manage projects)

### Vision: Course Dashboard Skills

A set of Claude Code skills that let the instructor manage the course entirely from WSL:
- `/fetch-submissions HW05` — download all student notebook submissions from Canvas
- `/grade-submission Student HW05` — grade and upload feedback to Canvas
- `/piazza-check` — show unanswered/unresolved Piazza posts
- `/piazza-respond @123` — draft and post instructor response
- `/course-status` — unified view of submissions, grades, forum activity

---

## Canvas LMS API — Research Summary

### Setup Required
1. **Generate API token**: Canvas > Account > Settings > Approved Integrations > New Access Token
2. **Store token**: Add `CANVAS_API_KEY` to `~/.bashrc`
3. **Install library**: `pip install canvasapi`
4. **Find course/assignment IDs**: From Canvas URLs (e.g., `courses/12345/assignments/67890`)

### Canvas Base URL
```
https://<institution>.instructure.com/api/v1/
```
UWL's would be something like `https://uwl.instructure.com/api/v1/` — confirm actual domain.

### Key canvasapi Operations
```python
from canvasapi import Canvas

canvas = Canvas("https://uwl.instructure.com", API_KEY)
course = canvas.get_course(COURSE_ID)

# Roster
enrollments = course.get_enrollments(type=["StudentEnrollment"])

# Assignments
assignments = course.get_assignments()

# Download submissions
assignment = course.get_assignment(ASSIGNMENT_ID)
submissions = assignment.get_submissions(include=["user"])
for sub in submissions:
    if hasattr(sub, 'attachments') and sub.attachments:
        url = sub.attachments[-1]['url']  # Direct download, no extra auth needed
        # Download with requests.get(url)

# Grade + comment
sub = assignment.get_submission(USER_ID)
sub.edit(
    submission={'posted_grade': '85'},
    comment={'text_comment': 'Good work!'}
)

# Upload feedback file as comment
sub.upload_comment("feedback.md")
```

### Key Notes
- Download URLs include verifier tokens — no extra auth for downloading files
- Rate limit: ~700 requests per 10 minutes
- Pagination handled automatically by canvasapi
- `canvasapi` v3.2.0 is the current stable version

---

## Piazza API — Research Summary

### Setup
```bash
pip install piazza-api  # v0.15.0
```

### Authentication
```python
from piazza_api import Piazza

p = Piazza()
p.user_login(email="your_email", password="your_password")
network = p.network("NETWORK_ID")  # From Piazza URL
```

### Key Operations
```python
# Get feed
feed = network.get_feed(limit=20, offset=0)

# Iterate all posts (generator)
for post in network.iter_all_posts(limit=10):
    print(post['history'][0]['subject'])

# Get single post
post = network.get_post("cid_123")

# Filter by property: 'unread', 'unresolved', 'following', 'instructors'
# (via PiazzaRPC lower-level calls)

# Post instructor answer
network.create_instructor_answer(post, answer_content, revision=1)

# Create followup
network.create_followup(post, followup_content)

# Search
results = network.search("transfer learning")
```

### Post Data Structure
- `type`: 'question', 'note', 's_answer', 'i_answer', 'followup', 'feedback'
- `children[]`: student answers, instructor answers, followups
- `history[]`: past states in reverse chronological order
- `i_answer`: instructor answer child object
- `s_answer`: student answer child object

### Caveats
- **Unofficial API** — no official Piazza API exists; could break at any time
- **Authentication**: email/password only (no OAuth/tokens)
- **Issue #68**: login failures reported — may need testing
- **Issue #74**: `create_instructor_answer` has a bug with string cid input
- **No official rate limiting docs** — add delays between requests to be safe
- **Piazza has LTI 1.3 integration with Canvas** but that's for SSO, not programmatic access

### Alternative: Piazza Scraping
- Selenium/Playwright-based scrapers exist on GitHub
- Much slower than API approach
- Only needed if `piazza-api` breaks

---

## Implementation Plan (For Next Session)

### Phase 1: Canvas Integration (Highest Priority)
1. **Get Canvas API token** from UWL Canvas settings
2. **Find DS776 course ID and assignment IDs**
3. **Add `CANVAS_API_KEY` to `~/.bashrc`**
4. **Install canvasapi**: `pip install canvasapi`
5. **Build `/fetch-submissions` skill**:
   - Takes HW number, downloads all student .ipynb submissions
   - Saves to `Developer/Submissions/Homework_XX/`
   - Maps Canvas user IDs to student names (cross-ref with roster)
6. **Build `/grade-submission` skill** (or extend `/diagnose-student`):
   - After diagnosing, optionally upload grade + feedback to Canvas

### Phase 2: Piazza Integration
1. **Install piazza-api**: `pip install piazza-api`
2. **Store Piazza credentials** in `~/.bashrc` or secure location
3. **Find network ID** from Piazza URL
4. **Build `/piazza-check` skill**:
   - Fetch unresolved/unanswered posts
   - Display summary with post numbers, subjects, timestamps
5. **Build `/piazza-respond` skill**:
   - Read a specific post
   - Draft response using LLM + course materials
   - Post as instructor answer (with confirmation)

### Phase 3: Unified Dashboard
- Combine Canvas + Piazza + CoCalc into a `/course-status` view
- Show: ungraded submissions, unanswered forum posts, student project status

---

## Key Files for This Work

### Existing Infrastructure
| File | Description |
|------|-------------|
| `~/.bashrc` | API keys (COCALC_API_KEY, COCALC_PROJECT_API_KEY — add CANVAS_API_KEY, PIAZZA_EMAIL/PASSWORD) |
| `Developer/OpenRouter/OpenRouter_CoCalc/names.csv` | Student roster (name, email, project ID) |
| `Developer/Course_Management/LOCAL_WORKFLOW.md` | CoCalc workflow documentation |
| `.claude/skills/diagnose-student/SKILL.md` | Existing diagnostic skill (3 modes) |

### To Create
| File | Description |
|------|-------------|
| `.claude/skills/fetch-submissions/SKILL.md` | Canvas submission download skill |
| `.claude/skills/piazza-check/SKILL.md` | Piazza monitoring skill |
| `.claude/skills/piazza-respond/SKILL.md` | Piazza response drafting skill |
| `Developer/Course_Management/CANVAS_INTEGRATION.md` | Canvas API setup and workflow docs |

---

## Important Reminders

1. **Commit before editing** — Always commit current state before changes
2. **Push frequently** — Don't accumulate unpushed commits
3. **No API keys in repo** — All keys in `~/.bashrc`, referenced as `$ENV_VAR`
4. **Developer/ is gitignored** — Submissions, feedback, and management files won't be committed
5. **Canvas token security** — Treat like a password, never hardcode
6. **Piazza is unofficial** — Test authentication first, have fallback plan

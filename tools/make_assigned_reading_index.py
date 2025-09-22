#!/usr/bin/env python3
"""
Extract Assigned Reading sections from lesson overviews.

Creates Lesson_XX/Readings/assigned_reading.md for each lesson by extracting
the "Assigned Reading" section from Overview.ipynb or Overview.md files.
"""

import json
import re
import pathlib
from typing import Optional

ROOT = pathlib.Path(".")
LESSON_GLOB = "Lesson_*"

def read_markdown_from_ipynb(nb_path: pathlib.Path) -> str:
    """Extract markdown content from Jupyter notebook."""
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        parts = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                parts.append("".join(cell.get("source", [])))
        return "\n\n".join(parts)
    except Exception as e:
        print(f"Error reading {nb_path}: {e}")
        return ""

def read_markdown_from_md(md_path: pathlib.Path) -> str:
    """Read markdown content from .md file."""
    try:
        return md_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {md_path}: {e}")
        return ""

def extract_assigned_reading(md: str) -> Optional[str]:
    """Extract 'Assigned Reading' section from markdown content."""
    # Split on the "Assigned Reading" heading (case insensitive)
    parts = re.split(r"^#{1,6}\s*Assigned Reading.*$", md, flags=re.I | re.M)

    if len(parts) < 2:
        return None

    # Take content after the "Assigned Reading" heading
    tail = parts[1]

    # Cut at the next heading of any level
    tail = re.split(r"^#{1,6}\s", tail, maxsplit=1, flags=re.M)[0]

    # Clean up whitespace
    tail = tail.strip()

    return tail if tail else None

def lesson_number_from_path(p: pathlib.Path) -> str:
    """Extract lesson number from path and zero-pad to 2 digits."""
    # Handle Lesson_01, Lesson_1, Lesson 01, etc.
    patterns = [
        r"Lesson[_\-\s]?(\d+)",  # Lesson_01, Lesson-01, Lesson 01, Lesson01
        r"L(\d+)",               # L01, L1
    ]

    for pattern in patterns:
        m = re.search(pattern, p.name, re.I)
        if m:
            return m.group(1).zfill(2)

    return "XX"  # Fallback

def process_lesson(lesson_dir: pathlib.Path) -> bool:
    """Process a single lesson directory."""
    print(f"📚 Processing {lesson_dir.name}...")

    # Look for overview files (prefer .ipynb over .md)
    overview_files = [
        lesson_dir / "Overview.ipynb",
        lesson_dir / f"{lesson_dir.name}_Overview.ipynb",  # Alternative naming
        lesson_dir / "L01_0_Overview.ipynb",  # Course-specific naming
        lesson_dir / "Overview.md",
        lesson_dir / f"{lesson_dir.name}_Overview.md",
    ]

    # Find first existing overview file
    overview_file = None
    for candidate in overview_files:
        if candidate.exists():
            overview_file = candidate
            break

    # Extract lesson number
    lesson_no = lesson_number_from_path(lesson_dir)

    # Create Readings directory
    readings_dir = lesson_dir / "Readings"
    readings_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    output_file = readings_dir / "assigned_reading.md"

    # Process content
    if overview_file is None:
        print(f"   ⚠️ No overview file found")
        content = "_No Overview file found in this lesson._"
    else:
        print(f"   📄 Found overview: {overview_file.name}")

        # Read markdown content based on file type
        if overview_file.suffix == ".ipynb":
            md_content = read_markdown_from_ipynb(overview_file)
        else:
            md_content = read_markdown_from_md(overview_file)

        # Extract assigned reading section
        reading_content = extract_assigned_reading(md_content)

        if reading_content:
            content = reading_content
            print(f"   ✅ Extracted assigned reading section")
        else:
            content = "_No 'Assigned Reading' section detected in Overview._"
            print(f"   ⚠️ No 'Assigned Reading' section found")

    # Write output file
    header = f"# Assigned Reading – Lesson {lesson_no}\n\n"
    full_content = header + content + "\n"

    output_file.write_text(full_content, encoding="utf-8")
    print(f"   💾 Created: {output_file}")

    return True

def main():
    """Main extraction process."""
    print("🔍 Extracting assigned readings from all lessons...")

    # Find all lesson directories
    lesson_dirs = sorted([d for d in ROOT.glob(LESSON_GLOB) if d.is_dir()])

    if not lesson_dirs:
        print("❌ No lesson directories found matching pattern 'Lesson_*'")
        return False

    print(f"📁 Found {len(lesson_dirs)} lesson directories")

    # Process each lesson
    success_count = 0
    for lesson_dir in lesson_dirs:
        try:
            if process_lesson(lesson_dir):
                success_count += 1
        except Exception as e:
            print(f"❌ Error processing {lesson_dir}: {e}")

    print(f"\n✅ Completed: {success_count}/{len(lesson_dirs)} lessons processed")
    print("📋 All assigned_reading.md files created/updated")

    return success_count == len(lesson_dirs)

if __name__ == "__main__":
    main()
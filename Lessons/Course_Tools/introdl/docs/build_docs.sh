#!/bin/bash
"""
Build introdl Documentation
Generates both HTML and Markdown formats for different use cases.

Usage:
    bash build_docs.sh [html|markdown|both|clean]
"""

set -e  # Exit on any error

DOCS_DIR="/mnt/e/GDrive_baggett.jeff/Teaching/Classes_current/2025-2026_Fall_DS776/DS776/Lessons/Course_Tools/introdl/docs"
echo "📚 Building introdl documentation..."
echo "📁 Documentation directory: $DOCS_DIR"

cd "$DOCS_DIR"

# Function to check if Sphinx is installed
check_sphinx() {
    if ! command -v sphinx-build &> /dev/null; then
        echo "❌ Sphinx not found. Installing..."
        pip install sphinx sphinx-rtd-theme myst-parser
    else
        echo "✅ Sphinx found"
    fi
}

# Function to build HTML
build_html() {
    echo "🌐 Building HTML documentation..."
    sphinx-build -b html source build/html
    echo "✅ HTML docs built: $DOCS_DIR/build/html/index.html"
}

# Function to build Markdown
build_markdown() {
    echo "📝 Building Markdown documentation..."
    sphinx-build -b markdown source build/markdown
    echo "✅ Markdown docs built: $DOCS_DIR/build/markdown/"
}

# Function to clean build directory
clean_build() {
    echo "🧹 Cleaning build directory..."
    rm -rf build/html/* build/markdown/*
    echo "✅ Build directory cleaned"
}

# Main script logic
check_sphinx

case "${1:-both}" in
    html)
        build_html
        ;;
    markdown)
        build_markdown
        ;;
    both)
        build_html
        build_markdown
        echo "✅ Both formats completed"
        ;;
    clean)
        clean_build
        ;;
    rebuild)
        clean_build
        build_html
        build_markdown
        echo "✅ Complete rebuild finished"
        ;;
    *)
        echo "Usage: $0 [html|markdown|both|clean|rebuild]"
        echo "  html     - Build HTML documentation only"
        echo "  markdown - Build Markdown documentation only"
        echo "  both     - Build both formats (default)"
        echo "  clean    - Clean build directory"
        echo "  rebuild  - Clean and build both formats"
        exit 1
        ;;
esac

echo "🎉 Documentation build complete!"
echo "📂 View HTML docs: open $DOCS_DIR/build/html/index.html"
echo "📂 Markdown files: $DOCS_DIR/build/markdown/"

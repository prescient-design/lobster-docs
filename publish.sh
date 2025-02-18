#!/bin/bash

source .venv/bin/activate

# Step 1: Build the JupyterBook documentation
jupyter-book build docs

# Step 2: Publish the built documentation to GitHub Pages
ghp-import -n -p -f docs/_build/html

echo "Documentation has been successfully published."

#!/bin/bash
set -e

# --- Script to initialize a git repository and push to GitHub ---

echo "--- GitHub Push Assistant for RustyFlow ---"
echo

# 1. Check for git
if ! command -v git &> /dev/null; then
    echo "Error: 'git' command not found. Please install Git to continue."
    exit 1
fi

# 2. Check if it's already a git repo
if [ ! -d ".git" ]; then
    echo "This directory is not a Git repository. Initializing a new one..."
    git init
    echo "New Git repository initialized."
else
    echo "Existing Git repository found."
fi

# 3. Get the remote URL from the user
read -p "Please paste your full GitHub repository URL (e.g., https://github.com/username/repo.git): " REMOTE_URL

if [ -z "$REMOTE_URL" ]; then
    echo "Error: No URL provided. Aborting."
    exit 1
fi

# 4. Check for existing 'origin' remote
if git remote | grep -q "origin"; then
    echo
    echo "Warning: A remote named 'origin' already exists."
    read -p "Do you want to overwrite it with the new URL? (y/n): " OVERWRITE
    if [[ "$OVERWRITE" == "y" || "$OVERWRITE" == "Y" ]]; then
        echo "Updating 'origin' remote..."
        git remote set-url origin "$REMOTE_URL"
    else
        echo "Aborting. No changes made to the remote."
        exit 0
    fi
else
    echo "Adding remote 'origin'..."
    git remote add origin "$REMOTE_URL"
fi

echo
echo "--- Preparing to push ---"

# 5. Add all files to staging
echo "Step 1/4: Staging all files..."
git add .
echo "All files staged."

# 6. Create an initial commit if no commits exist
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "Step 2/4: Creating initial commit..."
    git commit -m "Initial commit of RustyFlow project"
    echo "Initial commit created."
else
    echo "Step 2/4: Commits already exist. Creating a new one for any new changes."
    # Check if there are any changes to commit
    if git diff-index --quiet HEAD --; then
        echo "No new changes to commit. Skipping commit."
    else
        git commit -m "Sync: Update project files"
        echo "New commit created."
    fi
fi

# 7. Ensure the main branch is named 'main'
echo "Step 3/4: Ensuring branch is named 'main'..."
git branch -M main
echo "Branch set to 'main'."

# 8. Push to GitHub
echo "Step 4/4: Pushing to GitHub..."
git push -u origin main

echo
echo "--- Success! ---"
echo "Your RustyFlow project has been pushed to your GitHub repository."
echo "You can view it at: ${REMOTE_URL%.git}"
echo "------------------"

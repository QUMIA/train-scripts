#!/bin/bash

GITHUB_USERNAME="QUMIA"
GITHUB_REPOSITORY="train-scripts"
OUTPUT_DIR="/projects/0/einf6214/output/"



### Label ###

# Check if a label is provided
if [ "$#" -ne 1 ]; then
    echo "Error: A label argument is required."
    exit 1
fi

# The label is the first argument
LABEL=$1

# Check if the label contains invalid characters
if echo "$LABEL" | grep -q "[/:*?\"<>|\]"; then
    echo "Error: The label contains invalid characters for a file name."
    exit 1
fi

# Check for untracked files (we don't want to automatically add these, but they may be required)
untracked_files=$(git ls-files --others --exclude-standard)
if [ -n "$untracked_files" ]; then
    echo "There are untracked files:"
    echo "$untracked_files"
    exit 1
fi



### git commit ###

# Add all changes
git add .

# Create a new git commit with the label
git commit -m "[SESSION] $LABEL" --allow-empty



### output directory ###

# Create an output directory with the current date and time
OUTPUT_DIR="${OUTPUT_DIR}/$(date +%Y-%m-%d_%H-%M-%S)_${LABEL}"
mkdir "$OUTPUT_DIR"

# Copy the contents of the current directory to the output directory
cp -r ./* "$OUTPUT_DIR"

# Get the SHA of the latest commit
COMMIT_SHA=$(git rev-parse HEAD)
# Construct the GitHub commit link
GITHUB_COMMIT_URL="https://github.com/${GITHUB_USERNAME}/${GITHUB_REPOSITORY}/commit/${COMMIT_SHA}"


### Write details to .env ###
cat <<EOF > "$OUTPUT_DIR/.env"
SESSION_LABEL=$LABEL
GIT_SHA=$COMMIT_SHA
GITHUB_COMMIT_URL=$GITHUB_COMMIT_URL
EOF


### Queue the job ###

# Queue the train job inside the output directory (we want to have the log there also)
cd $OUTPUT_DIR
sbatch train-job.sh

echo Done

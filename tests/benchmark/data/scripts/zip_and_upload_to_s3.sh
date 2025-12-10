#!/bin/bash

################################################################################
# Zip and Upload to S3
#
# This script creates a zip archive from a folder and uploads it to S3.
# The zip file will contain only the folder itself (not the full path).
#
# Usage:
#   ./zip_and_upload_to_s3.sh <folder_path> [zip_name]
#
# Arguments:
#   folder_path  - Path to the folder to zip and upload (absolute or relative)
#   zip_name     - Optional: Name for the zip file (without .zip extension)
#                  If not provided, uses the folder name
#
# Examples:
#   # Zip and upload with default name (folder name)
#   ./zip_and_upload_to_s3.sh ../deep-1M-cosine-dim96-M32-efc200-disk-vectors
#
#   # Zip and upload with custom name
#   ./zip_and_upload_to_s3.sh ../deep-1M-cosine-dim96-M32-efc200-disk-vectors my_index
#
#   # Using absolute path
#   ./zip_and_upload_to_s3.sh /home/ubuntu/VectorSimilarity/tests/benchmark/data/my_index
#
# Output:
#   - Creates a zip file in the parent directory of the folder
#   - Uploads the zip to S3 bucket: s3://dev.cto.redis/VectorSimilarity/deep1b/
#   - Displays the S3 URL and download instructions
#
# Requirements:
#   - zip command must be installed
#   - AWS CLI must be installed and configured with appropriate credentials
#   - S3 bucket must be accessible with write permissions
#
# Environment Variables:
#   S3_BUCKET     - S3 bucket path (default: s3://dev.cto.redis/VectorSimilarity/deep1b)
#   S3_BASE_URL   - Base URL for downloads (default: https://dev.cto.redis.s3.amazonaws.com/VectorSimilarity/deep1b)
#
# Example with custom S3 bucket:
#   S3_BUCKET="s3://my-bucket/my-path" ./zip_and_upload_to_s3.sh my_folder
#
################################################################################

# Exit immediately if a command exits with a non-zero status
set -e

# Display usage information if no arguments provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [zip_name]"
    echo ""
    echo "Arguments:"
    echo "  folder_path  - Path to the folder to zip and upload (absolute or relative)"
    echo "  zip_name     - Optional: Name for the zip file (without .zip extension)"
    echo "                 If not provided, uses the folder name"
    echo ""
    echo "Examples:"
    echo "  # Zip and upload with default name"
    echo "  $0 ../deep-1M-cosine-dim96-M32-efc200-disk-vectors"
    echo ""
    echo "  # Zip and upload with custom name"
    echo "  $0 ../deep-1M-cosine-dim96-M32-efc200-disk-vectors my_index"
    echo ""
    echo "  # Using absolute path"
    echo "  $0 /home/ubuntu/VectorSimilarity/tests/benchmark/data/my_index"
    exit 1
fi

# Parse command line arguments
FOLDER_PATH="$1"

# S3 bucket configuration
# Can be overridden by setting the S3_BUCKET environment variable
# Example: S3_BUCKET="s3://my-bucket/my-path" ./zip_and_upload_to_s3.sh <folder>
S3_BUCKET="${S3_BUCKET:-s3://dev.cto.redis/VectorSimilarity/deep1b}"

# Construct the HTTPS URL from the S3 bucket path
# This can be overridden if you need a custom URL format
if [ -z "$S3_BASE_URL" ]; then
    # Extract bucket name and path from s3://bucket-name/path format
    S3_BUCKET_NAME=$(echo "$S3_BUCKET" | sed 's|s3://||' | cut -d'/' -f1)
    S3_BUCKET_PATH=$(echo "$S3_BUCKET" | sed 's|s3://||' | cut -d'/' -f2-)
    S3_BASE_URL="https://${S3_BUCKET_NAME}.s3.amazonaws.com/${S3_BUCKET_PATH}"
fi

# Validate that the folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo "Error: Folder does not exist: $FOLDER_PATH"
    exit 1
fi

# Determine the zip file name
# If a custom name is provided as second argument, use it
# Otherwise, use the folder's base name
FOLDER_NAME=$(basename "$FOLDER_PATH")
if [ $# -ge 2 ]; then
    ZIP_NAME="$2"
else
    ZIP_NAME="$FOLDER_NAME"
fi

ZIP_FILE="${ZIP_NAME}.zip"

# Display configuration summary
echo "=== Zip and Upload to S3 ==="
echo "Folder:      $FOLDER_PATH"
echo "Zip file:    $ZIP_FILE"
echo "S3 bucket:   $S3_BUCKET"
echo "============================"
echo ""

################################################################################
# Create Zip Archive
################################################################################

echo "Creating zip file..."

# Get the absolute path of the folder to ensure consistent behavior
FOLDER_ABS=$(realpath "$FOLDER_PATH")
FOLDER_PARENT=$(dirname "$FOLDER_ABS")
FOLDER_BASENAME=$(basename "$FOLDER_ABS")

# Change to the parent directory before zipping
# This ensures the zip contains only the folder name, not the full path
# For example, if folder is /home/user/data/my_index, the zip will contain:
#   my_index/
#     ├── file1
#     └── file2
# Instead of:
#   home/user/data/my_index/
#     ├── file1
#     └── file2
cd "$FOLDER_PARENT"
zip -r "$ZIP_FILE" "$FOLDER_BASENAME"
cd - > /dev/null  # Return to previous directory silently

# Verify that the zip file was created successfully
# The zip file is created in the parent directory of the folder
ZIP_FILE_PATH="$FOLDER_PARENT/$ZIP_FILE"
if [ ! -f "$ZIP_FILE_PATH" ]; then
    echo "Error: Failed to create zip file"
    exit 1
fi

# Display zip file information
ZIP_SIZE=$(du -h "$ZIP_FILE_PATH" | cut -f1)
echo "Zip file created: $ZIP_FILE ($ZIP_SIZE)"
echo "Location: $ZIP_FILE_PATH"
echo ""

################################################################################
# Upload to S3
################################################################################

echo "Uploading to S3..."
echo "Source: $ZIP_FILE_PATH"
echo "Destination: $S3_BUCKET/$ZIP_FILE"
echo ""

# Upload the zip file to S3 using AWS CLI
# Note: Ensure AWS CLI is configured with appropriate credentials
# If upload fails, the script will exit automatically due to 'set -e'
aws s3 cp "$ZIP_FILE_PATH" "$S3_BUCKET/$ZIP_FILE"

# If we reach here, upload was successful
echo ""
echo "=== Upload Complete ==="
echo "S3 URL: $S3_BASE_URL/$ZIP_FILE"
echo ""
echo "To download this file:"
echo "  # Using AWS CLI:"
echo "  aws s3 cp $S3_BUCKET/$ZIP_FILE ."
echo ""
echo "  # Using wget (if publicly accessible):"
echo "  wget $S3_BASE_URL/$ZIP_FILE"
echo ""
echo "To extract after download:"
echo "  unzip $ZIP_FILE"

#!/bin/bash

# Function to update shell profile with necessary paths
update_profile() {
    local profile_file=$1
    shift
    local paths=("$@")

    echo "Updating $profile_file with PATH additions"

    # Check if the profile exists
    if [[ ! -f $profile_file ]]; then
        touch "$profile_file"
    fi

    # Add each path to the profile if not already present
    for path in "${paths[@]}"; do
        echo "Processing path: $path"
        if ! grep -q "export PATH=\"$path:\$PATH\"" "$profile_file"; then
            echo "Adding path: $path to $profile_file"
            echo "export PATH=\"$path:\$PATH\"" >> "$profile_file" || { echo "Error: Failed to write to $profile_file"; exit 1; }
        else
            echo "Path $path is already present in $profile_file"
        fi
    done

    echo "Profile update completed successfully."
}

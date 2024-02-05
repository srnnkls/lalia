#!/bin/bash

# Check if an environment name was provided as a command-line argument
if [ $# -eq 0 ]; then
    ENV_NAME="default"
else
    ENV_NAME=$1
fi

# Remove the existing environment
echo "Removing the existing Hatch environment: $ENV_NAME"
hatch env remove $ENV_NAME

# Check if the environment removal was successful
if [ $? -eq 0 ]; then
    echo "Successfully removed the environment: $ENV_NAME"
else
    echo "Failed to remove the environment: $ENV_NAME. Exiting script."
    exit 1
fi

# Create the environment. Hatch automatically uses pyproject.toml in the current directory.
echo "Creating a new Hatch environment: $ENV_NAME"
hatch env create $ENV_NAME

# Check if the environment creation was successful
if [ $? -eq 0 ]; then
    echo "Successfully created the environment: $ENV_NAME"
else
    echo "Failed to create the environment: $ENV_NAME."
    exit 1
fi

echo "Environment $ENV_NAME has been recreated successfully."


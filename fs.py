import os
import chardet
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# The directory path to search in
dir_path = "../Rocket.Chat"

# Dictionary to store file content
file_contents = {}

# List of directories to ignore
ignore = [
    "/node_modules",
    "/.pnp",
    ".pnp.js",
    "/coverage",
    "/.next/",
    "/out/",
    "/build",
    ".DS_Store",
    "*.pem",
    "npm-debug.log*",
    "yarn-debug.log*",
    "yarn-error.log*",
    ".env*.local",
    ".vercel",
    "*.tsbuildinfo",
    ".env"
    "next-env.d.ts",
    "/.git",
    "/.husky",
    "/.vscode",
    "/.github",
    "public",
    "assets",
    ".yarn",
    "/_templates",
    ".houston",
    ".cache",
    ".json",
    "/apps/meteor/.meteor",
    "/apps/meteor/imports",
    "/apps/meteor/tests",
    "/.changeset",
    ".storybook",
]

# List to store files that encountered errors
error_files = []

# Function to check if a file is binary


def is_binary(file_path):
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(1024)
        result = chardet.detect(raw_data)
        return result['encoding'] is None
    except FileNotFoundError:
        return False


# Walk through the directory
for root, dirnames, filenames in os.walk(dir_path):
    # Check if the current directory should be ignored
    if any(ignore_dir in root for ignore_dir in ignore):
        continue

    # Filter out files from the ignored directories
    filenames = [filename for filename in filenames if not any(
        ignore_dir in os.path.join(root, filename) for ignore_dir in ignore)]

    # Read and store the content of each text-based file
    for filename in filenames:
        file_path = os.path.join(root, filename)
        try:
            if not is_binary(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_contents[file_path] = file.read()
        except Exception as e:
            error_files.append(file_path)

# Print the matching filenames and their content
for file_path, content in file_contents.items():
    print(f"File Loaded: {file_path}")
#     print(f"Content:\n{content}\n")
if len(error_files) == 0:
    print("No errors encountered")
else:
    print("Error files:")
    for file_path in error_files:
        print(file_path)
import os
import chardet
import logging
import json

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
    ".env",
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
    "/apps/meteor/.scripts",
    "/apps/meteor/tests",
    "/apps/meteor/app/irc/server",
    "apps/meteor/app/authentication",
    "/apps/meteor/app/slashcommands-hide",
    "/apps/meteor/app/slashcommands-status",
    "/apps/meteor/app/slashcommands-open",
    "/apps/meteor/private",
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

# Function to check if a directory should be ignored
def should_ignore_directory(dir_path):
    return any(ignore_dir in dir_path for ignore_dir in ignore) or "i18n" in dir_path

# List of keywords to ignore
ignored_keywords = ["require", "strict", "import", "export"]

# Add characters or patterns to ignore
characters_to_ignore = ["#", "*", "!", "$", "+", "-"]

# Function to extract code from content while ignoring specific characters or patterns
def extract_code_from_content(content):
    code_blocks = {}
    is_inside_code_block = False
    current_code_block = []
    current_key = None

    for line in content.split('\n'):
        # Check if the line contains any of the ignored keywords or characters
        if any(keyword in line for keyword in ignored_keywords) or any(char in line for char in characters_to_ignore):
            is_inside_code_block = False
            current_code_block = []
            current_key = None
        elif "const" in line or "function" in line:
            is_inside_code_block = True
            current_code_block.append(line)
            # Extract the "const" or "function" name as the key
            parts = line.split(' ')
            if len(parts) > 1:
                key_candidate = parts[1].split('(')[0].split('=')[0].strip()
                # Check if the key contains invalid characters
                if all(char.isalnum() or char == "_" for char in key_candidate):
                    current_key = key_candidate

        if is_inside_code_block:
            current_code_block.append(line)

            if "{" in line:
                brace_count = line.count("{")
                if "}" in line:
                    brace_count -= line.count("}")
                if brace_count == 0:
                    # Found the closing brace, end of code block
                    is_inside_code_block = False
                    if current_key:
                        code_blocks[current_key] = "\n".join(current_code_block)
                    current_code_block = []
                    current_key = None

    return code_blocks

# Function to calculate the depth of a directory
def calculate_depth(directory_path):
    return directory_path.count(os.path.sep)

# Function to calculate the depth rank based on a formula
def calculate_depth_rank(directory_path):
    # You can define your formula here based on the directory path
    # For example, a simple formula could be the depth itself:
    return calculate_depth(directory_path)

# Initialize the rank counter
rank = 1

# Directory depth information for code blocks
depth_info = {}

# Walk through the directory
for root, dirnames, filenames in os.walk(dir_path):
    if should_ignore_directory(root):
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

                    # Calculate the depth of the directory containing this file
                    directory_depth = calculate_depth(root)

                    # Calculate the depth rank based on the formula
                    depth_rank = calculate_depth_rank(root)

                    # Add the depth information for this file
                    depth_info[file_path] = {"depth": directory_depth, "rank": depth_rank}

        except Exception as e:
            error_files.append(file_path)

# Define the path for the output JSON file
output_json_path = "all_code_blocks.json"

# Process file contents
all_code_blocks = []
total_keys = 0

for file_path, content in file_contents.items():
    code_blocks = extract_code_from_content(content)

    # Ignore code blocks with keys that contain specific characters
    code_blocks = {key: value for key, value in code_blocks.items() if all(char not in key for char in characters_to_ignore)}
    
    # Remove empty keys (code blocks with empty values) and keys with brackets
    code_blocks = {key: value for key, value in code_blocks.items() if value.strip() and "[" not in key and "]" not in key}

    if code_blocks:
        depth_info_for_file = depth_info.get(file_path, {"depth": 0, "rank": 0})

        # Append the code blocks to the list
        all_code_blocks.append({
            "file": os.path.basename(file_path),
            "path": file_path,
            "code": code_blocks,
            "depth_rank": depth_info_for_file["rank"]
        })

        # Update the total keys
        total_keys += len(code_blocks)

# Write the collected code blocks to the output JSON file
with open(output_json_path, 'w', encoding='utf-8') as output_json_file:
    json.dump(all_code_blocks, output_json_file, indent=4)

print("All code blocks are stored in the output JSON file:", output_json_path)
print("Total number of keys:", total_keys)

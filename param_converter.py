import re
import sys
from typing import Dict, Optional

def convert_params(input_string: str) -> Dict[str, str]:
    """
    Converts command-line arguments from a list/string format into a
    model-preset dictionary format.

    Args:
        input_string: The raw string containing command-line arguments.

    Returns:
        A dictionary where keys are the preset names and values are their settings.
    """
    params = {}
    
    # Clean up the input string by replacing common separators and splitting
    # We normalize spaces and split by line breaks or consecutive spaces
    # Tokens capture either: 
    # 1. Full arguments starting with -- or - (--[\w-]+ or -[\w-]+)
    # 2. Any other non-whitespace string (\S+)
    tokens = re.findall(r'(--[\w-]+|\-[\w-]+|\S+)', input_string)
    
    current_key = None
    
    # Process tokens sequentially
    for token in tokens:
        # 1. Handle Flags/Keys (starting with -- or -)
        if token.startswith('--') or token.startswith('-'):
            # Strip leading dashes to get the clean key (e.g., --temp -> temp)
            clean_key = token.lstrip('-').replace('-', '_') 
            
            if len(token) > 1 and token[1].isalpha(): 
                current_key = clean_key
                # Initialize value storage for the next parameter
                params[current_key] = "" 
        
        # 2. Handle Values/Flags (following a key)
        elif current_key and token:
            
            # Check if the token itself represents a boolean flag (e.g., --no-mmap)
            if token.startswith('--') or token.startswith('-'):
                # If the token is a flag that implies TRUE (like no-mmap)
                if token.replace('-', '_') == current_key:
                     params[current_key] = "true"
                else:
                    # If it's an unrecognized flag token, treat it as a string value
                    params[current_key] = token
            
            # Handle numerical or simple string values
            else:
                # If the current key is set, assign the token as the value
                params[current_key] = token

    # Post-processing to ensure clean formatting and handle specific flag logic
    final_params = {}
    for key, value in params.items():
        # Replace internal underscores back to hyphens for final output keys
        clean_key = key.replace('_', '-')
        
        # Special handling for keys that were detected as flags without explicit values
        # and were stored as empty strings initially.
        if clean_key in ['no-mmap'] and value == "":
             final_params[clean_key] = "true"
        elif value:
            final_params[clean_key] = str(value)
        else:
            # If the key was detected but had no value (and wasn't handled by the boolean flag logic),
            # we might skip it or leave it as empty depending on desired behavior.
            # For this use case, we keep the captured value.
            final_params[clean_key] = value

    return final_params

def format_output(model_name: str, params: Dict[str, str]) -> str:
    """Formats the dictionary output into the required multi-line preset format."""
    output = [model_name]
    
    # Sort keys for consistent output, which is good practice for configuration files
    sorted_params = sorted(params.items())
    
    for key, value in sorted_params:
        output.append(f"{key} = {value}")
        
    return "\n".join(output)


if __name__ == "__main__":
    # --- EXAMPLE USAGE ---
    
    # 1. Define the input string matching the user's example
    input_args = """
      --temp 0.7 \
      -n 32768 \
      --no-mmap
    """
    
    # 2. Define the model name header
    model_name = "[user defined model display name]"

    # 3. Process the parameters
    try:
        parsed_params = convert_params(input_args)
        
        # 4. Generate the final formatted output
        formatted_preset = format_output(model_name, parsed_params)
        
        print("--- Generated Model Preset ---")
        print(formatted_preset)
        print("------------------------------")

    except Exception as e:
        print(f"An error occurred during conversion: {e}", file=sys.stderr)
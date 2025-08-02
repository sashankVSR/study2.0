import os
from huggingface_hub import InferenceClient
import streamlit as st

# Load API key from .streamlit/secrets.toml
HUGGINGFACE_API_KEY = st.secrets.get("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")

# Use a lightweight open-source model
model_name = "bigcode/starcoder2-3b"

# Setup Hugging Face InferenceClient
client = InferenceClient(token=HUGGINGFACE_API_KEY)

# Prompt template (modify as needed)
def get_prompt(user_query):
    return f"""### Task
Generate Python code for the following request:

{user_query}

### Code
```python
"""

# Main function
def generate_code(prompt: str) -> str:
    try:
        full_prompt = get_prompt(prompt)

        response = client.text_generation(
            prompt=full_prompt,
            model=model_name,
            max_new_tokens=256,
            temperature=0.2,
            stop=["```"]  # stop when code block ends
        )

        return response.strip()
    except Exception as e:
        return f"Error loading model: {str(e)}"

# For test purposes
if __name__ == "__main__":
    user_input = input("Enter your request (e.g., 'write a bubble sort in Python'): ")
    print("Generated Code:\n")
    print(generate_code(user_input))

from google import genai
from google.genai import types
import os
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()


# ptext="ear infection"
ptext="pink eye"
task = """
Generate a list of 10 medical terms related to the keywords above. Do not include any description.  
"""
prompt = f'{ptext}\n{task}'


# Count tokens using the new client method.
total_tokens = client.models.count_tokens(
    model="gemini-2.0-flash", contents=prompt
)
print('---')
print("total input tokens: ", total_tokens)
print('---')

response = client.models.generate_content(
    model="gemini-2.5-flash", contents=prompt,
    config=types.GenerateContentConfig(
        max_output_tokens = 200,
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
))
print("prompt: \n", prompt)
print("response: \n", response.text)
print('---')
print(response.usage_metadata)
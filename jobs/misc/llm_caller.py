import os
import torch  # Added to check for hardware
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, pipeline
# use hf auth login, then run this once. after, run with local_files_only=True.
# including a login command will make this thing try to always run online. 


device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

model_id = "google/gemma-3-270m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

if __name__ =='__main__':
    
    set_seed(42) # reproducablility
    ptext="ear infection"

    task = """
    ---
    Genrate a list of keywords related to the user key words above. 
    ---
    Response: 
    """
    prompt = f'{ptext}\n{task}'

    # load the model.     
    # Use the dynamic 'device' variable here
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    outputs = pipe(prompt, max_new_tokens=256, num_return_sequences=1)    
    print(outputs[0]["generated_text"])
    
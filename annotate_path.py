import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


st.set_page_config(page_title="Lung Cancer AI System", layout="wide")

@st.cache_resource
def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # Automatically maps model to GPU/CPU
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)


    model.eval()  
    return model, tokenizer

@torch.inference_mode()
def generate_response(model, tokenizer, prompt, generation_config):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024  
    ).to("cuda")


    pad_token_id = tokenizer.pad_token_id
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=generation_config.max_new_tokens,
        top_p=generation_config.top_p,
        temperature=generation_config.temperature,
        repetition_penalty=generation_config.repetition_penalty,
        do_sample=generation_config.do_sample,
        pad_token_id=pad_token_id  
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response




def configure_generation_settings():
    with st.sidebar:
        st.title("Generation Settings")
        max_new_tokens = 2500
        top_p = 0.9
        temperature = 0.00001
        repetition_penalty = 1.1
        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )
model_path = "/home/h392x566/DeepSeek-R1-Distill-Llama-8B"   
model, tokenizer = load_model_and_tokenizer(model_path)
st.sidebar.title("Model Settings")
st.sidebar.text(f"Model: {model_path}")
generation_config = configure_generation_settings()
full_prompt = "I am a helpful AI Lung Cancer Oncology Assistant."
response = generate_response(model, tokenizer, full_prompt, generation_config)
import pandas as pd

# Assume that model, tokenizer, and generation_config are defined.
# Also assume the function generate_response is defined as:
# response = generate_response(model, tokenizer, full_prompt, generation_config)

# Read the CSV file containing the extracted relations (e.g., 'extracted_relations.csv')
relations_df = pd.read_csv('relations_reduced.csv')
import re
def get_relation_direction(source, relation, target):
    """
    Build a prompt using source, relation, and target, then use the model to determine the direction.
    The expected outputs are:
      - suppressed: 0
      - active: 1
      - no relation: 2
      - not sure: 3
    """
    full_prompt = (
        f"I am a helpful AI Lung Cancer Oncology Assistant. Determine the relation direction between '{source}' and '{target}' "
        f"with relation '{relation}'. Just return one of these numbers: "
        "suppressed (0), active (1), no relation (2), or not sure (3). Do not include any explanation, chain-of-thought, or intermediate reasoning. Output format is: 'Final number is:'"
    )
    # Get the model's response
    response = generate_response(model, tokenizer, full_prompt, generation_config)
    
    # Attempt to parse the response into an integer.
    try:
        # print(response)
        pattern = r"Final number is:\s*(\d)"
        directions = re.search(pattern, response)
        # print(directions)
        dir = directions.group(1)
        # print(dir)
        direction = dir
    except Exception as e:
        # If parsing fails, default to 'not sure' (3)
        direction = 3
    return direction

# Apply the function to each row to get the relation direction
relations_df['Relation Direction'] = relations_df.apply(
    lambda row: get_relation_direction(row['Source Node'], row['Relation'], row['Target Node']),
    axis=1
)

# Write the updated DataFrame with the new column to a new CSV file
relations_df.to_csv('relations_with_direction.csv', index=False)

print("The relations with their directions have been written to 'relations_with_direction.csv'.")

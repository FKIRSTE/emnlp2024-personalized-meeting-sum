import os
import pandas as pd
import json
from backbone_model import BackboneModel
from config_loader import load_config

# Directory containing the JSON files

PATH = '' 
DIR = f'{PATH}dataset/'
CONFIG_PATH = f'{PATH}baseline/config_gpt.json'


# List to hold individual DataFrames
dataframes = []

# Function to concatenate items
def concat_items(items):
    if items is None:
        return ''
    return '\n\n'.join(
        f'filename: {item["filename"]}\ncontent: {(item.get("ocr_transcript") or "") + " " + (item.get("ocr_description") or "")}' for item in items
    )

def concat_ppt(items):
    if items is None:
        return ''
    return '\n\n'.join(
        f'filename: {item["filename"]}\ncontent: ' + '\n'.join(f'{k}: {v[0]}' if isinstance(v, list) and len(v) > 0 else f'{k}: {v}' for k, v in item["content"].items()) for item in items
    )

def build_ds(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            
            data = None
            with open(filepath, 'r') as file:
                data = json.load(file)
            
            # Creating DataFrame
            df = pd.DataFrame({
                "transcript": [data["transcript"]],
                "summary": [data["summary"]],
                "pens": [concat_items(data["pens"])],
                "whiteboard": [concat_items(data["whiteboard"])],
                "txt": [concat_items(data["shared-doc"]["txt"])],
                "ppt": [concat_ppt(data["shared-doc"]["ppt"])],
                "doc": [concat_items(data["shared-doc"]["doc"])]
            })      
            # Append the DataFrame to the list
            dataframes.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    final_df = pd.concat(dataframes, ignore_index=True)

    # Fill NaN values with empty strings and convert to string type
    final_df = final_df.fillna("").astype(str)

    final_df.to_csv(f'{DIR}ami_msi.csv', index=False)
    
df = pd.read_csv(f'{DIR}ami_msi.csv')

# Fill NaN values with empty strings and convert to string type
df = df.fillna("").astype(str)

print(df.columns)
print(len(df))

def build_prompt(transcript, pens, whiteboard, txt, ppt, doc):
    role = (
        "You are a professional meeting summarizer."
        "Your task is to generate abstractive summaries given a meeting transcript (main source) and additional material that may contain relevant information (secondary source)."
        "\n"
    )
    
    primary_source = (
        f"Consider the following transcript as your primary source to summarize the meeting:\n"
        f"<{transcript}\n"
    )
    
    # Initialize an empty list to hold the lines
    additional_sources_list = []
    additional_sources = ""

    # Check the lengths and append to the list if they contain content
    if len(pens) > 0:
        additional_sources_list.append(f"Notes by the participants: <{pens}>")

    if len(whiteboard) > 0:
        additional_sources_list.append(f"Drawings on the whiteboard: <{whiteboard}>")

    if len(txt) > 0:
        additional_sources_list.append(f"Shared text: <{txt}>")

    if len(ppt) > 0:
        additional_sources_list.append(f"Shared presentation: <{ppt}>")

    if len(doc) > 0:
        additional_sources_list.append(f"Shared documents: <{doc}>")

    # Only create the final string if there is at least one document with content
    if additional_sources_list:
        additional_sources = (
            "Here are some additional sources that may help you contextualize and summarize the meeting:\n" +
            "\n".join(additional_sources_list)
        )
    else:
        additional_sources = "No additional sources with content."

    format_prompt = (
        "Please write an abstractive, running text summary of the meeting transcript and use the additional sources - if available - for better contextualization and enhance quality."
        "Indicate, which additional sources you considered and provide a step-by-step reasoning (chain-of-thought) why you considered them."
        "Return everything in a json format: {summary: <your summary>, used sources: <chain-of-thought which sources were used and why, for every single source you used>}"
    )
    
    prompt = [
            {"role": "system", "content": role},
            {"role": "user", "content": f"{primary_source} \n\n {additional_sources} \n\n {format_prompt}"},
        ]
    return prompt

config = load_config(CONFIG_PATH)
model = BackboneModel(config, client_type="openai")

# Function to apply build_prompt and process_summary to each row
def apply_functions_to_row(row, model):
    # Build the prompt using the build_prompt function
    prompt = build_prompt(
        row['transcript'], 
        row['pens'], 
        row['whiteboard'], 
        row['txt'], 
        row['ppt'], 
        row['doc']
    )
    # Process the summary using the process_summary function
    predicted_summary = model.safe_model_call(prompt, 1000)
    print(predicted_summary, flush=True)
    return predicted_summary

# Apply the functions to each row and create a new column 'predicted_summary'
df['predicted_summary'] = df.apply(lambda row: apply_functions_to_row(row, model), axis=1)

# Save the modified DataFrame to a new CSV file
df.to_csv(f'{DIR}result/ami_ms_predicted_single.csv', index=False)
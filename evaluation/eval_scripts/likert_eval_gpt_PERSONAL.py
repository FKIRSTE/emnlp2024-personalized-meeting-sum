import pandas as pd
import logging
import json
import os
import numpy as np
import time
import random
from openai import AzureOpenAI
import re


# ************************************************************

with open('baseline/config.json') as config_file:
    config = json.load(config_file)

API_KEY = config["api_key"]
API_VERSION = config["api_version"]
ENDPOINT = config["endpoint"]
MODEL_NAME = config["model"]

client = AzureOpenAI(
    api_key=API_KEY,
    api_version= API_VERSION,
    azure_endpoint=ENDPOINT,
)

# ************************************************************

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PATH = 'dataset/'
sourcedirectory = f'{PATH}/<summary_location>/'

out_path = f'{PATH}/eval_output/'


# ************************************************************

eval_criteria = {
    "FAC": (
        "Your task is to assess the factuality of the given summary considering the meeting transcript."
        "Break down each sentence of the summary into atomic facts and assess if these facts align with the content of the transcript."
        "The summary should align with the transcript and not hallucinate any content - the summary should be a reliable image of the transcript."
        "Assign a score of 1 if there a severe problems with factuality and a score of 5 if the summary is perfectly fine."
    ),
    "INF": (
        "Criteria: Informativeness (INF) \n\n - The summary should be a concise and accurate representation of the main points and ideas of the article. It should avoid"
        "including irrelevant or minor details that are not essential to the article’s purpose or message. - The summary should have a clear"
        "and logical structure that follows the article’s original order of information, or provides a coherent alternative order if it improves"
        "the summarization. The summary should use transitions and connectors to link the sentences and paragraphs coherently. - The"
        "summary should use the same or similar terminology and tone as the article, unless there is a need to simplify or clarify some"
        "terms for the intended audience. The summary should avoid introducing new or unfamiliar words or concepts that are not in"
        "the article or relevant to the summary. - The summary should maintain the same perspective and point of view as the article,"
        "unless there is a reason to shift or contrast it. The summary should not express the summarizer’s own opinion, interpretation,"
        "or evaluation of the article, unless it is explicitly stated as such. - The summary should be grammatically correct and free of"
        "spelling, punctuation, and capitalization errors. The summary should use direct or indirect quotations and citations appropriately"
        "to acknowledge the source of the article. - The summary should be coherent and consistent with the article’s topic and genre."
        "The summary should avoid introducing information or claims that contradict or deviate from the article’s main message. The"
        "summary should also avoid repeating information or using unnecessary filler words."
    ),
    "REL": (
        "Criteria: Relevance (REL) \n\n - The summary should capture the main topic, events, and outcomes of the article in a concise and accurate way. - The summary"
        "should not omit any essential information that is necessary to understand the article’s purpose and significance. - The summary"
        "should not include any irrelevant or redundant details that distract from the article’s main points or introduce confusion. - The"
        "summary should use the same or similar terminology and tone as the article, unless the article uses obscure or jargon words"
        "that need to be simplified. - The summary should reflect the article’s structure and organization, presenting the information in"
        "a logical and coherent order. Examples of scoring: - Score 5: The summary meets all the criteria for relevance and provides a"
        "clear and comprehensive overview of the article, without any errors or gaps. - Score 4: The summary meets most of the criteria"
        "for relevance and provides a mostly clear and comprehensive overview of the article, but may have some minor errors or gaps,"
        "such as missing a minor detail, using a slightly different word, or omitting a transition. - Score 3: The summary meets some"
        "of the criteria for relevance and provides a partially clear and comprehensive overview of the article, but has some noticeable"
        "errors or gaps, such as missing a key detail, using a vague or inaccurate word, or skipping a logical connection. - Score 2:"
        "The summary meets few of the criteria for relevance and provides a vaguely clear and comprehensive overview of the article,"
        "but has many errors or gaps, such as missing several important details, using inappropriate or misleading words, or presenting"
        "the information in a confusing or contradictory order. - Score 1: The summary meets none or almost none of the criteria for"
        "relevance and provides a unclear and incomplete overview of the article, with severe errors or gaps, such as missing the main"
        "topic, using incorrect or irrelevant words, or omitting the entire conclusion."
        ),
    "OVR": (
        "Assess the quality of the given summary considering the transcript on the following criteria:"
        "1. The summary should not contain any content-wise redundant information, that does not aid the understanding or contextualization. \n"
        "2. The summary should be coherent, maintain logical flow, relevance, and clarity within a sentence and across sentences. \n"
        "3. The summary should use appropriate language with correct and grammatical use. Language should not be ambiguous. \n"
        "4. The summary should not ommit relevant content. Neither should content be completely absent or relevant details be missing. \n"
        "5. The summary should correctly reference statements and actions to the respective meeting participant. \n"
        "6. The summary should not add hallucinated content. This includes the additional of new content not present in the transcript as well as changing details. \n"
        "7. The summary should maintain the logical and temporal structure and not misplace topics or events. \n"
        "8. The summary should not contain irrelevant information but fovus on what is important. \n"
        "When encountering issues with any of these criteria, assess the impact and rate accordingly. Omission and hallucinated content are more severe issues than the other."
        "Provide a single overall score, with a score of 1 being an awful summary and a score of 5 being a flawless summary."
        "Again, only return one single score thate rates the overall quality of the summary."
        "Do not return one score per criteria."
    ),
}


def save_df_to_csv(df, file_name):
    """
    Save a DataFrame to a CSV file.
    """
    df.to_csv(file_name, index=False)

def call_gpt(message, max_tokens):
    """
    Call the GPT-4 API to generate completions for the given message.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=message,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.01,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        return response.choices[0].message.content.strip() # type: ignore
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}") from e

def secure_model_call(prompt, base_delay=3.0, max_attempts=6, max_tokens=2000):
    attempt = 0
    ranking = "0"
    print("Calling model...", flush=True)
    while attempt < max_attempts:
        try:
            ranking = call_gpt(prompt, max_tokens)
            break
        except Exception as e:
            if "429" in str(e):  # Check if the exception is due to rate limiting
                # Exponential backoff with jitter
                sleep_time = (2 ** (attempt+1)) + \
                    (random.randint(0, 1000) / 1000)
                logging.warning("Rate limit hit, backing off for %s seconds.", sleep_time)
                time.sleep(sleep_time)
                attempt += 1
            else:
                logging.error("Error encountered: %s", str(e))
                break  # Break the loop on non-rate limiting errors
        finally:
            # Sleep to ensure compliance with API rate limits
            time.sleep(base_delay)
    return ranking

def parse_ranking(rankings, criteria="OOO"):
    """
    Parse the ranking response from the GPT-4 API.
    """
    match = re.search(r'```json\n({.*?})\n```', rankings, re.DOTALL)
    if match:
        json_string = match.group(1)
        
        # Load the JSON string into a dictionary
        json_object = json.loads(json_string)
        
        # Assuming the JSON object contains only one key-value pair
        key, value = next(iter(json_object.items()))

        # Create a new dictionary with your own key and the extracted value
        new_dict = {criteria: value}

        # Display the new dictionary
        print(new_dict)
        return new_dict

    else:
        print("No JSON object found")
        json_object = {
            criteria: 0,
        }
        print(json_object)
        return json_object

def build_prompt(transcript, summary, criteria, persona):
    """
    Build the prompt for the GPT-4 API based on the input data.
    """
    role = (
        "You are an expert in the field of summarizing meetings and are tasked with evaluating the quality of the following summary."
        "Score the summary according to the scoring criteria with a likert score between 1 (worst) to 5 (best)."
    )

    material = (
        f"Transcript: <{transcript}>\n"
        f"Summary: <{summary}>\n"
        f"Criteria: <{criteria}>\n"
        f"Target Persona: <{persona}>\n" 
    )
    
    task = (
        "Your task is to rank the summaries based on the criteria provided."
        "The summary you evaluate should fit to the given transcript and the target persona."
        "Really evaluate if the summary fits the given criteria and the persona."
        "Remember to consider the quality of the summaries and how well they capture the key points of the original transcript."
        "First provide an argumentation for your ranking. Therefore, use chain-of-thought and think step by step."
        "Return a json object with the ranking for the evaluation criteria."
        "The output should be in the following format:"
        "<explanation, step-by-step> \n\n ! \n\n <json object>"
        "The json object should follow the structure ```json \n {<evaluation criteria> : <Likert Score>} \n```"
        "The json object should only contain the single likert score for the currently assessed criteria."
    )

    prompt = [
        {"role": "system", "content": f"{role}"},
        {"role": "user", "content": f"{material}\n\n{task}"},
    ]


    return prompt

def compute_scores(transcript, summary, persona):
    """
    Compute the scores for the given transcript, summary, and predicted summary.
    """
    
    one_row_scores = {}
    
    for criteria, description in eval_criteria.items():
        prompt = build_prompt(transcript, summary, description, persona)
        score = secure_model_call(prompt)
        logging.info("Score for criteria %s: %s", criteria, score)
        extracted_score = parse_ranking(score, criteria)
        
        for key, value in extracted_score.items():
            one_row_scores[key] = value

    return one_row_scores


logger.info("Reading dataset...")


all_dirs = os.listdir(sourcedirectory)
complete_dirs = []


# dirs is all dirs without the complete dirs
dirs = [i for i in all_dirs if i not in complete_dirs]
print(len(dirs))

# summaries = {'Summary':'Gold'}
_TRANSCRIPT = "Modified Transcript"


output_csv_path = os.path.join(out_path, 'output.csv')
header_written = False
all_roles = set()
        
        
if not os.path.exists(output_csv_path):
    with open(output_csv_path, 'w') as f:
        pass

# First pass to determine all unique roles
for item in dirs:
    try:
        path = os.path.join(os.getcwd(), sourcedirectory, item)
        with open(path, encoding="utf8") as f:
            jsondict = json.load(f)

        for participant in jsondict.get('Meeting Participants', []):
            role = participant.get('role', 'Unknown Role')
            all_roles.add(role)
    except Exception as e:
        logger.exception("Error occurred with file: %s", item)

# Process each item in dirs
for item in dirs:
    try:
        path = os.path.join(os.getcwd(), sourcedirectory, item)
        with open(path, encoding="utf8") as f:
            jsondict = json.load(f)

        # personas = jsondict.get('Personas', [])

        for participant in jsondict.get('Meeting Participants', []):
            transcript = participant.get(_TRANSCRIPT, '')
            role = participant.get('role', 'Unknown Role')
            summaries_list = participant.get('automated_summary', [])
            #persona = personas.get(role, "")
            
            persona = role
            if persona == "General Summary":
                print(f"Skipping {role}...")
                continue
            
            print(f"Processing {role}...")
            print(f"Transcript: {transcript[:100]}")
            print(f"Persona: {persona[:100]}")

            for summary_dict in summaries_list:
                summary = summary_dict.get('summary', '')
                logger.info("***** Computing scores for %s for %s *****", role, item)
                score_pre = compute_scores(transcript, summary, persona)

                row = {
                    'Item': item,
                    'Transcript': transcript
                }
                
                # Initialize all role columns with empty strings or default values
                for r in all_roles:
                    row[f'Summary_{r}'] = ''
                    row[f'Score_{r}'] = ''

                # Assign the current role's summary and score
                row[f'Summary_{role}'] = summary
                row[f'Score_{role}'] = score_pre

                # Convert the row dictionary to a DataFrame
                row_df = pd.DataFrame([row])

                # Save to CSV, appending and avoiding header if already written
                with open(output_csv_path, 'a', encoding='utf8', newline='') as f:
                    row_df.to_csv(f, header=not header_written, index=False)
                    header_written = True

    except Exception as e:
        logger.exception("Error occurred with file: %s", item)
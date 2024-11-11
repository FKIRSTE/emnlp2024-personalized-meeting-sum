from groq import Groq
from openai import OpenAI
import openai
from openai import AzureOpenAI
import time
import random
import logging


def askllama(prompt, key):
  client = Groq(
    api_key=key,
    )
  chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": prompt},
                    ],
        }
    ],
    model="llama3-8b-8192",
    )

  return chat_completion.choices[0].message.content


def askgptcomplex(prompt, key, max_tokens, system="", max_attempts=6, base_delay=3.0):

    attempt = 0
    while attempt < max_attempts:
        try:
            response = gpt_model_call(prompt, key, max_tokens, system=system)
            print(response)
            return response
        except Exception as e:
            if "429" in str(e):
                sleep_time = (2 ** (attempt + 1)) + (random.randint(0, 1000) / 1000)
                logging.warning(f"Rate limit hit, backing off for {sleep_time} seconds.", flush=True)
                time.sleep(sleep_time)
                attempt += 1
            else:
                print(f"Error encountered: {str(e)}", flush=True)
                break
        finally:
            time.sleep(base_delay)



def gpt_model_call(prompt, key, max_tokens, system):
    client = AzureOpenAI(
                api_key=key,
                api_version="2024-02-01",
                azure_endpoint="<YOUR API ENDPOINT>"
            )



    try:
        response = client.chat.completions.create(
            model="gpt4-turbo",
            messages= [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}") from e
    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass

def askgpt(prompt, key, max_tokens):
    return askgptcomplex(prompt, key, max_tokens)

    client = OpenAI(
        api_key=key
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass


def askgptsystem(prompt, system, key, max_tokens):
    return askgptcomplex(prompt, key, max_tokens, system)

    client = OpenAI(
        api_key=key
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        pass
    except openai.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        pass
    except openai.RateLimitError as e:
        # Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        pass


def askgptjson(text, system, key, max_tokens, max_attempts=6, base_delay=3.0):

    attempt = 0

    while attempt < max_attempts:
        try:
            response = gpt_model_call(text, key, max_tokens, system=system)
            # print(response)
            return response
        except Exception as e:
            if "429" in str(e):
                sleep_time = (2 ** (attempt + 1)) + (random.randint(0, 1000) / 1000)
                print(f"Rate limit hit, backing off for {sleep_time} seconds.", flush=True)
                time.sleep(sleep_time)
                attempt += 1
            else:
                print(f"Error encountered: {str(e)}", flush=True)
                break
        finally:
            time.sleep(base_delay)



def build_prompt(transcript, summary, criteria):
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
    )

    task = (
        "Your task is to rank the summaries based on the criteria provided."
        "Remember to consider the quality of the summaries and how well they capture the key points of the original transcript."
        "First provide an argumentation for your ranking. Therefore, use chain-of-thought and think step by step."
        "Return a json object with the ranking for the evaluation criteria."
        "The output should be in the following format:"
        "<explanation, step-by-step> \n\n ! \n\n <json object>"
        "The json object should follow the structure <evaluation criteria> : <Likert Score>"
        "The json object should only contain the single likert score for the currently assessed criteria."
    )

    prompt = [
        {"role": "system", "content": f"{role}"},
        {"role": "user", "content": f"{material}\n\n{task}"},
    ]


    return prompt
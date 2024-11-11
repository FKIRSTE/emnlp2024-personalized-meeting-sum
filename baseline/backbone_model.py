import logging
import random
import time
from openai import AzureOpenAI


class BackboneModel:
    def __init__(self, config, client_type="openai"):
        self.client = self.init_model(config, client_type)
        self.model_name = config["model"]
        self.max_tokens_feedback = config.get("max_tokens_feedback", 4000)
        self.max_tokens_refinement = config.get("max_tokens_refinement", 200)
        self.client_type = client_type
 
    def init_model(self, config, client_type):
        api_key = config.get("api_key")
        api_version = config.get("api_version")
        endpoint = config.get("endpoint")
        if client_type == "openai":
            print("Using Azure OpenAI")
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )

        return client

    def model_call(self, message, max_tokens):
        print(f"calling model {self.model_name}")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=0.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            print("API call successful")
            return response.choices[0].message.content.strip()
        
        except APIError as e:
            # Catch OpenAI API-specific errors
            raise Exception(f"API call failed with status code {e.http_status}: {str(e)}") from e
        except APIConnectionError as e:
            # Catch network-related errors
            raise Exception(f"API connection error: {str(e)}") from e
        except RateLimitError as e:
            # Catch rate limit errors
            raise Exception(f"Rate limit error: {str(e)}") from e
        except InvalidRequestError as e:
            # Catch invalid request errors
            raise Exception(f"Invalid request: {str(e)}") from e
        except AuthenticationError as e:
            # Catch authentication errors
            raise Exception(f"Authentication error: {str(e)}") from e
        except OpenAIError as e:
            # Catch all other OpenAI errors
            raise Exception(f"OpenAI error: {str(e)}") from e
        except Exception as e:
            # Catch any other exceptions
            raise Exception(f"An unexpected error occurred: {str(e)}") from e


    def safe_model_call(self, message, max_tokens, max_attempts=6, base_delay=3.0):
        attempt = 0
        while attempt < max_attempts:
            try:
                model_to_call = self.model_call
                response = model_to_call(message, max_tokens)
                return response
            except Exception as e:
                if "429" in str(e):
                    sleep_time = (2 ** (attempt + 1)) + \
                        (random.randint(0, 1000) / 1000)
                    logging.warning(
                        "Rate limit hit, backing off for %s seconds.", sleep_time)
                    time.sleep(sleep_time)
                    attempt += 1
                else:
                    logging.error("Error encountered: %s", str(e))
                    attempt += 1  # Increment attempt for errors other than 429
                    time.sleep(base_delay)

        return "ERROR: Max attempts reached. Could not get a successful response."
        raise Exception(
            "Max attempts reached. Could not get a successful response.")

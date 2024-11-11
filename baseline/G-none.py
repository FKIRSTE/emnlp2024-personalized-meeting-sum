import pandas as pd
import logging
import json
import os
import numpy as np
import time
import random
from openai import AzureOpenAI
import re

from backbone_model import BackboneModel
from config_loader import load_config

# ************************************************************

PATH = '' 
DIR = f'{PATH}dataset/'
CONFIG_PATH = f'{PATH}baseline/config_gpt.json'

config = load_config(CONFIG_PATH)
model = BackboneModel(config, client_type="openai")


# ************************************************************
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

out_path = f'{PATH}/dataset/result/'


# ************************************************************

logger.info("Reading dataset...")

all_dirs = ['ES2002a.json', 'ES2002b.json', 'ES2002c.json', 'ES2002d.json', 'ES2003a.json', 'ES2003b.json', 'ES2003c.json', 'ES2003d.json', 'ES2004a.json', 'ES2004b.json', 'ES2004c.json', 'ES2004d.json', 'ES2005a.json', 'ES2005b.json', 'ES2005c.json', 'ES2005d.json', 'ES2006a.json', 'ES2006b.json', 'ES2006c.json', 'ES2006d.json', 'ES2007a.json', 'ES2007b.json', 'ES2007c.json', 'ES2007d.json', 'ES2008a.json', 'ES2008b.json', 'ES2008c.json', 'ES2008d.json', 'ES2009a.json', 'ES2009b.json', 'ES2009c.json', 'ES2009d.json', 'ES2010a.json', 'ES2010b.json', 'ES2010c.json', 'ES2010d.json', 'ES2011a.json', 'ES2011b.json', 'ES2011c.json', 'ES2011d.json', 'ES2012a.json', 'ES2012b.json', 'ES2012c.json', 'ES2012d.json', 'ES2013a.json', 'ES2013b.json', 'ES2013c.json', 'ES2013d.json', 'ES2014a.json', 'ES2014b.json', 'ES2014c.json', 'ES2014d.json', 'ES2015a.json', 'ES2015b.json', 'ES2015c.json', 'ES2015d.json', 'ES2016a.json', 'ES2016b.json', 'ES2016c.json', 'ES2016d.json', 'IS1000a.json', 'IS1000b.json', 'IS1000c.json', 'IS1000d.json', 'IS1001a.json', 'IS1001b.json', 'IS1001c.json', 'IS1001d.json', 'IS1002b.json', 'IS1002c.json', 'IS1002d.json', 'IS1003a.json', 'IS1003b.json', 'IS1003c.json', 'IS1003d.json', 'IS1004a.json', 'IS1004b.json', 'IS1004c.json', 'IS1004d.json', 'IS1005a.json', 'IS1005b.json', 'IS1005c.json', 'IS1006a.json', 'IS1006b.json', 'IS1006c.json', 'IS1006d.json', 'IS1007a.json', 'IS1007b.json', 'IS1007c.json', 'IS1007d.json', 'IS1008a.json', 'IS1008b.json', 'IS1008c.json', 'IS1008d.json', 'IS1009a.json', 'IS1009b.json', 'IS1009c.json', 'IS1009d.json', 'TS3003a.json', 'TS3003b.json', 'TS3005a.json', 'TS3005b.json', 'TS3005c.json', 'TS3005d.json', 'TS3006a.json', 'TS3006b.json', 'TS3006c.json', 'TS3007a.json', 'TS3007b.json', 'TS3008a.json', 'TS3008b.json', 'TS3008c.json', 'TS3008d.json', 'TS3009a.json', 'TS3009b.json', 'TS3009c.json', 'TS3009d.json', 'TS3010a.json', 'TS3010b.json', 'TS3010c.json', 'TS3011a.json', 'TS3011b.json', 'TS3011c.json', 'TS3012a.json', 'TS3012b.json']


complete_dirs = pd.read_csv(f'{out_path}summaries_gpt4.csv')['Item'].tolist()

# Strip the '_summary' suffix from complete_dirs
complete_dirs_stripped = [filename.replace('_summary', '') for filename in complete_dirs]

# dirs is all dirs without the complete dirs
dirs = [i for i in all_dirs if i not in complete_dirs_stripped]



_TRANSCRIPT = 'transcript'

output_df = pd.DataFrame()
for item in dirs:
    try:
        logging.info("Summarizing file: %s", item)
        path = os.path.join(os.getcwd(), DIR, item)
        with open(path, encoding="utf8") as f:
            jsondict = json.load(f)

        transcript = jsondict.get(_TRANSCRIPT, '')
        row = {'Item': item, 'Transcript': transcript}
            
        system_prompt = f"You are a professional meeting summarizer. Generate an abstractive summary using at most 250 tokens."
        user_prompt = f"Summarize the following meeting transcript: {transcript}"
        prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
        summary = model.secure_model_call(prompt, 250)

        row['Predicted_GPT4'] = summary
                
        # Convert the row dictionary to a DataFrame and concatenate
        row_df = pd.DataFrame([row])
        output_df = pd.concat([output_df, row_df], ignore_index=True)
        
    except Exception as e:
        logging.exception("Error occurred with file: %s", item)

output_csv_path = os.path.join(out_path, 'summaries_gpt4_p2.csv')
output_df.to_csv(output_csv_path, index=False)
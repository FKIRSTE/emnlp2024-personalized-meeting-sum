import nltk
import os
import AMI_sum_RAG as sumeval
import logging
import json

llamakey = ""
f = open("baseline/gptkey.txt", "r")

CONFIG_PATH = f'baseline/config_gpt.json'
config = None
with open(CONFIG_PATH, encoding='utf-8') as config_file:
    config = json.load(config_file)
gptkey = config.get("api_key")


max_tokens = 1000
max_attempts = 6
vector_size = 20
sourcedirectory = 'code/ami_collection'  # where you have your JSON files containing all the documents
sumtargetdirectory = 'code/simple_RAG_output'  # where to save to after the summary step
evaltargetdirectory = 'code/eval_output'  # where to save to after the evaluation step

# dirs = os.listdir(sourcedirectory)

# all_dirs = ['ES2012b.json', 'TS3006a.json', 'IS1007a.json', 'ES2016c.json', 'ES2007a.json', 'ES2013d.json', 'ES2008d.json', 'TS3008b.json', 'IS1008d.json', 'TS3009d.json', 'IS1009b.json', 'ES2009b.json', 'ES2014c.json', 'TS3011b.json', 'IS1005a.json', 'ES2011d.json', 'ES2005a.json', 'ES2010b.json', 'ES2015d.json', 'IS1001a.json', 'ES2010c.json', 'ES2014b.json', 'TS3011c.json', 'ES2009c.json', 'IS1009c.json', 'TS3008c.json', 'ES2016b.json', 'ES2003a.json', 'ES2012c.json', 'IS1003a.json', 'ES2008b.json', 'TS3008d.json', 'IS1008b.json', 'TS3009b.json', 'IS1009d.json', 'ES2009d.json', 'TS3005a.json', 'ES2011b.json', 'ES2010d.json', 'ES2004a.json', 'ES2015c.json', 'TS3010b.json', 'IS1004a.json', 'ES2006a.json', 'ES2012d.json', 'IS1006a.json', 'TS3012b.json', 'TS3007a.json', 'ES2013b.json', 'ES2013c.json', 'ES2002a.json', 'ES2016d.json', 'TS3003a.json', 'ES2015b.json', 'TS3010c.json', 'IS1000a.json', 'ES2011c.json', 'ES2014d.json', 'TS3009c.json', 'IS1008c.json', 'ES2008c.json', 'IS1004c.json', 'ES2015a.json', 'IS1001d.json', 'ES2004c.json', 'IS1000b.json', 'TS3005c.json', 'IS1002b.json', 'ES2002b.json', 'ES2003d.json', 'IS1006c.json', 'TS3003b.json', 'ES2006c.json', 'IS1003d.json', 'ES2006b.json', 'IS1006b.json', 'TS3012a.json', 'IS1007d.json', 'TS3007b.json', 'ES2002c.json', 'IS1002c.json', 'ES2013a.json', 'ES2007d.json', 'ES2008a.json', 'IS1008a.json', 'TS3009a.json', 'TS3005b.json', 'ES2005d.json', 'ES2011a.json', 'IS1000c.json', 'ES2004b.json', 'TS3010a.json', 'IS1004b.json', 'ES2007c.json', 'IS1002d.json', 'ES2016a.json', 'ES2002d.json', 'IS1007c.json', 'TS3006c.json', 'ES2003b.json', 'IS1003b.json', 'IS1001b.json', 'IS1000d.json', 'ES2005c.json', 'IS1005c.json', 'ES2014a.json', 'TS3008a.json', 'IS1009a.json', 'ES2009a.json', 'TS3011a.json', 'TS3005d.json', 'IS1005b.json', 'ES2005b.json', 'ES2004d.json', 'ES2010a.json', 'IS1001c.json', 'IS1004d.json', 'IS1003c.json', 'ES2012a.json', 'ES2006d.json', 'IS1006d.json', 'TS3006b.json', 'ES2003c.json', 'IS1007b.json', 'ES2007b.json']


# all_dirs = ['ES2002a.json', 'ES2002b.json', 'ES2002c.json', 'ES2002d.json', 'ES2003a.json', 'ES2003b.json', 'ES2003c.json', 'ES2003d.json', 'ES2004a.json', 'ES2004b.json', 'ES2004c.json', 'ES2004d.json', 'ES2005a.json', 'ES2005b.json', 'ES2005c.json', 'ES2005d.json', 'ES2006a.json', 'ES2006b.json', 'ES2006c.json', 'ES2006d.json', 'ES2007a.json', 'ES2007b.json', 'ES2007c.json', 'ES2007d.json', 'ES2008a.json', 'ES2008b.json', 'ES2008c.json', 'ES2008d.json', 'ES2009a.json', 'ES2009b.json', 'ES2009c.json', 'ES2009d.json', 'ES2010a.json', 'ES2010b.json', 'ES2010c.json', 'ES2010d.json', 'ES2011a.json', 'ES2011b.json', 'ES2011c.json', 'ES2011d.json', 'ES2012a.json', 'ES2012b.json', 'ES2012c.json', 'ES2012d.json', 'ES2013a.json', 'ES2013b.json', 'ES2013c.json', 'ES2013d.json', 'ES2014a.json', 'ES2014b.json', 'ES2014c.json', 'ES2014d.json', 'ES2015a.json', 'ES2015b.json', 'ES2015c.json', 'ES2015d.json', 'ES2016a.json', 'ES2016b.json', 'ES2016c.json', 'ES2016d.json', 'IS1000a.json', 'IS1000b.json', 'IS1000c.json', 'IS1000d.json', 'IS1001a.json', 'IS1001b.json', 'IS1001c.json', 'IS1001d.json', 'IS1002b.json', 'IS1002c.json', 'IS1002d.json', 'IS1003a.json', 'IS1003b.json', 'IS1003c.json', 'IS1003d.json', 'IS1004a.json', 'IS1004b.json', 'IS1004c.json', 'IS1004d.json', 'IS1005a.json', 'IS1005b.json', 'IS1005c.json', 'IS1006a.json', 'IS1006b.json', 'IS1006c.json', 'IS1006d.json', 'IS1007a.json', 'IS1007b.json', 'IS1007c.json', 'IS1007d.json', 'IS1008a.json', 'IS1008b.json', 'IS1008c.json', 'IS1008d.json', 'IS1009a.json', 'IS1009b.json', 'IS1009c.json', 'IS1009d.json', 'TS3003a.json', 'TS3003b.json', 'TS3005a.json', 'TS3005b.json', 'TS3005c.json', 'TS3005d.json', 'TS3006a.json', 'TS3006b.json', 'TS3006c.json', 'TS3007a.json', 'TS3007b.json', 'TS3008a.json', 'TS3008b.json', 'TS3008c.json', 'TS3008d.json', 'TS3009a.json', 'TS3009b.json', 'TS3009c.json', 'TS3009d.json', 'TS3010a.json', 'TS3010b.json', 'TS3010c.json', 'TS3011a.json', 'TS3011b.json', 'TS3011c.json', 'TS3012a.json', 'TS3012b.json']

all_dirs = os.listdir(sourcedirectory)

complete_dirs = os.listdir(sumtargetdirectory)
print(complete_dirs)
print(len(complete_dirs))

# sth = ['ES2002a_summary.json', 'ES2002b_summary.json', 'ES2002c_summary.json', 'ES2002d_summary.json', 'ES2003a_summary.json', 'ES2003b_summary.json', 'ES2003d_summary.json', 'ES2004a_summary.json', 'ES2004c_summary.json', 'ES2004d_summary.json', 'ES2005a_summary.json', 'ES2005b_summary.json', 'ES2005c_summary.json', 'ES2005d_summary.json', 'ES2006a_summary.json', 'ES2006b_summary.json', 'ES2006c_summary.json', 'ES2006d_summary.json', 'ES2007a_summary.json', 'ES2007b_summary.json', 'ES2007c_summary.json', 'ES2007d_summary.json', 'ES2008a_summary.json', 'ES2008b_summary.json', 'ES2008d_summary.json', 'ES2009a_summary.json', 'ES2009b_summary.json', 'ES2009c_summary.json', 'ES2009d_summary.json', 'ES2010a_summary.json', 'ES2010b_summary.json', 'ES2010c_summary.json', 'ES2010d_summary.json', 'ES2011b_summary.json', 'ES2011c_summary.json', 'ES2011d_summary.json', 'ES2012a_summary.json', 'ES2012b_summary.json', 'ES2012c_summary.json', 'ES2012d_summary.json', 'ES2013a_summary.json', 'ES2013b_summary.json', 'ES2013c_summary.json', 'ES2013d_summary.json', 'ES2014a_summary.json', 'ES2014b_summary.json', 'ES2014c_summary.json', 'ES2014d_summary.json', 'ES2015a_summary.json', 'ES2015b_summary.json', 'ES2015c_summary.json', 'ES2015d_summary.json', 'ES2016a_summary.json', 'ES2016b_summary.json', 'ES2016c_summary.json', 'ES2016d_summary.json', 'IS1000a_summary.json', 'IS1000d_summary.json', 'IS1001a_summary.json', 'IS1001b_summary.json', 'IS1001c_summary.json', 'IS1001d_summary.json', 'IS1002b_summary.json', 'IS1002c_summary.json', 'IS1002d_summary.json', 'IS1003a_summary.json', 'IS1003b_summary.json', 'IS1003c_summary.json', 'IS1003d_summary.json']

# Strip the '_summary' suffix from complete_dirs
complete_dirs_stripped = [filename.replace('_summary', '') for filename in complete_dirs]


# dirs is all dirs without the complete dirs
dirs = [i for i in all_dirs if i not in complete_dirs_stripped]
print(len(dirs))

list = []
for i in dirs:
    try:
        sumeval.runthrough(i, sourcedirectory, sumtargetdirectory, evaltargetdirectory, llamakey, gptkey, max_tokens, max_attempts, vector_size, only_general_summary=False)
    except Exception as e:
        logging.exception("Error occurred with file: %s", i)

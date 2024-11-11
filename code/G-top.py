import os
import openai
from openai import OpenAI
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import numpy
import chatwithmodels as cwm
import re
import ast
import logging


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Mute the logging for the specific package
logging.getLogger('gensim').setLevel(logging.WARNING)


def generate_summaries(item, sourcedirectory, llamakey, gptkey, vector_size, max_attempts, max_tokens, only_general_summary=False):

    output = {'Success': 'False'}
    logging.info("********** Processing %s **********", item)

    path = os.path.join(os.getcwd(), sourcedirectory, item)
    with open(path, encoding="utf8") as f:
        jsondict = json.load(f)

    transcript = jsondict.get('transcript', '')
    output['Transcript'] = transcript
    output['Summary'] = jsondict.get('summary', '')

    whiteboard = jsondict.get('whiteboard') or []
    for i in whiteboard:
        i['content'] = i.get('ocr_transcript', '')

    pens = jsondict.get('pens') or []
    for i in pens:
        i['content'] = i.get('ocr_transcript', '')

    shared_doc = jsondict.get('shared-doc') or {}
    supplementary = (pens + whiteboard +
                     (shared_doc.get('txt') or []) +
                     (shared_doc.get('doc') or []))
    ppt = shared_doc.get('ppt') or []

    # merges each ppt extract into one continuous string
    for i in ppt:
        content = []
        for j in i.get('content', {}):
            content.extend(i['content'][j])
        content = "\n".join(content)
        i['content'] = content
        supplementary.append(i)

    meeting_participants = []
    success = True
    if not only_general_summary:
        # success, meeting_participants = failsafe(get_participants, max_attempts, transcript=transcript, gptkey=gptkey, max_tokens=max_tokens)
        success = True
        meeting_participants = [
            'User Experience', 'Marketing', 'Project Manager', 'Industrial Design']

    meeting_participants.append('General Summary')
    logging.info("Meeting Participants: %s", meeting_participants)

    if not success:
        output['Meeting Participants'] = []
        return output, success

    output['Meeting Participants'] = meeting_participants

    overallsuccess = True

    print()
    logging.info("********** Identifying top k documents **********")
    # success, topk = failsafe(get_topk, max_attempts, transcript=transcript,
    # supplementary=supplementary, vector_size=vector_size, topk=3)
    success, topk = get_topk(transcript, supplementary, vector_size, 5)
    if not success:
        overallsuccess = False

    if not overallsuccess:
        return output, overallsuccess

    # topk should be a list or a dictionary ?

    output['Top k Documents'] = topk
    topk_pertinent = topk[0]['pertinent']
    # print(topk_pertinent, flush=True)


    summaries = []
    gptentry = {'summarizer': 'gpt-4-turbo'}
    gptentry['summary'] = cwm.askgpt(f"Summarize this transcript. Create an abstractive summary. Make the summary 250 tokens or less. {transcript} \n\n Also consider these additional sources: {topk_pertinent}", gptkey, 250)
    summaries.append(gptentry)
    automated_summary = summaries

    output['automated_summary'] = automated_summary

    # output['automated_summary'] = {'summary': cwm.askgpt("Summarize this transcript. " + transcript, gptkey, max_tokens)}

    return output, overallsuccess


def get_modified_transcript(transcript, questions):
    for i in questions:
        length = len(i['insert'])
        start = transcript.find(i['insert'])
        end = start + length
        transcript = transcript[:end] + ' [' + \
            i['answer']['answertext'] + ']' + transcript[end:]
    return transcript


def get_participants(transcript, gptkey, max_tokens):
    logging.info("********** Participant identification **********")
    participants_answer = cwm.askgptsystem("Who are the participants in the following meeting? The different participants are typically the names at the start of a speaker turn, e.g., User Experience, Marketing, Project Manager, Industrial Design. Provide your answer in the form of a Python array: '[<name 1>, <name 2>, ...]'" + transcript,
                                           "Provide your answer in the form of a Python array: '[<name 1>, <name 2>, ...]'", gptkey, max_tokens)

    # Remove the starting and ending markers
    cleaned_string = re.sub(r'```(python|json)\n', '', participants_answer)
    cleaned_string = re.sub(r'\n```', '', cleaned_string)

    # Convert the cleaned string to a list using ast.literal_eval
    output_list = ast.literal_eval(cleaned_string)
    #    participants_answer = re.search(r'^[^\[]*(\[[\s\S]*\])[^\]]*$', participants_answer).group(0)
    meeting_participants = output_list  # ast.literal_eval(participants_answer)

    return meeting_participants


def get_questions(participant, transcript, key, max_tokens):
    logging.info("********** Questions for %s **********", participant)
    system = (
        "Take the role of a question generator that takes the role of a defined participant and points out unclarities and open questions in a transcriot. Generate at most 5 questions. Only ask the 5 most relevant questions."
    )

    user_p1 = "" if (participant == 'General Summary') else f"If you were participant '{participant}',"

    user_p2 = (
        f" what open questions would you still have in regards to the following transcript: {transcript}?"
        "Your answer shall only contain a Python array of dictionaries: '[{<question>, <insert>}, {<question>, <insert>}, {<question>, <insert>}, ...]'. "
        "Each dict must contain an entry called 'question' containing the question itself and an entry called 'insert' containing an exact copy of the sentence from the transcript that is most relevant to the question."
    )
    user = user_p1 + user_p2
    open_questions = cwm.askgptsystem(user, system, gptkey, max_tokens)
    #open_questions = re.search(r'^[^\[]*(\[[\s\S]*\])[^\]]*$', open_questions).group(0)

    open_questions = re.sub(r'```(python|json)\n', '', open_questions)
    open_questions = re.sub(r'\n```', '', open_questions)

    print("< " + open_questions + " >", flush=True)

    list_of_dicts = ast.literal_eval(open_questions)
    # list_of_dicts = list_of_dicts[:5]
    outputlist = []
    count_true = 0
    for i in list_of_dicts:
        if i['insert'] in transcript:
            count_true += 1
            outputlist.append(i)
        if count_true == 5:
            break
    logging.info("Number of matching insertions: %s / 5", count_true)
    return outputlist

def get_topk(transcript, supplementary, vector_size, topk):
    supplementary_data = []
    for i in supplementary:
        if 'sum.' not in i['filename'].lower() and 'final.report.doc' not in i['filename'].lower():
            supplementary_data.append(i)

    quevectors, docvectors = vectorize(transcript, supplementary_data, [transcript], vector_size)



    # Assuming quevectors and docvectors are lists of dictionaries containing the vectors under the key 'vector'
    for query in quevectors:
        # Convert query vector and document vectors to numpy arrays
        query_vector = np.array(query['vector']).reshape(1, -1)
        doc_vectors = np.array([doc['vector'] for doc in docvectors])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, doc_vectors)[0]

        # Combine similarities with document vectors
        distances = [{'distance': similarity, 'vector': docvectors[i]} for i, similarity in enumerate(similarities)]

        # Sort the distances list by similarity in descending order (higher similarity is better)
        distances.sort(key=lambda x: x['distance'], reverse=True)

        # Get the top k elements with the highest similarity
        top_k = distances[:topk]

        # Add the top k results to the query
        query['pertinent'] = top_k

    # for i in quevectors:
    #     distances = []
    #     for j in range(len(docvectors)):
    #         dist = numpy.linalg.norm(docvectors[j]['vector'] - i['vector'])
    #         distances.append({'distance': dist, 'vector': docvectors[j]})
    #     # Sort the distances list by the distance
    #     distances.sort(key=lambda x: x['distance'])
    #     # Get the top k elements with the smallest distance
    #     top_k = distances[:topk]
    #     i['pertinent'] = top_k

    return True, quevectors



def answer_questions(questions, transcript, supplementary, vector_size, key, max_attempts, max_tokens):
    supplementary_data = []

    for i in supplementary:
        if 'sum.' not in i['filename'].lower() and 'final.report.doc' not in i['filename'].lower():
            supplementary_data.append(i)

    quevectors, docvectors = vectorize(
        transcript, supplementary_data, questions, vector_size)

    for i in quevectors:
        distances = []
        for j in range(len(docvectors)):
            dist = numpy.linalg.norm(docvectors[j]['vector'] - i['vector'])
            distances.append({'distance': dist, 'vector': docvectors[j]})
        dist = min(distances, key=lambda x: x['distance'])
        i['pertinent'] = dist['vector']

    success = True
    for i in quevectors:
        success, i['answer'] = failsafe(get_gpt_json, max_attempts, quevector=i, transcript=transcript, key=key,
                                        max_tokens=max_tokens)

    return success, quevectors


def get_gpt_json(quevector, transcript, gptkey, max_tokens):
    system = ("Format your entire answer as a JSON object, with an entry named \"answer\" containing "
              "your answer and an entry \"able\" containing a binary value (true or false, "
              "all lower case) for whether you were actually able to answer the "
              "question. Base your answer strictly on information contained in the prompt, "
              "without speculating. "
              "The answer should be a single running text string, not a list or dictionary."
    )
    prompt = quevector['question'] + " Answer based on the following transcript and a supplemental file. Transcript: " \
             + transcript + "\nEnd of Transcript. Supplemental file:\n" + quevector['pertinent']['content']
    answer = cwm.askgptjson(prompt, system, gptkey, max_tokens)
    answer = answer.replace("```json\n", "").replace("\n```", "").replace("```python\n", "")
    jsonobject = json.loads(answer)
    print(type(jsonobject))
    able = jsonobject['able']
    if not type(able) == bool:
        raise Exception("Able is not a Boolean.")
    answertext = jsonobject['answer']
    return {'able': able, 'answertext': answertext}



def get_embedding(text, model="text-embedding-3-small"):
    client = OpenAI(api_key='<YOUR API KEY>')
    text = text.replace("\n", " ")
    max_chars = 8000 * 4
    if len(text) > max_chars:
        text = text[:max_chars]
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def vectorize(transcript, supplementary, questions, vector_size):
    # define a list of documents.
    questionsplain = []
    for i in questions:
        questionsplain.append(i)
    data = [transcript] + questionsplain + \
        [d.get('content') for d in supplementary]

    # preproces the documents, and create TaggedDocuments
    tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()),
                                  tags=[str(i)]) for i,
                   doc in enumerate(data)]


    # get the document vectors
    quevectors = []
    docvectors = []
    for q in questions:
        # vector = model.infer_vector(word_tokenize(q.lower()))
        vector = get_embedding(q)
        entry = {'question': q,
                 'vector': vector}
        quevectors.append(entry)
    for d in supplementary:
        content = d.get('content')
        # vector = model.infer_vector(word_tokenize(content.lower()))
        vector = get_embedding(content)
        entry = {'document': d['filename'],
                 'content': content, 'vector': vector}
        docvectors.append(entry)

    return quevectors, docvectors


def evaluate(transcript, summary, gptkey, max_tokens):
    complex_prompt = cwm.build_prompt(transcript, summary, cwm.eval_criteria)
    answer = cwm.askgptcomplex(complex_prompt, gptkey, max_tokens)
    return answer


def failsafe(func, max_attempts, **kwargs):
    success = False
    i = 0
    answer = []
    while not success and i < max_attempts:
        try:
            answer = func(**kwargs)
            success = True
        except Exception as e:
            i += 1
            logging.error("Failure: %s", str(func) + str(e))
    return success, answer


def justtesting(file, sourcedirectory, sumtargetdirectory, evaltargetdirectory, llamakey, gptkey, max_tokens, max_attempts, vector_size):
    print(readfile(file, sourcedirectory, sumtargetdirectory))


def readfile(file, sourcedirectory, sumtargetdirectory):
    path = os.path.join(os.getcwd(), sourcedirectory, file)
    with open(path, encoding="utf8") as f:
        jsondict = json.load(f)

    with open(os.path.join(os.getcwd(), sumtargetdirectory, file[:-5] + '_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(jsondict, f, ensure_ascii=False, indent=4)


def runthrough(file, sourcedirectory, sumtargetdirectory, evaltargetdirectory, llamakey, gptkey, max_tokens, max_attempts, vector_size, only_general_summary=False):
    output, overallsuccess = generate_summaries(file, sourcedirectory, llamakey, gptkey, vector_size=vector_size,
                                                max_attempts=max_attempts, max_tokens=max_tokens, only_general_summary=only_general_summary)

    # for i in output['Meeting Participants']:
    #     for j in i['questions']:
    #         j['pertinent']['vector'] = str(j['pertinent']['vector'])
    #         j['vector'] = str(j['vector'])

    for i in output['Top k Documents']:
        i['question'] = '<transcript>'
        i['vector'] = str(i['vector'])
        for j in i['pertinent']:
            j['vector']['vector'] = str(j['vector']['vector'])
            j['distance'] = str(j['distance'])
    print()
    with open(os.path.join(os.getcwd(), sumtargetdirectory, file[:-5] + '_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    if not overallsuccess:
        return "Failure"



if __name__ == '__main__':
    nltk.download('punkt')
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
    # where you have your JSON files containing all the documents
    sourcedirectory = 'code/ami_collection'
    # where to save to after the summary step
    sumtargetdirectory = 'code/simple_RAG_output'
    # where to save to after the evaluation step
    evaltargetdirectory = 'code/eval_output'
    runthrough("ES2002c.json", sourcedirectory, sumtargetdirectory, evaltargetdirectory,
               llamakey, gptkey, max_tokens, max_attempts, vector_size, only_general_summary=True)
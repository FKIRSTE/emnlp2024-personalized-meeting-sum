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
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Mute the logging for the specific package
logging.getLogger('gensim').setLevel(logging.WARNING)

def generate_summaries(item, sourcedirectory, llamakey, gptkey, vector_size, max_attempts, max_tokens, only_general_summary = False):

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


    #logging.info("********** Identifying Participants **********")
    meeting_participants = []
    success = True
    if not only_general_summary:
        #success, meeting_participants = failsafe(get_participants, max_attempts, transcript=transcript, gptkey=gptkey, max_tokens=max_tokens)
        success = True
        meeting_participants = ['User Experience', 'Marketing', 'Project Manager', 'Industrial Design']

        personas = {}
        for participant in meeting_participants:
            success_personas, detailed_participants = failsafe(get_participant_personas, max_attempts, transcript=transcript, gptkey=gptkey, max_tokens=4000, participant=participant)
            personas[participant] = detailed_participants

    logging.info("Meeting Participants: %s", meeting_participants)

    if not success:
        output['Meeting Participants'] = []
        return output, success

    output['Meeting Participants'] = meeting_participants

    if success_personas:
        output['Personas'] = personas

    overallsuccess = True


    logging.info("********** Questions **********")
    total_questions = []
    for i in meeting_participants:
        persona = output['Personas'][i] if i in output['Personas'] else None
        success, questions_extract = failsafe(get_questions, max_attempts, transcript=transcript, gptkey=gptkey, max_tokens=4000, participant=i, persona=persona)
        entry = {'role': i, 'questions': questions_extract}
        total_questions.append(entry)
        if not success:
            overallsuccess = False

    output['Meeting Participants'] = total_questions

    if not overallsuccess:
        return output, overallsuccess


    logging.info("********** Answering questions **********")
    for i in total_questions:
        # print(i)
        logging.info(">>> Answering %s's questions <<<", i['role'])
        persona = output['Personas'][i['role']] if i['role'] in output['Personas'] else ""
        print(f">>>>>                             {persona}  <<", flush=True)
        success, answer = answer_questions(i['questions'], transcript, supplementary, vector_size, gptkey, max_attempts,
                                           max_tokens, persona=persona)
        if not success:
            overallsuccess = False
        i['questions'] = answer

    output['Meeting Participants'] = total_questions

    if not overallsuccess:
        return output, overallsuccess


    logging.info("********** Summarizing **********")
    logging.info("Total questions: %s", len(total_questions))
    for i in total_questions:
        role = i['role']
        logging.info(">>> %s <<<", role)
        answers = []
        for j in i['questions']:
            if j['answer']['able']:
                answers.append(j)
        modified_transcript = get_modified_transcript(output['Transcript'], answers)
        i['Modified Transcript'] = modified_transcript

        summaries = []
        gptentry = {'summarizer': 'gpt-4-turbo'}
        persona = output['Personas'][role] if role in output['Personas'] else ""

        summary_prompt_1 = (
            "You are a professional summarizer and have been tasked with creating an abstractive summary for a participant in a meeting. "
            "Your summary should be 250 tokens or less."
            "Carefully analyze the following transcript and provide a detailed summary for the participant. "
        )
        summary_prompt_2 = (
            f"Consider the target persona who will have to work with the summary: <{persona}>."
            "The generated summary should help the persona to understand the meeting content even after a long time and the summary should be the perfect source for the persona to post-process the meeting content and prepare for the next steps."
            #"Therefore, tailor your summary to the participant's role, personality traits, point of views, contributions, knowledge that the brought to the meeting, information that they did not know, and any other relevant information. "
            "Focus on what is relevant for the participant to know and add what the participant needs to know to best work with the meeting content."
        )

        summary_prompt = summary_prompt_1 + summary_prompt_2


        gptentry['summary'] = cwm.askgptsystem("Summarize this transcript. Create an abstractive summary. Make the summary 250 tokens or less." + modified_transcript, summary_prompt, gptkey, 250)
        summaries.append(gptentry)
        i['automated_summary'] = summaries

    output['Meeting Participants'] = total_questions

    # output['automated_summary'] = {'summary': cwm.askgpt("Summarize this transcript. " + transcript, gptkey, max_tokens)}

    return output, overallsuccess


def get_modified_transcript(transcript, questions):
    for i in questions:
        length = len(i['insert'])
        start = transcript.find(i['insert'])
        end = start + length
        transcript = transcript[:end] + ' [' + i['answer']['answertext'] + ']' + transcript[end:]
    return transcript


def get_participant_personas(transcript, participant, gptkey, max_tokens):
    logging.info("********** %s persona **********", participant)
    system_prompt = (
        "You are a professional profiler and have been tasked with creating a persona for a participant in a meeting. "
        "Carefully analyze the following transcript and provide a detailed persona for the participant. "
        "In your answer, include the participant's role, personality traits from the Big Five, point of views, contributions, knowledge that the brought to the meeting, information that they did not know, and any other relevant information. "
        "Make sure to provide a detailed and comprehensive persona."
        "Your answer should be a string containing a running text."
    )
    user_prompt = f"Create a persona for participant '{participant}' based on the following transcript: {transcript}"
    persona = cwm.askgptsystem(user_prompt, system_prompt, gptkey, max_tokens)
    print(persona)
    return persona


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
    meeting_participants = output_list #ast.literal_eval(participants_answer)

    return meeting_participants


def get_questions(participant, transcript, gptkey, max_tokens, persona = ""):
    logging.info("********** Questions for %s **********", participant)

    system_1 = (
        "For the following task, respond in a way that matches this description: "
        f"<{persona}> \n"
    ) if persona else ""
    system_2 = (
        "Take the role of a question generator that takes the role of a defined participant and points out unclarities and open questions in a transcriot. Generate at most 5 questions. Only ask the 5 most relevant questions."
    )
    system = system_1 + system_2

    user_p1 = "Considering an arbitrary participants," if (participant == 'General Summary') else f"If you were participant '{participant}',"

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

    list_of_dicts = ast.literal_eval(open_questions)
    outputlist = []
    count_true = 0
    for i in list_of_dicts:
        if i['insert'] in transcript:
            count_true += 1
            outputlist.append(i)
    logging.info("Number of matching insertions: %s / 5", count_true)
    return outputlist


def answer_questions(questions, transcript, supplementary, vector_size, gptkey, max_attempts, max_tokens, persona):
    supplementary_data = []

    for i in supplementary:
        if 'sum.' not in i['filename'].lower() and 'final.report.doc' not in i['filename'].lower():
            supplementary_data.append(i)

    quevectors, docvectors = vectorize(transcript, supplementary_data, questions, vector_size)

    for i in quevectors:
        distances = []
        for j in range(len(docvectors)):
            dist = numpy.linalg.norm(docvectors[j]['vector'] - i['vector'])
            distances.append({'distance': dist, 'vector': docvectors[j]})
        dist = min(distances, key=lambda x: x['distance'])
        i['pertinent'] = dist['vector']

    for i in quevectors:
        success, i['answer'] = failsafe(get_gpt_json, max_attempts, quevector=i, transcript=transcript, gptkey=gptkey, max_tokens=max_tokens, persona=persona)

    return success, quevectors


def get_gpt_json(quevector, transcript, gptkey, max_tokens, persona):
    system = ("Format your entire answer as a JSON object, with an entry named \"answer\" containing "
              "your answer and an entry \"able\" containing a binary value (true or false, all lower case) for whether you were actually able to answer the "
              "question. Base your answer strictly on information contained in the prompt, "
              "without speculating. "
              f"Tailor your answer so it fits best to this persona: < {persona} >. "
              "The answer should be a single running text string, not a list or dictionary."
              )
    prompt = quevector['question'] + " Answer based on the following transcript and a supplemental file. Transcript: " \
             + transcript + "\nEnd of Transcript. Supplemental file:\n" + quevector['pertinent']['content']
    answer = cwm.askgptjson(prompt, system, gptkey, max_tokens)
    answer = answer.replace("```json\n", "").replace("\n```", "").replace("```python\n", "")
    jsonobject = json.loads(answer)
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


def runthrough(file, sourcedirectory, sumtargetdirectory, evaltargetdirectory, llamakey, gptkey, max_tokens, max_attempts, vector_size, only_general_summary = False):
    output, overallsuccess = generate_summaries(file, sourcedirectory, llamakey, gptkey, vector_size=vector_size, max_attempts=max_attempts, max_tokens=max_tokens, only_general_summary = only_general_summary)

    for i in output['Meeting Participants']:
        for j in i['questions']:
            j['pertinent']['vector'] = str(j['pertinent']['vector'])
            j['vector'] = str(j['vector'])
    with open(os.path.join(os.getcwd(), sumtargetdirectory, file[:-5] + '_summary_TIME.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    if not overallsuccess:
        return "Failure"


if __name__ == '__main__':
    # nltk.download('punkt')

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
    sumtargetdirectory = 'code/personal_sum_output'  # where to save to after the summary step
    evaltargetdirectory = 'code/eval_output'  # where to save to after the evaluation step
    time_start = time.time()
    runthrough("ES2002c.json", sourcedirectory, sumtargetdirectory, evaltargetdirectory, llamakey, gptkey, max_tokens, max_attempts, vector_size, only_general_summary = False)
    time_end = time.time()
    print("Time elapsed: ", time_end - time_start)
from model_utils import get_answer, get_answer_roberta, get_answer_longformer
import json
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import difflib

def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    answers_text = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
                    answers_text.append(answer['text'])
    return contexts, questions, answers, answers_text

val_contexts, val_questions, val_answers, val_ans_text = read_squad('datasets/2_dbert/dev.json')

df_validation = pd.DataFrame()
df_validation['context'] = val_contexts
df_validation['question'] = val_questions
df_validation['answers'] = val_ans_text

results = {'match': 0, 'count': 0, 'overlap': 0, 'answers_len': 0}

for ind, example in tqdm(df_validation.iterrows()):
    output = get_answer(example['context'], example['question'])
    results['overlap'] += len(get_overlap(output, example['answers']))
    results['answers_len'] += len(example['answers'])
    results['match'] += int(output in example['answers'])
    results['count'] += 1

print(f"Correct examples: {results['match']}/{results['count']}")
print(f"Overlap ratio: {results['overlap']*100/ results['answers_len']} %")
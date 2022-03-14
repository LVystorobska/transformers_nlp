from model_utils import get_answer, get_answer_roberta
import json
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

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

results = {'match': 0, 'count': 0}

for ind, example in tqdm(df_validation.iterrows()):
    output = get_answer_roberta(example['context'], example['question'])

    results['match'] += int(output in example['answers'])
    results['count'] += 1

print(f"Correct examples: {results['match']}/{results['count']}")
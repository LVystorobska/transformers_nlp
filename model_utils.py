from transformers import QuestionAnsweringPipeline, DistilBertTokenizerFast, DistilBertForQuestionAnswering, AutoModelForQuestionAnswering, AutoTokenizer, pipeline


tokenizer = DistilBertTokenizerFast.from_pretrained('fine_tune_BERT/')
model = DistilBertForQuestionAnswering.from_pretrained('fine_tune_BERT/')
q_a_pipeline =  QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
model_name = 'deepset/roberta-base-squad2'
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
q_a_roberta =  QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)


question = 'What is teaching not considered due to stress?'
text = 'A 2000 study found that 42% of UK teachers experienced occupational stress, twice the figure for the average profession. A 2012 study found that teachers experienced double the rate of anxiety, depression, and stress than average workers.'

def get_answer(text, question):
    res = q_a_pipeline({'question': question, 'context': text})
    return res['answer']


def get_answer_roberta(text, question):
    res = q_a_roberta({'question': question, 'context': text})
    return res['answer']

# print(get_answer_roberta(text, question))
from transformers import QuestionAnsweringPipeline, DistilBertTokenizerFast, DistilBertForQuestionAnswering, AutoModelForQuestionAnswering, AutoTokenizer, pipeline, LongformerTokenizerFast, LongformerForQuestionAnswering

question = 'What is teaching not considered due to stress?'
text = 'A 2000 study found that 42% of UK teachers experienced occupational stress, twice the figure for the average profession. A 2012 study found that teachers experienced double the rate of anxiety, depression, and stress than average workers.'

def get_answer(text, question):
    tokenizer = DistilBertTokenizerFast.from_pretrained('fine_tune_BERT/')
    model = DistilBertForQuestionAnswering.from_pretrained('fine_tune_BERT/')
    q_a_pipeline =  QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
    res = q_a_pipeline({'question': question, 'context': text})
    return res['answer']


def get_answer_roberta(text, question):
    model_name = 'deepset/roberta-base-squad2'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    q_a_roberta =  QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
    res = q_a_roberta({'question': question, 'context': text})
    return res['answer']

def get_answer_longformer(text, question):
    tokenizer_trivia = LongformerTokenizerFast.from_pretrained("finetuned-trivia/")
    model_trivia = LongformerForQuestionAnswering.from_pretrained("finetuned-trivia/")
    q_a_longformer =  QuestionAnsweringPipeline(model=model_trivia, tokenizer=tokenizer_trivia)
    res = q_a_longformer({'question': question, 'context': text})
    return res['answer']

# print(get_answer_longformer(text, question))
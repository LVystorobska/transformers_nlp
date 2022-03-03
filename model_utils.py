from transformers import QuestionAnsweringPipeline, DistilBertTokenizerFast, DistilBertForQuestionAnswering


tokenizer = DistilBertTokenizerFast.from_pretrained("fine_tuned_BERT/")
model = DistilBertForQuestionAnswering.from_pretrained("fine_tuned_BERT/")
q_a_pipeline =  QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)


# question = 'What is teaching not considered due to stress?'
# text = "A 2000 study found that 42% of UK teachers experienced occupational stress, twice the figure for the average profession. A 2012 study found that teachers experienced double the rate of anxiety, depression, and stress than average workers."

def get_answer(text, question):
    res = q_a_pipeline({'question': question, 'context': text})
    return res['answer']

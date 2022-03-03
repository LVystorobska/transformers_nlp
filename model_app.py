from flask import Blueprint, Flask, request, render_template
from model_utils import get_answer
import os

bp = Blueprint('routes', __name__, url_prefix='/')

@bp.route('/')
def start_page():
    return render_template('home.html', start_page='True')

@bp.route('/custom_task')
def custom_task():
    return render_template('text_input_form.html', task_name='Question answering - Transformers')

@bp.route('/text_process', methods=['POST'])
def text_process():
    text = request.values['text']
    question = request.values['question']
    prediction = get_answer(text, question)
    return render_template('text_process_output.html', task_name='Answer prediction Output', class_p_=str(prediction))

class Config(object):
    SECRET_KEY = 'YOUR_SECRET_KEY'
    DATABASE = os.path.join('instance', 'project.sqlite')
    FILES_DIR = 'add_files'

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config())
    os.makedirs(app.instance_path, exist_ok=True)
    app.register_blueprint(bp)
    print('SERVER READY')
    return app

app = create_app()


if __name__ == '__main__':
    app.run()



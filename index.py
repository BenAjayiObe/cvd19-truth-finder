import os
import json
import torch
from flask import Flask, render_template, request

from drqa import pipeline

from loguru import logger


# starting app
app = Flask(__name__)

# configuring directory and environment variables
drqa_data_directory = 'data/covid19/'

config = {
    'reader-model': os.path.join(drqa_data_directory, 'model', '20200509-c9d21e14.mdl'),
    'retriever-model': os.path.join(drqa_data_directory, 'doc_ranker', 'doc-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'),
    'doc-db': os.path.join(drqa_data_directory, 'doc-db', 'doc.db'),
    'embedding-file': os.path.join(drqa_data_directory, 'glove.6B.50d.txt'),
    'tokenizer': 'spacy',
    'no-cuda': True,
    'gpu': 0
}

cuda = torch.cuda.is_available() and not config.get('no-cuda', False)
if cuda:
    torch.cuda.set_device(config.get('gpu', 0))
    logger.info('CUDA enabled (GPU %d)' % config.get('gpu', 0))
else:
    logger.info('Running on CPU only.')


logger.info('Initializing pipeline...')
drqa_pipeline_instance = pipeline.DrQA(
    cuda=cuda,
    reader_model=config['reader-model'],
    ranker_config={'options': {'tfidf_path': config['retriever-model']}},
    db_config={'options': {'db_path': config['doc-db']}},
    tokenizer=config['tokenizer'],
    embedding_file=config['embedding-file'],
)


def process_query(question, candidates=None, top_n=1, n_docs=5):
    predictions = drqa_pipeline_instance.process(
        question, candidates, top_n, n_docs, return_context=True
    )
    answers = []
    for i, p in enumerate(predictions, 1):
        answers.append({
            'index': i,
            'span': p['span'],
            'doc_id': p['doc_id'],
            'span_score': '%.5g' % p['span_score'],
            'doc_score': '%.5g' % p['doc_score'],
            'text': p['context']['text'],
            'start': p['context']['start'],
            'end': p['context']['end']
        })
    return answers


@logger.catch
@app.route("/")
def index():
    return render_template('index.html')


@logger.catch
@app.route("/query", methods=["POST"])
def query():
    query_text = request.form['query']
    answers = process_query(question=query_text, top_n=5)
    return render_template('results.html', query=query_text, **answers[0])

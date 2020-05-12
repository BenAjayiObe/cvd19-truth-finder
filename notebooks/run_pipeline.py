import torch
import argparse
import code
import prettytable

from termcolor import colored
from drqa import pipeline
from drqa.retriever import utils

# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------


def process(question, candidates=None, top_n=1, n_docs=5):
    predictions = DrQA.process(
        question, candidates, top_n, n_docs, return_context=True
    )
    table = prettytable.PrettyTable(
        ['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score']
    )
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p['span'], p['doc_id'],
                       '%.5g' % p['span_score'],
                       '%.5g' % p['doc_score']])
    print('Top Predictions:')
    print(table)
    print('\nContexts:')
    for p in predictions:
        text = p['context']['text']
        start = p['context']['start']
        end = p['context']['end']
        output = (text[:start] +
                  colored(text[start: end], 'green', attrs=['bold']) +
                  text[end:])
        print('[ Doc = %s ]' % p['doc_id'])
        print(output + '\n')

if __name__ == '__main__':
    # Arguments
    # args_reader_model="/home/ubuntu/cvd19-truth-finder/data/covid19/model/20200509-c9d21e14.mdl"
    # args_retriever_model="/home/ubuntu/cvd19-truth-finder/data/covid19/doc_ranker/doc-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"
    # args_doc_db="/home/ubuntu/cvd19-truth-finder/data/covid19/doc-db/doc.db"
    # args_tokenizer="spacy"
    # args_candidate_file=None
    # args_no_cuda=True
    # args_gpu=-1

    args_reader_model="../data/covid19/model/20200509-c9d21e14.mdl"
    args_retriever_model="../data/covid19/doc_ranker/doc-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"
    args_doc_db=".. /data/covid19/doc-db/doc.db"
    args_tokenizer="spacy"
    args_candidate_file=None
    args_no_cuda=True
    args_gpu=-1

    args_cuda = not args_no_cuda and torch.cuda.is_available()
    if args_cuda:
        torch.cuda.set_device(args_gpu)

    if args_candidate_file:
        candidates = set()
        with open(args_candidate_file) as f:
            for line in f:
                line = utils.normalize(line.strip()).lower()
                candidates.add(line)
    else:
        candidates = None

    DrQA = pipeline.DrQA(
        cuda=args_cuda,
        fixed_candidates=candidates,
        reader_model=args_reader_model,
        ranker_config={'options': {'tfidf_path': args_retriever_model}},
        db_config={'options': {'db_path': args_doc_db}},
        tokenizer=args_tokenizer
    )

    process("How many people are infected?", top_n=2)
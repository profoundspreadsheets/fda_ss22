#!/usr/bin/env python3

from cgitb import text
import pprint # pretty printer
import logging
from sklearn.datasets import fetch_20newsgroups
from fda_helper import preprocess_data
from gensim import corpora, models, similarities
import numpy as np

np.random.seed(420)

# enable logging to display what is happening
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# read dataset 20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

texts = preprocess_data(documents)
dictionary = corpora.Dictionary(texts)

bow_corpus = [dictionary.doc2bow(text) for text in texts] # bow = Bag Of Words
#pprint.pprint(bow_corpus[5][1]) # one example document, words maped to ids

tfidf = models.TfidfModel(bow_corpus) # train tf-idf model
corpus_tfidf = tfidf[bow_corpus] # apply transformation on the whole corpus

##  TODO: transform your tfidf model into a LSI Model
##  using python gensim, use num_topics=200

# Source: https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html
model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)

## TODO: query! pick a random document and formulate a query based on the
## terms in the document.
#randomDoc = np.random.choice(bow_corpus)
randomDoc = bow_corpus[69]
pprint.pprint(f">>>>> {randomDoc}")
randomDocLSI = model[randomDoc]

## TODO: initialize a query structure for your LSI space
index = similarities.MatrixSimilarity(model[corpus_tfidf])
index.save('test.index')

## TODO: perform the query on the LSI space, interpret the result and summarize your findings in the report
sims = index[randomDocLSI]

sims = sorted(enumerate(sims), key=lambda item: item[1])
for doc_position, doc_score in sims:
    print(doc_score, documents[doc_position])
#! /usr/bin/env python3

from time import time

# imports for classification
from make_model import *
from classify import *

# imports for load_sim_db()
import sqlite3
from io import StringIO

# imports for load_w2v()
from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer


def load_sim_db():
    print("Loading 'SEWordSim-r1.db' into ram")

    # Next two sections from Stack-Overflow:
    # https://stackoverflow.com/questions/3850022/how-to-load-existing-db-file-to-memory-in-python-sqlite3/53253110
    temp_db = sqlite3.connect('SEWordSim-r1.db')
    temp_file: StringIO = StringIO()
    # noinspection PyTypeChecker
    for line in temp_db.iterdump():
        temp_file.write('%s\n' % line)
    temp_db.close()
    temp_file.seek(0)

    print("Transferring data into in memory database")

    similarity_database = sqlite3.connect(":memory:")
    sim_db_conn = similarity_database.cursor()
    sim_db_conn.executescript(temp_file.read())
    similarity_database.commit()
    similarity_database.row_factory = sqlite3.Row

    return SimDB(sim_db_conn)


def load_w2v():
    print("Loading gensim pre-trained model")
    # model = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)
    # Above is intolerably slow and large, normed by code found here: https://stackoverflow.com/a/56963501
    model = KeyedVectors.load("SO_vectors_normed", mmap='r')

    # Use this to load the provided AnnoyIndex
    annoy_index = AnnoyIndexer()
    annoy_index.load('SO_vectors_normed_annoy_index')

    # Use this to generate a new AnnoyIndex in ram, number is n-gram size (2 is recommended and seems to work best here)
    # annoy_index = AnnoyIndexer(model, 3)

    return Word2Vec(model, index=annoy_index)


# List of valid models, data loading functions above correspond in order (RAW loads no additional data)
valid_models = {'WordSimSEDB': (load_sim_db, True),
                'Word2VecSE': (load_w2v, False),
                'Raw': (lambda: None, True)}

if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[2] not in valid_models.keys() or sys.argv[3] not in ['True', 'False']:
        print('Usage: ./categorize labeledData.csv [Raw | WordSimSEDB | Word2VecSE] [True | False]')
        exit(1)

    # Segmentation data must always be loaded, as make_model depends upon it
    print("Loading wordsegment data")
    wordsegment.load()

    # Enable Segmentation depending on option
    segment = (sys.argv[3] == "True")

    # Retrieve desired configuration data
    loading_func, stemmed_database = valid_models[sys.argv[2]]
    my_sim_func = loading_func()

    print("Beginning experiments\n")

    print('Accuracy percentage, ', end='')
    print('Zeros percentage, ', end='')
    print('Average score, ', end='')
    print('Filter Threshold %, ', end='')
    print('Num similar terms, ', end='')
    print('Min similarity %, ', end='')
    print('Sec/Document, ', end='')
    print()

    # Here put loops that depend on regenerating model
    # for filter_threshold in range(0, 101, 10):
    filter_threshold = 0.0

    # TODO: Make this load trainData
    with open(sys.argv[1], 'r') as fi:
        labeled_data = csv.DictReader(fi, delimiter=',')

        (term_frequencies, doc_frequencies) = generate_frequencies(labeled_data,
                                                                   filter_threshold=(filter_threshold * 0.0001))
        my_idF = TFidF(term_frequencies, doc_frequencies)

    # Here put loops that don't depend on regenerating model
    # for num_similar in range(1, 2):
    for num_similar in range(1, 11, 1):
        # for min_similarity in range(0, 1):
        for min_similarity in range(0, 101, 10):

            # TODO: Make this load testData
            with open(sys.argv[1], 'r') as fi:
                labeled_data = csv.DictReader(fi, delimiter=',')

                # Loop data, counts of total score and num_docs for average score, num_match and num_docs for accuracy
                # num_zero for tracking number of documents that failed to match against anything at all
                sum_score = 0
                num_match = 0
                num_docs = 0
                num_zero = 0

                start = time()

                for doc in labeled_data:
                    expected_cat = doc["Category"].lower()  # some of the labels are inconsistent in case
                    new_message = doc["message"].lower()

                    category = classify(my_idF,
                                        new_message,
                                        sim_func=my_sim_func,
                                        num_similar=num_similar,
                                        min_similarity=0.01*min_similarity,
                                        stemmed_database=stemmed_database,
                                        segment=segment)

                    if category[0] == expected_cat:
                        num_match += 1
                    sum_score += category[1]
                    if category[1] == 0:
                        num_zero += 1
                    num_docs += 1

                end = time()

                # Main information
                match_percent = 100 * num_match / num_docs
                zeros_percent = 100 * num_zero / num_docs
                average = sum_score / num_docs
                sec_per_doc = (end-start)/num_docs

                print(match_percent, zeros_percent, average, filter_threshold/100,
                      num_similar, min_similarity, sec_per_doc, sep=", ")

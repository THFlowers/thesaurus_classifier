#! /usr/bin/env python3

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

    annoy_index = AnnoyIndexer()
    annoy_index.load('SO_vectors_normed_annoy_index')

    # To generate a new AnnoyIndex in ram, number is n-gram size (2 is recommended and seems to work best here)
    # annoy_index = AnnoyIndexer(model, 3)

    return Word2Vec(model, index=annoy_index)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: ./categorize labeledData.csv')
        exit(1)

    print("Loading wordsegment data")
    wordsegment.load()

    # my_sim_func = load_sim_db()
    # stemmed_database = True

    my_sim_func = load_w2v()
    stemmed_database = False

    # my_sim_func = None
    # stemmed_database = True

    print("Beginning experiments\n")

    print('Match percentage, ', end='')
    print('Zeros percentage, ', end='')
    print('Average score, ', end='')
    print('Filter Threshold %, ', end='')
    print('Num similar terms, ', end='')
    print('Min similarity %, ', end='')
    print()

    # TODO: Here put loops that depend on regenerating model
    # for filter_threshold in range(0, 101, 10):
    filter_threshold = 0.0

    # TODO: Make this load trainData
    with open(sys.argv[1], 'r') as fi:
        labeled_data = csv.DictReader(fi, delimiter=',')

        (term_frequencies, doc_frequencies) = generate_frequencies(labeled_data,
                                                                   filter_threshold=(filter_threshold * 0.0001))
        my_idF = TFidF(term_frequencies, doc_frequencies)

    # TODO: Here put loops that don't depend on regenerating model
    #for num_similar in range(1, 2):
    for num_similar in range(1, 11, 1):
        #for min_similarity in range(0, 1):
        for min_similarity in range(70, 91, 1):

            # TODO: Make this load testData
            with open(sys.argv[1], 'r') as fi:
                labeled_data = csv.DictReader(fi, delimiter=',')

                # Loop data, counts of total score and num_docs for average score, num_match and num_docs for accuracy
                # num_zero for tracking number of documents that failed to match against anything at all
                sum_score = 0
                num_match = 0
                num_docs = 0
                num_zero = 0

                for doc in labeled_data:
                    expected_cat = doc["Category"].lower()  # some of the labels are inconsistent in case
                    new_message = doc["message"].lower()

                    category = classify(my_idF,
                                        new_message,
                                        sim_func=my_sim_func,
                                        num_similar=num_similar,
                                        min_similarity=0.01*min_similarity,
                                        stemmed_database=stemmed_database)

                    if category[0] == expected_cat:
                        num_match += 1
                    sum_score += category[1]
                    if category[1] == 0:
                        num_zero += 1
                    num_docs += 1

                # Main information
                match_percent = 100 * num_match / num_docs
                zeros_percent = 100 * num_zero / num_docs
                average = sum_score / num_docs

                print(match_percent, zeros_percent, average,
                      filter_threshold/100, num_similar, min_similarity, sep=", ")

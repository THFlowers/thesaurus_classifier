#!/usr/bin/env python3

# NLP Based Model -- normalize first, PorterStemmer, work_tokenize
# Produces the model from the provided labeled data (labeledData.csv)
#
# Input: labeledData.csv, csv file that must have columns labeled "message" and "Category"
# Output: term/doc_frequency.json files
#
# Note: Normalizes case of category names
# We produce for each category 1) number of docs (messages) in that category, 2) Counter of the words
#
# These words are filtered/normalized by 1) Tokenizing, 2) Removing stop words, 3) stemming,
# 4) removing words that appear less than FREQUENCY_THRESHOLD percentage of times in a category
#
# Looking at words output from phase 4) shows this probably won't work
# Number of words per category are very low at 10% and not really relevant at 3%
# Determination: We need *lots* more data
#
# Possible solutions:
# 1) Crowd source, have Knowledgeable individuals Receive samples & drop down list of cats. Choose most popular cat
# 2) Determine a heuristic and mine from Publicly available records (more gitter channels?)
# 3) Attenuate: Label the words in SEWordSim by category, ignore mining
#
# Another problem, presence of code (esp. urls) and punctuation. Due to nature of this field such characters may be
# important, however they appear everywhere and have little semantic info
# Urls are destroyed/mutilated by word_tokenize, punctuation/code mostly deleted by RegexpTokenizer
# (However this gives very clean output)
#
# Another minor problem, some code and terms in CamelCase need to be split (ie to "Camel" and "Case")
# Observation: Product names appear frequently (obvious but may be useful)

# NOTE: Run the following code before running this script (put in an install / init script!)
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

import sys
import csv
import json
from copy import deepcopy
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import PorterStemmer

import wordsegment


def generate_frequencies(labeled_data,  filter_threshold=0.03):
    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')
    categories = dict()  # dict(category_name, {num_docs : int, counts : Counter(words)})
    # word_tokenize = lambda x: RegexpTokenizer(r'\w+').tokenize(x)

    for doc in labeled_data:
        category = doc["Category"].lower()  # some of the labels are inconsistent in case
        # if category == 'uninformative':
        #    continue
        if category not in categories.keys():
            categories[category] = {'num_docs': 1, 'counts': Counter()}
        else:
            categories[category]['num_docs'] += 1

        # use word_tokenize to parse words, make unique, remove stopwords
        # leaves non word things like '?', and "`", in input
        # NOTE: 2/27/20 -- Found forgot to call lower here
        message = doc["message"].lower().strip()
        message = word_tokenize(message)

        segmented_message = []
        for wd in message:
            segmented_message.append(wd)
            segments = wordsegment.segment(wd)
            if len(segments) > 1:
                segmented_message.extend(segments)

        processed_message = [stemmer.stem(wd) for wd in segmented_message
                             if wd not in stop_words and
                             sum(map((lambda x: 1 if x[1].isalnum() else 0),
                                     enumerate(wd))) > 0]

        for wd in processed_message:
            categories[category]['counts'][wd] += 1

    term_freqs = deepcopy(categories)
    doc_freqs = Counter()

    for cat in categories:
        category = categories[cat]
        for wd in category['counts']:

            # calculate term frequency % (within a single category)
            # Note: can also do number of times word appears across all categories
            count = category['counts'][wd]
            freq = count / category['num_docs']
            if freq < filter_threshold:
                del term_freqs[cat]['counts'][wd]
            # else:
                # print(cat, " : ('", wd, "', ", freq, ")", sep='')

            # Increase document frequency (here doc refers to category)
            # each word should appear only once per category,
            # so this counts number of categories a word appears in

            doc_freqs[wd] += 1

    return term_freqs, doc_freqs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./make_model labeledData.csv")
        exit(1)

    # Need to load here so that library calls above work correctly
    wordsegment.load()

    with open(sys.argv[1]) as csv_file:
        term_frequencies, doc_frequencies = generate_frequencies(csv.DictReader(csv_file, delimiter=','))

    with open("term_frequencies.json", 'w') as fi:
        json.dump(term_frequencies, fi, indent=4, sort_keys=True)

    with open("doc_frequencies.json", 'w') as fi:
        json.dump(doc_frequencies, fi, indent=4, sort_keys=True)

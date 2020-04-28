# Thesaurus Classifier
Experiment making a simple tf-idf based classifier augmented by a thesaurus (semantic resource)

Experiment conducted and report written as part of requirements for Masters degree in Computer Science (project track) from Florida State University

Note: This was originally called "gitter_classifier" as the training data originally came from several gitter servers.  As this project has nothing to do with gitter itself, a more sensible name was chosen upon uploading here.

To run the experiments you will need the following libraries installed either on your system, a virtual environment, or via your IDE (be sure to install all their requirements as well):

* wordsegment 1.3.1 (https://pypi.org/project/wordsegment/)
* nltk 3.4.5
* gensim 3.8.1

You will also need to download the archives in the following drive folder and place them in your project directory:

* https://drive.google.com/open?id=1Z1WYt29MKuPAbOqLLbJ9cNk5sq5x2SAT

## Before running the scripts
* Install the required libraries
* Download the semantic resources
* Run the following from a python shell in the project directory to download NLTK datasets:

> import nltk  
> nltk.download('stopwords')  
> nltk.download('punkt')  

## How to run the scripts

### classify_test.py

Must be provided training data in a compatible CSV format, a semantic resource option, and True/False to enable segmentation. 
> ./classify_test.py labeledData.csv [Raw | WordSimSEDB | Word2VecSE] [True | False]

### make_model.py

Performs the “training” steps only using labeledData.csv. Outputs term_frequencies.json and doc_frequencies.json.

> ./make_model labeledData.csv

### classify.py

Performs a raw tf-idf classification using the above json files on a plain text file.

> ./classify term_frequencies.json doc_frequencies.json message.txt

# TODO
* Find better hosting solution
* Project goals and results summary on this page

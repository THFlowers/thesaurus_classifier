# thesaurus_classifier
Experiment making a simple tf-idf based classifier augmented by a thesaurus (semantic resource)

Experiment conducted and report written as part of requirements for Masters degree in Computer Science (project track) from Florida State University

Note: This was originally called "gitter_classifier" as the training data originally came from several gitter servers.  As this project has nothing to do with gitter itself, a more sensible name was chosen upon uploading here.

To run the experiments you will need the following libraries installed either on your system, a virtual environment, or via your IDE (be sure to install all their requirements as well):

* wordsegment 1.3.1 (https://pypi.org/project/wordsegment/)
* nltk 3.4.5
* gensim 3.8.1

You will also need to download the archieves in the following drive folder and place them in your project directory:

* https://drive.google.com/open?id=1Z1WYt29MKuPAbOqLLbJ9cNk5sq5x2SAT

# How to run the scripts

classify_test.py takes no parameters, edit the code to select desired semantic resource and experiment scan range for each parameter via the various for loops, then run it.

# TODO
* Write short guide on running the other scripts
* Find better hosting solution
* Include finished report
* Project goals and results summary on this page

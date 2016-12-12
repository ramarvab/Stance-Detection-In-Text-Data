# Stance-Detection-In-Text-Data
Predicting whether the tweet is Favored or Against using TF-IDF and Global Vector Models.


Running Instructions

Prerequisites:

1.	Python 2.7
2.	Pandas
3.	Numpy
4.	Sci-kitlearn(sklearn)
5.	Natural Language Processing Toolkit (nltk)
6.	Natural Language Processing Toolkit Data(words,punkt)
    To download the above package the following commands need to be executed from python shell
    Import nltk
    nltk.download(‘punkt’)
    nltk.download(‘words’)
7.	Download Global Vector prebuilt model from http://nlp.stanford.edu/data/glove.twitter.27B.zip


Running Instructions
1.	Unzip the folder downloaded from above url and place all the files in code folder provided with the submission.
2.	To run the program type the below command after navigating to code foler 
python biasDetectionMain.py --trainingfile training.txt --testfile test-gold.txt --choice g --glovefile glove.twitter.27B.25d.txt --evaluation k --classifier svc

	parameters need to be supplied :
a)	–trainingfile : path to training.txt file in code folder
b)	–testfile :path to test-gold.txt file code folder
c)	–glovefile: path to global vector file.  We can give file with 25 dimensions or 50 dimensions or 100 dimensions or 200 dimensions 
eg for 50 dimensions
--glovefile glove.twitter.27B.25d.txt

d)	–choice :
to run the model with glove file enter ‘g’ or to  run the model with tf idf enter t

e)	–evaluation : enter ‘k’ for k fold cross validation or ‘a’ for accuracy 
f)	–classifier : enter svc for support vector machines ,gbc for gradient boosting classifier, rfc for random forest classifier

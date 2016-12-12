import string
import re
import numpy as np
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class Vector:
    def __init__(self,global_vectors_dict=None,train_data =None,test_data =None):
        self.global_vectors_dict =global_vectors_dict
        self.test_data =test_data
        self.train_data = train_data

    def simplify(self,word):
        simplified_list = []
        temp = ''
        words_list = filter(None,re.split("([A-Z][^A-Z]*)",word))
        if len(words_list) == len(word):
            return word.lower()
        for i in range(len(words_list)):
            words_list[i] = words_list[i].lower()
            if len(words_list[i]) == 1:
                temp = temp + words_list[i]
                if temp in words.words() and len(temp) > 2:
                    simplified_list.append(temp)
                    temp = ''
            else:
                simplified_list.append(words_list[i])
        return simplified_list

    def vectorsum(self,modified_list):
        nonzerocount = 0
        dummyvector = np.zeros_like(self.global_vectors_dict["dummy"])
        for word in modified_list:
            if word in self.global_vectors_dict:
                vect = self.global_vectors_dict[word]
            else:
                vect =np.zeros_like(self.global_vectors_dict["dummy"])
            if vect.sum() != 0:
                dummyvector += vect
                nonzerocount += 1
        if nonzerocount:
            dummyvector = dummyvector/nonzerocount
        return dummyvector

    def tokenize_glove(self,data):
        text_list=[]
        target_list =[]
        text_vector =[]
        for index,row in data.iterrows():
            # Create a sentence using target and the text. Word vector will be formed from this.
            sample_text_row = str(row["Target"]) + " " + str(row["Tweet"])

            #delete punctuation
            modified_text_row = sample_text_row.translate(None, string.punctuation)
            words_list = word_tokenize(modified_text_row)
            join_words = ' '.join([i for i in words_list if i.isalpha()])

            #  stem the tokens from the string
            modified_list =[]
            words_list = word_tokenize(join_words)
            for word in words_list:
                modified_list += self.simplify(word)
            modified_text_row =''.join(modified_list)
            # All tweets from training data
            text_list.append(modified_text_row)
            #All stances from training data
            target_list.append(row["Stance"])
            text_vector.append(self.vectorsum(modified_list))
        return text_list, target_list, text_vector

    def tokenize_tfidf(self,data):
        text_list =[]
        target_list =[]
        stemmer = SnowballStemmer("english")
        for index,row in data.iterrows():
            # Create a sentence using target and the text. Word vector will be formed from this.
            sample_text_row = str(row["Target"]) + " " + str(row["Tweet"])
            #delete punctuation
            modified_text_row = sample_text_row.translate(None, string.punctuation)
            words_list = word_tokenize(modified_text_row)
            join_words = ' '.join([i for i in words_list if i.isalpha()])

            #  stem the tokens from the string
            modified_list =[]
            words_list = word_tokenize(join_words)
            for word in words_list:
                modified_list += self.simplify(word)
            modified_text_row =''.join(stemmer.stem(word)for word in modified_list)
            # All tweets from training data
            text_list.append(modified_text_row)
            #All stances from training data
            target_list.append(row["Stance"])
        return text_list,target_list


    def globalVector(self):
        # Remove punctuation from and tokenize the training tweets
        train_text_list,train_target_list,train_text_vector =self.tokenize_glove(self.train_data)
        # Remove punctuation from and tokenize the test tweets
        test_text_list,test_target_list,test_text_vector = self.tokenize_glove(self.test_data)

        train_text=np.asarray(train_text_vector)
        train_target = np.asarray(train_target_list)
        test_text = np.asarray(test_text_vector)
        test_target =np.asarray(test_target_list)
        return train_text,train_target,test_text,test_target



    def tfidf(self):

        # Remove punctuation from and tokenize the training tweets
        train_text_list,train_target_list = self.tokenize_tfidf(self.train_data)

        # Remove punctuation from and tokenize the test tweets
        test_text_list,test_target_list = self.tokenize_tfidf(self.test_data)

        # We vectorize the tweets into unigrams and bigrams after removing the stopwords
        vector_count = CountVectorizer(ngram_range = (1,2), stop_words="english")
        transform_tfidf = TfidfTransformer()

        # build a document term matrix of training data and then convert it to a tfidf matrix
        count_train = vector_count.fit_transform(train_text_list)
        count_train_tfidf = transform_tfidf.fit_transform(count_train)
        train_text = count_train_tfidf.toarray()
        train_target = np.asarray(train_target_list)
        # build a vector based on words from training data and get a document term matrix on test data
        # and then change it to a tfidf matrix
        count_test = vector_count.fit(train_text_list).transform(test_text_list)
        count_test_tfidf = transform_tfidf.fit_transform(count_test)
        test_text = count_test_tfidf.toarray()
        test_target = np.asarray(test_target_list)
        return train_text, train_target, test_text, test_target
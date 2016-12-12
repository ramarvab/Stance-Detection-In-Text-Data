import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sys
from WordsVector import Vector

class BiasDetection:
    def __init__(self,vectorchoice=None,evaluationmodel=None,classifier=None):
        self.global_vectors_dict ={}
        self.target_dict ={0:"AGAINST", 1:"FAVOR", 2:"NONE"}
        self.train_data = []
        self.test_data = []# dict of results for a branch, None for everything except endpoints
        self.vectorchoice =vectorchoice
        self.evaluationmodel = evaluationmodel
        self.classifier = classifier
        self.distinctTrainTargets =[]


    def read_globevec_data(self,glovefile):
        file = open(glovefile, 'r')
        for line in file.readlines():
            line = line.strip().split()
            word = line[0]
            vector = line[1:]
            self.global_vectors_dict[word] = np.array(vector, dtype=float)


    def predict_target(self,data):
        for key,val in self.target_dict.items():
            data.loc[data["Stance"]==val, "Stance"]=int(key)
        return data

    def runmodel(self,train,test,glove):
        if self.vectorchoice == 'g':
            print "loading glove data"
            self.read_globevec_data(glove)
            print "loding completed"
        self.train_data = pd.read_csv(train,sep ='\t',header =0)
        self.test_data =pd.read_csv(test,sep ='\t',header =0)

        self.distinctTrainTargets = self.train_data.Target.unique()
        trainModel = self.predict_target(self.train_data)
        testModel = self.predict_target(self.test_data)

        total_accuracy = 0
        for target in self.distinctTrainTargets:
            print "building model for target "+target+"..."
            if(self.vectorchoice=='g'):
                wordsvector = Vector(self.global_vectors_dict,trainModel[trainModel["Target"]==target],testModel[testModel["Target"]==target])
                train_features,train_target,test_features,test_target = wordsvector.globalVector()
            else:
                self.global_vectors_dict={}
                wordsvector = Vector(self.global_vectors_dict,trainModel[trainModel["Target"]==target],testModel[testModel["Target"]==target])
                train_features,train_target,test_features,test_target = wordsvector.tfidf()

            if (self.evaluationmodel =="a"):
                if self.classifier.lower() == "rfc":
                    modelfunction = RandomForestClassifier(n_estimators=70).fit(train_features,train_target)
                    accuracy  = modelfunction.score(test_features,test_target)
                    print "Accuracy for Random Forest Classifier for the"+target+ "-"+accuracy
                if self.classifier.lower() == "gbc":
                    modelfunction = GradientBoostingClassifier.fit(train_features,train_target)
                    accuracy = modelfunction.score(test_features,test_target)
                    print "Accuracy for Grdient Boost Classifier for the"+target+ "-"+accuracy
                if self.classifier.lower() == "svc":
                    modelfunction = SVC(kernel="rbf").fit(train_features,train_target)
                    accuracy =modelfunction.score(test_features,test_target)
                    print "Accuracy for SVC Classifier for the"+target+ "-"+accuracy
            else:
                if self.classifier.lower() =="rfc":
                    modelfunction = RandomForestClassifier(n_estimators=70)
                if self.classifier.lower() =="gbc":
                    modelfunction =GradientBoostingClassifier()
                if self.classifier.lower() =="svc":
                    modelfunction = SVC(kernel ="rbf")
                crossvalidation = cross_validation.ShuffleSplit(len(train_features),n_iter=1,test_size=0.1,random_state=0)
                crossvalidationaccuracy = list(cross_validation.cross_val_score(modelfunction,train_features,train_target,cv=crossvalidation))
                accuracy = float(sum(crossvalidationaccuracy))/len(crossvalidationaccuracy)
                print "K-fold accuracy for "+target+"-" ,accuracy
            total_accuracy += accuracy
        final_accuracy = total_accuracy/len(self.distinctTrainTargets)
        print "\n total accuracy of the classifier for all the targets is "+str(round(final_accuracy*100,2))+"%"+"\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--training', required=True, help='Filename for training set')
    parser.add_argument('-te', '--test', required=True, help='Filename for test set')
    parser.add_argument('-ch','--choice',required =True,help ='select choice for vectorization g for using glove,t for using tfidf')
    parser.add_argument('-gv','--glove',required=True,help ='Filename for glove data')
    parser.add_argument('-e','--evaluation',required = True, help ='evaluation method k for k-fold and a for accuracy ')
    parser.add_argument('-cl','--classifier',required = True,help ="select the classifier")
    args=parser.parse_args()
    if args.choice not in ["g","t"]:
        print "please select g for glove or t for tf-idf...exiting"
        sys.exit(0)
    if args.evaluation not in ['k', 'a']:
        print "please select k for k-fold or a for accuracy.. terminating..."
        sys.exit(0)
    if args.classifier not in ['svc','rfc','gbc']:
        print "please select svc for support vector classifier or rfc for random forest classifier or gbc for gradient boosting classifier"
    print "Thanks for your inputs ..we are analysing your data..."
    db = BiasDetection()
    db.vectorchoice=args.choice
    db.evaluationmodel=args.evaluation
    db.classifier=args.classifier
    db.runmodel(args.training,args.test,args.glove)

if __name__ == "__main__":
    main()
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from dpss import ssv, wo, dp, flex, agreg, union, parse_text, getWords, intersection, getWordsX

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = svm.LinearSVC(max_iter=10000)
clf5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
def blast():
    with open('MSRParaphraseCorpus/MSR_paraphrase_train.txt') as f:
        MSRtrain = f.readlines()
    train = []
    for each in MSRtrain:
        train.append(each.split('\t'))
    with open('testdata/features-output.txt') as f:
        mypredictions = f.readlines()
    predictions = []
    for each in mypredictions:
        predictions.append(getWordsX(each))
    Y = []
    for each in train:
        Y.append(int(each[0]))
    X = []
    for each in predictions:
        '''X.append([float(each[0]), float(each[1]), float(each[2]), float(each[3]), float(each[4]),
                  float(each[3])*float(each[3])*float(each[1]), float(each[3])*float(each[3]), float(each[5]), float(each[6]), float(each[7]),
                  float(each[8]), float(each[9]), float(each[10]), float(each[11]), float(each[12]), float(each[13]), float(each[14]), float(each[15]),
                  float(each[16]), float(each[17]), float(each[18])])'''
        X.append([float(each[3])*0.80 + float(each[1])*0.20,float(each[5]), float(each[6]), float(each[8]),float(each[9]),float(each[10]),float(each[11]),
        float(each[18]), float(each[1]), float(each[2])])
    #print X
    Xtrain = X[0:len(Y)]
    Xtest = X[len(Y):]
    print len(Xtrain), len(Y), len(Xtest)
    eclf1 = VotingClassifier(estimators=[('lr', clf1),('nn', clf5)], voting='hard')
    eclf1 = eclf1.fit(Xtrain, Y)
    pred_train = eclf1.predict(Xtrain)
    pred_test = eclf1.predict(Xtest)
    with open('data/test-pred-test-blast.txt', 'w') as f:
        for each in pred_test:
            f.write(str(each)+'\n')
    with open('data/test-pred-train-blast.txt', 'w') as f:
        for each in pred_train:
            f.write(str(each)+'\n')

def cross():
    print '-----test------'
    with open('data/test-pred-test-blast.txt') as f:
        zz = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_test.txt') as f:
        MSRtest = f.readlines()
    test = []
    for each in MSRtest:
        test.append(each.split('\t'))
    for i in range(len(zz)):
        zz[i] = int(zz[i])
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    j = 0
    while j < len(test):
        if zz[j] == 0 and int(test[j][0]) == 0:
            tn = tn + 1
        if zz[j] == 0 and int(test[j][0]) == 1:
            fn = fn + 1
        if zz[j] == 1 and int(test[j][0]) == 1:
            tp = tp + 1
        if zz[j] == 1 and int(test[j][0]) == 0:
            fp = fp + 1
        j = j + 1
    print 'tp ', tp
    print 'tn ', tn
    print 'fp ', fp
    print 'fn ', fn
    precision = (float(tp)) / (float(tp) + float(fp))
    recall = (float(tp)) / (float(tp) + float(fn))
    F1 = 2*precision*recall/(precision+recall)
    accuracy = float(tp+tn) / float(tp+tn+fp+fn)
    print 'accuracy ', accuracy
    print 'F1 ', F1
    print 'precision', precision
    print 'recall ', recall
    print '----------------'

blast()
cross()

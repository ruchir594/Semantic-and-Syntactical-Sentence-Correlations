import re
def getWords(data):
    return re.compile(r"[\w]+").findall(data)

def getWordsX(data):
    return re.compile(r"[\w'.]+").findall(data)

def onetime():
    with open('MSRParaphraseCorpus/MSR_paraphrase_train.txt') as f:
        MSRtrain = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_test.txt') as f:
        MSRtest = f.readlines()
    g = []
    i=1
    while i < len(MSRtrain):
        a = MSRtrain[i].split('\t')
        g.append(a[1] + '\t' + a[2] + '\t' + a[3] + '\t' + a[4])
        i=i+1

    i=1
    while i < len(MSRtest):
        a = MSRtest[i].split('\t')
        g.append(a[1] + '\t' + a[2] + '\t' + a[3] + '\t' + a[4])
        i=i+1

    with open('MSR-easy-full.txt', 'w') as f:
        for each in g:
            f.write(each)

def tree():
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    X = [[0, 0,0], [0.6,0.6,0],[1, 1,1]]
    Y = [0, 0,1]
    clf = tree.DecisionTreeClassifier()
    clf = RandomForestClassifier(n_estimators=3)
    clf = clf.fit(X, Y)
    with open('MSRParaphraseCorpus/MSR_paraphrase_train.txt') as f:
        MSRtrain = f.readlines()
    train = []
    for each in MSRtrain:
        train.append(each.split('\t'))
    with open('testdata/singleton-output.txt') as f:
        mypredictions = f.readlines()
    predictions = []
    for each in mypredictions:
        predictions.append(getWordsX(each))
    Y = []
    for each in train:
        Y.append(int(each[0]))
    X = []
    for each in predictions:
        X.append([float(each[0]), float(each[1]), float(each[2])])
    Xtrain = X[0:len(Y)]
    Xtest = X[len(Y):]
    print len(Xtrain), len(Y), len(Xtest)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Xtrain, Y)
    pred = clf.predict(Xtest)
    with open('test-pred.txt', 'w') as f:
        for each in pred:
            f.write(str(each)+'\n')

def cross():
    with open('test-pred.txt') as f:
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

tree()
cross()

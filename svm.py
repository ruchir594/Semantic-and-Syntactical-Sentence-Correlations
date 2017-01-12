import re, word2vec
from dpss import ssv, wo, dp, flex, agreg, union, parse_text, dictlength
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

def predict():
    from spacy.en import English
    parser = English()
    model = word2vec.load('./latents.bin')
    predictions = []
    with open('MSRParaphraseCorpus/MSR-easy-full.txt') as f:
        data = f.readlines()
    #f = open('testdata/output-2.txt', 'w')
    block = []
    for each in data:
        every = each.split('\t')
        block.append([every[0],every[1],agreg(flex(getWords(every[2].lower()))),agreg(flex(getWords(every[3].lower())))])
    #print block
    i = 0
    while i < len(block):
        s1 = block[i][2]
        s2 = block[i][3]
        sent1 = unicode(s1, "utf-8")
        p1, d1 = parse_text(parser, sent1, 1)
        sent2 = unicode(s2, "utf-8")
        p2, d2 = parse_text(parser, sent2, 1)
        t1 = getWords(s1)
        t2 = getWords(s2)
        t1 = flex(t1)
        t2 = flex(t2)
        t = union(t1, t2)
        #print d1
        #print d2
        #print t1
        #print t2
        # -------------- sementic similarity between two sentences ------- #
        similarity_ssv = ssv(t, t1, t2, model)
        #print 'ssv ', similarity_ssv

        # ----------------- word similarity between sentences ------------ #
        similarity_wo = wo(t, t1, t2, model)
        #print 'wo ', similarity_wo

        # ---- dependency matrix based similarity ------------------------ #
        similarity_dp, similarity_dp_cnze = dp(t, t1, t2, d1, d2, model)
        #similarity_dp = 0
        #alpha = 0.8
        c1 = 0.8*similarity_ssv + 0.2*similarity_wo
        #c2 = similarity_dp*similarity_dp
        with open('testdata/singleton-output.txt','a') as f:
            f.write(str(similarity_ssv)+'\t'+str(similarity_wo)+'\t'+str(similarity_dp)+'\t'+str(similarity_dp_cnze)+'\t'+str(block[i][0])+'\t'+str(block[i][1])+'\n')
        z = 0
        i=i+1

def svm():
    from sklearn import svm
    clf = svm.LinearSVC(max_iter=10000)
    with open('MSRParaphraseCorpus/MSR_paraphrase_train.txt') as f:
        MSRtrain = f.readlines()
    train = []
    for each in MSRtrain:
        train.append(each.split('\t'))
    with open('testdata/features-output.txt') as f:
        mypredictions = f.readlines()
    with open('testdata/METEOR-scores.txt') as f:
        METEOR_scores = f.readlines()
    predictions = []
    for each in mypredictions:
        predictions.append(getWordsX(each))
    Y = []
    for each in train:
        Y.append(int(each[0]))
    X = []
    i=0
    for each in predictions:
        '''X.append([float(each[0]), float(each[1]), float(each[2]), float(each[3]), float(each[4]),
                  float(each[3])*float(each[3])*float(each[1]), float(each[3])*float(each[3]), float(each[5]), float(each[6]), float(each[7]),
                  float(each[8]), float(each[9]), float(each[10]), float(each[11]), float(each[12]), float(each[13]), float(each[14]), float(each[15]),
                  float(each[16]), float(each[17]), float(each[18])])'''
        #X.append([float(each[3])*0.80 + float(each[1])*0.20, float(each[18]), float(each[17]), float(each[16]), float(each[12])])
        '''X.append([float(each[3])*0.80 + float(each[1])*0.20,float(each[5]), float(each[6]), float(each[8]),float(each[9]),float(each[10]),float(each[11]),
        float(each[18]), float(each[1]), float(each[2]), float(each[19]) ])'''
        #X.append([float(METEOR_scores[i].split('\t')[1][:-1])])
        X.append([float(each[3])])
        i=i+1

    #print X
    Xtrain = X[0:len(Y)]
    Xtest = X[len(Y):]
    print len(Xtrain), len(Y), len(Xtest)
    clf = clf.fit(Xtrain, Y)
    pred_train = clf.predict(Xtrain)
    pred_test = clf.predict(Xtest)
    with open('data/test-pred-test-svm.txt', 'w') as f:
        for each in pred_test:
            f.write(str(each)+'\n')
    with open('data/test-pred-train-svm.txt', 'w') as f:
        for each in pred_train:
            f.write(str(each)+'\n')

def cross():
    print '-----train-----'
    """with open('data/test-pred-train-svm.txt') as f:
        zz = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_train.txt') as f:
        MSRtrain = f.readlines()
    test = []
    for each in MSRtrain:
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
    print 'recall ', recall"""
    print '-----test------'
    with open('data/test-pred-test-svm.txt') as f:
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

#predict()
svm()
cross()
#print dictlength()

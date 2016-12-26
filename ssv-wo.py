import re
import word2vec
import math
from scipy import spatial

def getWords(data):
    return re.compile(r"[\w']+").findall(data)

def getWordsX(data):
    return re.compile(r"[\w'.]+").findall(data)

def union(t1, t2):
    t = []
    for each in t1:
        if each not in t:
            t.append(each)
    for each in t2:
        if each not in t:
            t.append(each)
    return t

def flex(t):
    tminus = []
    for each in t:
        if each == "don't":
            tminus.append('do')
            tminus.append('not')
        else:
            tminus.append(each)
    return tminus

def agreg(t):
    sent = ''
    for each in t:
        sent = sent + t + ' '
    return sent[:-1]

def suit_sim(b, v):
    delta = 0.6
    r = []
    for each in v:
        r.append(1 - spatial.distance.cosine(b, each))
    m = max(r)
    if m > delta:
        return m
    return 0
#
# Li, Y., McLean, D., Bandar, Z. A., O'Shea, J. D., and Crockett, K. (2006)
# Sentence Similarity Based on Semantic Nets and Corpus Statistics.
# IEEE Transactions on Knowledge and Data Engineering 18, 8, 1138-1150.
#
def ssv(t, t1, t2, model):
    s1 = []
    s2 = []
    v1 = []
    v2 = []
    for i in range(len(t1)):
        try:
            baset1 = model[t1[i]]
        except Exception, e:
            baset1 = [0] * 99
            baset1.append(0.001)
            #print "word not found v1 ssv " + t1[i]
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 99
            baset2.append(0.001)
            #print "word not found v2 ssv " + t2[i]
        v2.append(baset2)
    #print v1, v2
    #print len(t), len(v1), len(v2), len(t1), len(t2), t
    for i in range(len(t)):
        if t[i] in t1:
            s1.append(1)
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 99
                baset.append(0.001)
                #print "word not found t[i] ssv " + t[i]
            s1.append(suit_sim(baset, v1))
        if t[i] in t2:
            s2.append(1)
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 99
                baset.append(0.001)
                #print "word not found t[i] ssv " + t[i]
            s2.append(suit_sim(baset, v2))
    #print 'sss ',s1, s2
    similarity = 1 - spatial.distance.cosine(s1, s2)
    return similarity

def suit_index(b, v):
    delta = 0.6
    r = []
    for each in v:
        r.append(1 - spatial.distance.cosine(b, each))
    m = max(r)
    if m > delta:
        return r.index(m) + 1
    return 0

def norm(r):
    total = 0
    for each in r:
        total = total + each*each
    return math.sqrt(total)

#
# Li, Y., McLean, D., Bandar, Z. A., O'Shea, J. D., and Crockett, K. (2006)
# Sentence Similarity Based on Semantic Nets and Corpus Statistics.
# IEEE Transactions on Knowledge and Data Engineering 18, 8, 1138-1150.
#
def wo(t, t1, t2, model):
    delta = 0.6
    r1 = []
    r2 = []
    v1 = []
    v2 = []
    for i in range(len(t1)):
        try:
            baset1 = model[t1[i]]
        except Exception, e:
            baset1 = [0] * 99
            baset1.append(0.001)
            #print "word not found v1 wo " + t1[i]
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 99
            baset2.append(0.001)
            #print "word not found v2 wo " + t2[i]
        v2.append(baset2)

    for i in range(len(t)):
        if t[i] in t1:
            r1.append(t1.index(t[i])+1)
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 99
                baset.append(0.001)
                #print "word not found t[i] wo " + t[i]
            r1.append(suit_index(baset, v1))
        if t[i] in t2:
            r2.append(t2.index(t[i])+1)
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 99
                baset.append(0.001)
                #print "word not found t[i] wo " + t[i]
            r2.append(suit_index(baset, v2))
    #print r1, r2
    r = []
    q = []
    for i in range(len(r1)):
        r.append(r1[i]-r2[i])
        q.append(r1[i]+r2[i])
    r = norm(r)
    q = norm(q)
    #print r,q
    return (1 - r/q)

def test():
    # ------------ common between two measurments ---------------------------- #
    t1 = "a quick brown dog jumps over the lazy fox"
    t2 = "a quick brown fox jumps over the lazy dog"
    t2 = "jumps over the lazy fox is a quick brown dog"
    #t1 = "Amrozi accused his brother, whom he called the witness, of deliberately distorting his evidence.".lower()
    #t2 = "Referring to him as only the witness, Amrozi accused his brother of deliberately distorting his evidence.".lower()
    #t1 = "i have to find you, tell me you need me."
    #t2 = "don't wanna know who is taking you home"
    t1 = getWords(t1)
    t2 = getWords(t2)
    t1 = flex(t1)
    t2 = flex(t2)
    t = union(t1, t2)
    #t = ["a", "brown", "jumps", "the", "fox", "dog", "quick", "over", "lazy"]
    print t

    model = word2vec.load('./latents.bin')
    # -------------- sementic similarity between two sentences --------------- #
    similarity_ssv = ssv(t, t1, t2, model)
    print 'ssv ', similarity_ssv

    # ----------------- word similarity between sentences -------------------- #
    similarity_wo = wo(t, t1, t2, model)
    print 'wo ', similarity_wo

    alpha = 0.8
    print alpha*similarity_ssv + (1-alpha)*similarity_wo

def predict():
    model = word2vec.load('./latents.bin')
    predictions = []
    with open('MSRParaphraseCorpus/MSR_easy.txt') as f:
        data = f.readlines()
    block = []
    for each in data:
        block.append(flex(getWords(each.lower())))
    i = 1
    while i+1 < len(block):
        if int(block[i][0]) - int(block[i+1][0]) < 200 and int(block[i][0]) - int(block[i+1][0]) > -200:
            t1 = block[i][1:]
            t2 = block[i+1][1:]
            t = union(t1, t2)
            # -------------- sementic similarity between two sentences ------- #
            similarity_ssv = ssv(t, t1, t2, model)
            #print 'ssv ', similarity_ssv

            # ----------------- word similarity between sentences ------------ #
            similarity_wo = wo(t, t1, t2, model)
            #print 'wo ', similarity_wo

            alpha = 0.8
            similarity = alpha*similarity_ssv + (1-alpha)*similarity_wo
            print similarity, str(block[i][0]), str(block[i+1][0])
            predictions.append([similarity, str(block[i][0]), str(block[i+1][0])])
            i = i + 2
        else:
            i = i + 1


def cross():
    # ------------ comparing with both test and train as we have not
    # -------- trained any model using trainset ------------------------------ #
    #thresh = 0.54
    thresh = 0.56
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    neg = 0
    pos = 0
    with open('MSRParaphraseCorpus/output-ssv-wo.txt') as f:
        mypredictions = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_train.txt') as f:
        MSRtrain = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_test.txt') as f:
        MSRtest = f.readlines()
    train = []
    test = []
    predictions = []
    for each in mypredictions:
        predictions.append(getWordsX(each))
    for each in MSRtrain:
        train.append(getWords(each))
    for each in MSRtest:
        test.append(getWords(each))
    #print len(test), len(train), len(predictions)
    #print type(predictions[0][0]), type(predictions[1][1]), type(predictions[2][2])
    #print predictions[0][0], predictions[1][1], predictions[2][2]
    neg = 0
    pos = 0
    i = 1
    for every in train:
        for each in predictions:
            if each[1] == every[1] and each[2] == every[2]:
                if float(each[0]) > thresh:
                    if int(every[0]) == 1:
                        s1 = s1 + 1
                    else:
                        s2 = s2 + 1
                if float(each[0]) < thresh:
                    if int(every[0]) == 0:
                        s3 = s3 + 1
                    else:
                        s4 = s4 + 1
            if each[1] == every[2] and each[2] == every[1]:
                if float(each[0]) > thresh:
                    if int(every[0]) == 1:
                        s1 = s1 + 1
                    else:
                        s2 = s2 + 1
                if float(each[0]) < thresh:
                    if int(every[0]) == 0:
                        s3 = s3 + 1
                    else:
                        s4 = s4 + 1
    for every in test:
        for each in predictions:
            if each[1] == every[1] and each[2] == every[2]:
                if float(each[0]) > thresh:
                    if int(every[0]) == 1:
                        s1 = s1 + 1
                    else:
                        s2 = s2 + 1
                if float(each[0]) < thresh:
                    if int(every[0]) == 0:
                        s3 = s3 + 1
                    else:
                        s4 = s4 + 1
            if each[1] == every[2] and each[2] == every[1]:
                if float(each[0]) > thresh:
                    if int(every[0]) == 1:
                        s1 = s1 + 1
                    else:
                        s2 = s2 + 1
                if float(each[0]) < thresh:
                    if int(every[0]) == 0:
                        s3 = s3 + 1
                    else:
                        s4 = s4 + 1
    #print len(test) + len(train), s1 + s2 +s3 + s4
    #print neg, pos, s1, s4, s3, s2
    '''total = float(neg+pos)
    true_positive = float(s1)/float(pos)
    false_negative = float(s3)/float(neg)
    true_negative = float(s4)/float(neg)
    false_positive = float(s2)/float(pos)
    '''
    true_positive = float(s1)
    false_positive = float(s2)
    false_negative = float(s4)
    true_negative = float(s3)
    precision = (float(true_positive)) / (float(true_positive) + float(false_positive))
    recall = (float(true_positive)) / (float(true_positive) + float(false_negative))
    #print 'total ', total
    print 'true_positive ', true_positive
    print 'false_negative ', false_negative
    print 'true_negative ', true_negative
    print 'false_positive ', false_positive
    print 'precision ', precision
    print 'recall ', recall
    print 'F1 ', 2*precision*recall/(precision+recall)
    print 'accuracy ', (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

#test()
#predict()
cross()

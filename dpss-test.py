import requests, json, re, word2vec, math
from scipy import spatial
import numpy
import word2vec
from space_action import get_postagging, get_dependency

def parse_text(parser, sentence, state = None):
    parsedEx = parser(sentence)
    # extract POS tagging from the parsed response
    full_pos = get_postagging(parsedEx)
    # extract dependency tree from parsed response
    full_dep = get_dependency(parsedEx)
    return full_pos, full_dep

def get_root(full_dep):
    root_word = ''
    for each_dep in full_dep:
        if each_dep[1] == 'ROOT':
            root_word = each_dep[0].lower()
    return root_word

def getWords(data):
    return re.compile(r"[\w]+").findall(data)

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
        sent = sent + each + ' '
    return sent[:-1]

def suit_sim(b, v):
    delta = 0.6
    r = []
    #print type(b)
    for each in v:
        #print type(each)
        r.append(1 - spatial.distance.cosine(b, each))
        #print 1 - spatial.distance.cosine(b, each)
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
            baset1 = numpy.random.rand(100,1)
            #print "word not found v1 ssv " + t1[i]
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 99
            baset2.append(0.001)
            baset2 = numpy.random.rand(100,1)
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
                #print 'exception at ' + t[i]
                baset = [0] * 99
                baset.append(0.001)
                baset = numpy.random.rand(100,1)
                #print "word not found t[i] ssv " + t[i]
            #print suit_sim(baset, v1)
            s1.append(suit_sim(baset, v1))
        if t[i] in t2:
            s2.append(1)
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                #print 'exception at ' + t[i]
                baset = [0] * 99
                baset.append(0.001)
                baset = numpy.random.rand(100,1)
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
            baset1 = numpy.random.rand(100,1)
            #print "word not found v1 wo " + t1[i]
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 99
            baset2.append(0.001)
            baset2 = numpy.random.rand(100,1)
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
                baset = numpy.random.rand(100,1)
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
                baset = numpy.random.rand(100,1)
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

def dp(t, t1, t2, d1, d2, model):
    v1 = []
    v2 = []
    for i in range(len(t1)):
        try:
            baset1 = model[t1[i]]
        except Exception, e:
            baset1 = [0] * 99
            baset1.append(0.001)
            baset1 = numpy.random.rand(100,1)
            #print "word not found v1 wo " + t1[i]
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 99
            baset2.append(0.001)
            baset2 = numpy.random.rand(100,1)
            #print "word not found v2 wo " + t2[i]
        v2.append(baset2)
    # not the v1 and v2 have all vectors
    m1 = numpy.zeros((len(t),len(t)))
    m2 = numpy.zeros((len(t),len(t)))
    for i in range(len(t)):
        if t[i] in t1:
            m1[i][i] = 1*4
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = numpy.random.rand(100,1)
            m1[i][i] = suit_sim(baset, v1)*4
        if t[i] in t2:
            m2[i][i] = 1*4
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = numpy.random.rand(100,1)
            m2[i][i] = suit_sim(baset, v2)*4
    for i in range(len(d1)):
        d1[i][1] = 'DEP'
        d1[i][3] = []
        d1[i][4] = []
    for i in range(len(d2)):
        d2[i][1] = 'DEP'
        d2[i][3] = []
        d2[i][4] = []
    for i in range(len(t)):
        for j in range(len(t)):
            if [t[i],'DEP',t[j],[],[]] in d1:
                m1[i][j] = 1
                m1[j][i] = 1
            if [t[i],'DEP',t[j],[],[]] in d2:
                m2[i][j] = 1
                m2[j][i] = 1

    print m1
    print m2
    similarity_dp = 1 - float(numpy.count_nonzero(m1-m2)) / float((numpy.count_nonzero(m1) + numpy.count_nonzero(m2)))
    print 'cnze ', similarity_dp
    similarity_dp = 1 - numpy.linalg.norm(m1-m2)/(numpy.linalg.norm(m1)+numpy.linalg.norm(m2))
    print 'norm ', similarity_dp
    return similarity_dp
#dp(["hello", "a","b","c"],[],[],[],[])

def advance_ssv(t, t1, t2):
    v1 = []
    v2 = []


def test():
    from spacy.en import English
    parser = English()
    model = word2vec.load('./latents.bin')
    t1 = "a quick brown dog jumps over the lazy fox"
    t2 = "a fast brown fox jumps over the lazy dog"
    #t2 = "he is a brown fox"
    #t2 = "jumps over the lazy fox is a quick brown dog"
    sentence_1 = unicode(t1, "utf-8")
    p1, d1 = parse_text(parser, sentence_1, 1)
    sentence_2 = unicode(t2, "utf-8")
    p2, d2 = parse_text(parser, sentence_2, 1)
    t1 = getWords(t1)
    t2 = getWords(t2)
    t1 = flex(t1)
    t2 = flex(t2)
    t = union(t1, t2)
    #print d1
    #print d2
    similarity_dp = dp(t, t1, t2, d1, d2, model)
    similarity_dp = dp_old(t, t1, t2, d1, d2)
    # -------------- sementic similarity between two sentences ------- #
    similarity_ssv = ssv(t, t1, t2, model)
    #print 'ssv ', similarity_ssv

    # ----------------- word similarity between sentences ------------ #
    similarity_wo = wo(t, t1, t2, model)
    print similarity_ssv, similarity_wo, similarity_dp

#test()
def predict():
    from spacy.en import English
    parser = English()
    model = word2vec.load('./latents.bin')
    predictions = []
    with open('MSRParaphraseCorpus/MSR_easy.txt') as f:
        data = f.readlines()
    f = open('testdata/output-2.txt', 'w')
    block = []
    for each in data:
        block.append(flex(getWords(each.lower())))
    i = 1
    while i+1 < len(block):
        if int(block[i][0]) - int(block[i+1][0]) < 200 and int(block[i][0]) - int(block[i+1][0]) > -200:
            t1 = block[i][1:]
            t2 = block[i+1][1:]
            s1 = agreg(t1)
            s2 = agreg(t2)
            s1 = unicode(s1, "utf-8")
            p1, d1 = parse_text(parser, s1, 1)
            s2 = unicode(s2, "utf-8")
            p2, d2 = parse_text(parser, s2, 1)
            t1 = flex(t1)
            t2 = flex(t2)
            t = union(t1, t2)
            # -------------- sementic similarity between two sentences ------- #
            similarity_ssv = ssv(t, t1, t2, model)
            #print 'ssv ', similarity_ssv

            # ----------------- word similarity between sentences ------------ #
            similarity_wo = wo(t, t1, t2, model)
            #print 'wo ', similarity_wo

            # ---- dependency matrix based similarity ------------------------ #
            similarity_dp = dp(t, t1, t2, d1, d2)

            #alpha = 0.8
            z = 0
            similarity = float(distr[z][0])*similarity_ssv + float(distr[z][1])*similarity_wo + float(distr[z][2])*similarity_dp
            with open('testdata/output-testing.txt', 'a') as f:
                f.write(str(similarity))
                f.write(' ')
                f.write(str(block[i][0]))
                f.write(' ')
                f.write(str(block[i+1][0]))
                f.write('\n')
            #similarity = 0.75*similarity_ssv + 0.15*similarity_wo + 0.10*similarity_dp
            print i, i+1
            '''f.write(str(similarity))
            f.write(' ')
            f.write(str(block[i][0]))
            f.write(' ')
            f.write(str(block[i+1][0]))
            f.write('\n')'''
            #predictions.append([similarity, str(block[i][0]), str(block[i+1][0])])
            i = i + 2
        else:
            i = i + 1

def cross():
    # ------------ comparing with both test and train as we have not
    # -------- trained any model using trainset ------------------------------ #
    #thresh = 0.54
    m_accuracy = 0
    m_z = 0
    m_thresh = 0
    thols = [0.45,0.47,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.58,0.60]
    thresh = 0.54
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    neg = 0
    pos = 0
    with open('testdata/output-testing.txt') as f:
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
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
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
    F1 = 2*precision*recall/(precision+recall)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    #print 'total ', total
    print 'true_positive ', true_positive
    print 'false_negative ', false_negative
    print 'true_negative ', true_negative
    print 'false_positive ', false_positive
    print 'precision ', precision
    print 'recall ', recall
    print 'F1 ', F1
    print 'accuracy ', accuracy

def dp_old(t, t1, t2, d1, d2):
    m1 = numpy.zeros((len(t),len(t)))
    m2 = numpy.zeros((len(t),len(t)))
    for each in d1:
        try:
            m1[t.index(each[0])][t.index(each[2])] = 1
            m1[t.index(each[2])][t.index(each[0])] = 1
            m1[t.index(each[0])][t.index(each[0])] = 1
            m1[t.index(each[2])][t.index(each[2])] = 1
        except Exception, e:
            j = None
    for each in d2:
        try:
            m2[t.index(each[0])][t.index(each[2])] = 1
            m2[t.index(each[2])][t.index(each[0])] = 1
            m2[t.index(each[0])][t.index(each[0])] = 1
            m2[t.index(each[2])][t.index(each[2])] = 1
        except Exception, e:
            j = None
    print m1
    print m2
    similarity_dp_1 = 1 - numpy.linalg.norm(m1-m2)/(numpy.linalg.norm(m1)+numpy.linalg.norm(m2))
    similarity_dp_2 = 1 - float(numpy.count_nonzero(m1-m2)) / float((numpy.count_nonzero(m1) + numpy.count_nonzero(m2)))
    #similarity_dp = 1 - numpy.linalg.norm(m1-m2)/numpy.linalg.norm(m1+m2)
    print 'old similarity_dp ', similarity_dp_1
    return similarity_dp_1

test()
#predict()
#cross()

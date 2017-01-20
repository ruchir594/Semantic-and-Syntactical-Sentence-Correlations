import requests, json, re, word2vec, math
from scipy import spatial
import numpy

from space_action import get_postagging, get_dependency

hashing = dict()

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

def intersection(t1, t2):
    t = [val for val in t1 if val in t2]
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

def polish(t):
    tminus = []
    remobe = ['PUNCT', 'DET']
    for each in t:
        if each[1] not in remobe:
            tminus.append(each[0])
    return tminus

def agreg(t):
    sent = ''
    for each in t:
        sent = sent + each + ' '
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
    """with open('hashing.json') as data_file:
        hashing = json.load(data_file)"""
    for i in range(len(t1)):
        try:
            baset1 = model[t1[i]]
        except Exception, e:
            baset1 = [0] * 99
            baset1.append(0.001)
            #baset1 = numpy.random.rand(100,1)
            #print "word not found v1 ssv " + t1[i]
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 99
            baset2.append(0.001)
            #baset2 = numpy.random.rand(100,1)
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
                #baset = numpy.random.rand(100,1)
                #print "word not found t[i] ssv " + t[i]
            #print suit_sim(baset, v1)
            s1.append(suit_sim(baset, v1))
        if t[i] in t2:
            s2.append(1)
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 99
                baset.append(0.001)
                #baset = numpy.random.rand(100,1)
                #print "word not found t[i] ssv " + t[i]
            s2.append(suit_sim(baset, v2))
    #print 'sss ',s1, s2
    similarity = 1 - spatial.distance.cosine(s1, s2)
    """with open('hashing.json', 'w') as fp:
        json.dump(hashing, fp)"""
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
    """with open('hashing.json') as data_file:
        hashing = json.load(data_file)"""
    for i in range(len(t1)):
        try:
            baset1 = model[t1[i]]
        except Exception, e:
            baset1 = [0] * 99
            baset1.append(0.001)
            #baset1 = numpy.random.rand(100,1)
            #print "word not found v1 wo " + t1[i],  baset1
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 99
            baset2.append(0.001)
            #baset2 = numpy.random.rand(100,1)
            #print "word not found v2 wo " + t2[i]
            #print "word not found v2 wo " + t2[i],  baset2
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
                #baset = numpy.random.rand(100,1)
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
                #baset = numpy.random.rand(100,1)
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
    """with open('hashing.json', 'w') as fp:
        json.dump(hashing, fp)"""
    return (1 - r/q)

def dp(t, t1, t2, d1, d2, model):
    weight = 4
    v1 = []
    v2 = []
    """with open('hashing.json') as data_file:
        hashing = json.load(data_file)"""
    for i in range(len(t1)):
        try:
            baset1 = model[t1[i]]
        except Exception, e:
            baset1 = [0] * 99
            baset1.append(0.001)
            #baset1 = numpy.random.rand(100,1)
            #print "word not found v1 wo " + t1[i]
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 99
            baset2.append(0.001)
            #baset2 = numpy.random.rand(100,1)
            #print "word not found v2 wo " + t2[i]
        v2.append(baset2)
    # not the v1 and v2 have all vectors
    m1 = numpy.zeros((len(t),len(t)))
    m2 = numpy.zeros((len(t),len(t)))
    for i in range(len(t)):
        if t[i] in t1:
            m1[i][i] = 1*weight
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 99
                baset.append(0.001)
                #baset = numpy.random.rand(100,1)
            m1[i][i] = suit_sim(baset, v1)*weight
        if t[i] in t2:
            m2[i][i] = 1*weight
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 99
                baset.append(0.001)
                #baset = numpy.random.rand(100,1)
            m2[i][i] = suit_sim(baset, v2)*weight
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

    #print m1
    #print m2
    similarity_dp_cnze = 1 - float(numpy.count_nonzero(m1-m2)) / float((numpy.count_nonzero(m1) + numpy.count_nonzero(m2)))
    #print 'cnze ', similarity_dp_cnze
    similarity_dp = 1 - numpy.linalg.norm(m1-m2)/(numpy.linalg.norm(m1)+numpy.linalg.norm(m2))
    #print 'norm ', similarity_dp
    """with open('hashing.json', 'w') as fp:
        json.dump(hashing, fp)"""
    return similarity_dp, similarity_dp_cnze
#dp(["hello", "a","b","c"],[],[],[],[])

def advance_ssv(t, t1, t2):
    v1 = []
    v2 = []

def test():
    from spacy.en import English
    parser = English()
    model = word2vec.load('./latents.bin')
    t1 = "a quick brown dog jumps over the lazy fox"
    t2 = "a quick brown fox jumps over the lazy dog"
    t2 = "jumps over the lazy fox is a quick brown dog"
    t1="The DVD-CCA then appealed to the state Supreme Court."
    t2="The DVD CCA appealed that decision to the U.S. Supreme Court."
    sentence_1 = unicode(t1, "utf-8")
    p1, d1 = parse_text(parser, sentence_1, 1)
    sentence_2 = unicode(t2, "utf-8")
    p2, d2 = parse_text(parser, sentence_2, 1)
    t1 = getWords(t1)
    t2 = getWords(t2)
    t1 = flex(t1)
    t2 = flex(t2)
    t = union(t1, t2)
    print d1
    print d2
    print t1
    print t2
    similarity_dp = dp(t, t1, t2, d1, d2, model)
    print similarity_dp

distr = [[0.7,0.2,0.1],[0.7,0.19,0.11],[0.7,0.18,0.12],[0.7,0.17,0.13],[0.7,0.16,0.14],[0.7,0.15,0.15],
         [0.7,0.14,0.15],[0.7,0.13,0.16],[0.7,0.12,0.17],[0.7,0.11,0.19],[0.7,0.1,0.2],[0.7,0.21,0.09],
         [0.7,0.22,0.08],[0.7,0.23,0.07],[0.7,0.24,0.06],[0.7,0.08,0.22],[0.7,0.06,0.24],
         [0.8,0.02,0.18],[0.8,0.04,0.16],[0.8,0.06,0.14],[0.8,0.08,0.12],[0.8,0.1,0.1],[0.8,0.12,0.08],
         [0.8,0.14,0.06],[0.8,0.16,0.04],[0.8,0.18,0.02],[0.8,0.2,0],[0.8,0,0.2]]
distr = [[0.8,0.2,0],[0.79,0.21,0],[0.78,0.22,0],[0.77,0.23,0],[0.76,0.24,0],[0.75,0.25,0],[0.74,0.26,0],[0.73,0.27,0],[0.72,0.28,0],[0.71,0.29,0],
         [0.81,0.19,0],[0.82,0.18,0],[0.83,0.17,0],[0.84,0.16,0],[0.85,0.15,0]]
distr = [[0.8,0.2,0],[0.76,0.19,0.05],[0.72,0.18,0.1],[0.78,0.22,0],[0.78,0.11,0.11],[0.78,0.12,0.1],[0.78,0.1,0.12]]
distr = [[0,0,1],[0,0.2,0.8],[0,0.19,0.81],[0,0.18,0.82],[0,0.17,0.83],[0,0.16,0.84],[0,0.21,0.79],[0,0.23,0.77],[0,0.24,0.76],[0,0.3,0.7],
         [0,0.4,0.6],[0.4,0.2,0.4],[0.3,0.3,0.4],[0.33,0.33,0.34],[0.5,0,0.5]]
distr = [[1,0,0],[0,1,0],[0,0,1]]
#len(distr)
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
        block.append([every[0],every[1],agreg(flex(getWords(every[2]))),agreg(flex(getWords(every[3])))])
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
        similarity_dp = dp(t, t1, t2, d1, d2, model)
        #similarity_dp = 0
        #alpha = 0.8
        with open('testdata/singleton-output.txt','a') as f:
            f.write(str(similarity_ssv)+'\t'+str(similarity_wo)+'\t'+str(similarity_dp)+'\t'+str(block[i][0])+'\t'+str(block[i][1])+'\n')
        z = 0
        '''while z < len(distr):
            similarity = float(distr[z][0])*similarity_ssv + float(distr[z][1])*similarity_wo + float(distr[z][2])*similarity_dp
            with open('testdata/output-'+str(z)+'.txt', 'a') as f:
                f.write(str(similarity))
                f.write(' ')
                f.write(str(block[i][0]))
                f.write(' ')
                f.write(str(block[i][1]))
                f.write('\n')
            z = z + 1'''
        i=i+1
        #similarity = 0.75*similarity_ssv + 0.15*similarity_wo + 0.10*similarity_dp
        #print i, i+1
        '''f.write(str(similarity))
        f.write(' ')
        f.write(str(block[i][0]))
        f.write(' ')
        f.write(str(block[i+1][0]))
        f.write('\n')'''
        #predictions.append([similarity, str(block[i][0]), str(block[i+1][0])])

def cross():
    # ------------ comparing with both test and train as we have not
    # -------- trained any model using trainset ------------------------------ #
    #thresh = 0.54
    m_accuracy = 0
    m_z = 0
    m_thresh = 0
    thols = [0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.60]
    #thols = [0.55,0.555,0.558,0.5585,0.559,0.5595,0.56,0.561,0.562,0.565,0.57]
    thresh = 0.54
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    neg = 0
    pos = 0
    #with open('testdata/output-2.txt') as f:
    #    mypredictions = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_train.txt') as f:
        MSRtrain = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_test.txt') as f:
        MSRtest = f.readlines()
    train = []
    test = []
    #predictions = []
    #for each in mypredictions:
    #    predictions.append(getWordsX(each))
    for each in MSRtrain:
        train.append(getWords(each))
    for each in MSRtest:
        test.append(getWords(each))
    #print len(test), len(train), len(predictions)
    #print type(predictions[0][0]), type(predictions[1][1]), type(predictions[2][2])
    #print predictions[0][0], predictions[1][1], predictions[2][2]
    z = 0
    y = 0
    while z < len(distr):
        with open('testdata/output-'+str(z)+'.txt') as f:
            mypredictions = f.readlines()
        predictions = []
        for each in mypredictions:
            predictions.append(getWordsX(each))
        for thresh in thols:
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
            print z, thresh, accuracy
            if accuracy > m_accuracy:
                m_accuracy = accuracy
                m_z = z
                m_thresh = thresh
            #print 'total ', total
            with open('testdata/result-'+str(y)+'.txt', 'w') as f:
                f.write('true_positive ' + str(true_positive) + '\n')
                f.write('false_negative ' + str(false_negative) + '\n')
                f.write('true_negative ' + str(true_negative) + '\n')
                f.write('false_positive ' + str(false_positive) + '\n')
                f.write('precision ' + str(precision) + '\n')
                f.write('recall ' +str(recall) + '\n')
                f.write('F1 ' + str(F1) + '\n')
                f.write('accuracy ' + str(accuracy) + '\n')
            y = y + 1
            '''print 'true_positive ', true_positive
            print 'false_negative ', false_negative
            print 'true_negative ', true_negative
            print 'false_positive ', false_positive
            print 'precision ', precision
            print 'recall ', recall
            print 'F1 ', 2*precision*recall/(precision+recall)
            print 'accuracy ', (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)'''
        print z
        z = z + 1
    print '---------------------------------------'
    print m_z, m_thresh
    print m_accuracy

def adv_cross():
    with open('testdata/singleton-output.txt') as f:
        predictions = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_train.txt') as f:
        MSRtrain = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_test.txt') as f:
        MSRtest = f.readlines()
    train = []
    test = []
    for each in MSRtrain:
        train.append(getWords(each))
    for each in MSRtest:
        test.append(getWords(each))
    i = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    t1=0
    t2=0
    t3=0
    t4=0
    thresh = 0.56
    zz= []
    print len(train), len(test), len(predictions)
    while i < len(train):
        p = predictions[i].split('\t')
        p[2] = float(p[2])
        p[1] = float(p[1])
        p[0] = float(p[0])
        train[i][0] = int(train[i][0])
        if p[2] < 0.6 and train[i][0] == 0:
            tn = tn + 1
        elif p[0] > 0.58 and train[i][0] == 1:
            tp = tp + 1
        else:
            pred = p[0]*0.8 + p[1]*0.2
            if pred > thresh:
                if train[i][0] == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if train[i][0] == 0:
                    tn = tn + 1
                else:
                    fn = fn + 1
        i = i + 1
    j = 0
    while j < len(test):
        p = predictions[i].split('\t')
        p[2] = float(p[2])
        p[1] = float(p[1])
        p[0] = float(p[0])
        test[j][0] = int(test[j][0])
        if p[2] < 0.6 and test[j][0] == 0:
            tn = tn + 1
        elif p[0] > 0.58 and test[j][0] == 1:
            tp = tp + 1
        else:
            pred = p[0]*0.8 + p[1]*0.2
            if pred > thresh:
                if test[j][0] == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if test[j][0] == 0:
                    tn = tn + 1
                else:
                    fn = fn + 1
        i = i + 1
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



def double_cross():
    with open('testdata/singleton-output.txt') as f:
        predictions = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_train.txt') as f:
        MSRtrain = f.readlines()
    with open('MSRParaphraseCorpus/MSR_paraphrase_test.txt') as f:
        MSRtest = f.readlines()
    train = []
    test = []
    for each in MSRtrain:
        train.append(getWords(each))
    for each in MSRtest:
        test.append(getWords(each))
    i = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    t1=0
    t2=0
    t3=0
    t4=0
    thresh = 0.55
    zz= []
    while i < len(train):
        zz.append(0)
        p = predictions[i].split('\t')
        p[2] = float(p[2])
        p[1] = float(p[1])
        p[0] = float(p[0])
        train[i][0] = int(train[i][0])
        if p[2] < 0.6:
            zz[i] = 0
        elif (p[2] + p[0])/2 > 0.60:
            zz[i] = 1
        else:
            pred = p[0]*0.8 + p[1]*0.2
            if pred > thresh:
                zz[i]=1
            else:
                zz[i]=0
        if p[1] < 0.5:
            zz[i] =0
        if p[2] > 0.70:
            zz[i] = 1
        i = i + 1
    j = 0
    while j < len(test):
        zz.append(0)
        p = predictions[i].split('\t')
        p[2] = float(p[2])
        p[1] = float(p[1])
        p[0] = float(p[0])
        test[j][0] = int(test[j][0])
        if p[2] < 0.6:
            zz[i] = 0
        elif (p[2] + p[0])/2 > 0.60:
            zz[i] = 1
        else:
            pred = p[0]*0.8 + p[1]*0.2
            if pred > thresh:
                zz[i]=1
            else:
                zz[i]=0
        if p[1] < 0.5:
            zz[i] =0
        if p[2] > 0.70:
            zz[i] = 1
        i = i + 1
        j = j + 1
    print 'zz ', len(zz)
    i=0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    while i < len(train):
        if zz[i] == 0 and train[i][0] == 0:
            tn = tn + 1
        if zz[i] == 0 and train[i][0] == 1:
            fn = fn + 1
        if zz[i] == 1 and train[i][0] == 1:
            tp = tp + 1
        if zz[i] == 1 and train[i][0] == 0:
            fp = fp + 1
        i = i + 1
    j = 0
    while j < len(test):
        if zz[i] == 0 and test[j][0] == 0:
            tn = tn + 1
        if zz[i] == 0 and test[j][0] == 1:
            fn = fn + 1
        if zz[i] == 1 and test[j][0] == 1:
            tp = tp + 1
        if zz[i] == 1 and test[j][0] == 0:
            fp = fp + 1
        i = i + 1
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
#
#adv_cross()
#test()
#predict()
#double_cross()
#cross()
def dictlength():
    with open('hashing.json') as data_file:
        hashing = json.load(data_file)
    return len(hashing)

'''
print int(p[3]),int(p[4]),int(train[i][1]),int(train[i][2])
if int(train[i][1]) != int(p[3]):
    print 'fuck it'
    print p, train[i]
    break
if int(train[i][2]) != int(p[4]):
    print 'fuck it'
    print p, train[i]
    break
if int(test[j][1]) != int(p[3]):
    print 'fuck it'
    print p, test[i]
    break
if int(test[j][2]) != int(p[4]):
    print 'fuck it'
    print p, test[j]
    break
'''

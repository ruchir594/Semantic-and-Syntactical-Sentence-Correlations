import re
import word2vec
import math
from scipy import spatial

def getWords(data):
    return re.compile(r"[\w']+").findall(data)

def union(t1, t2):
    t = []
    for each in t1:
        if each not in t:
            t.append(each)
    for each in t2:
        if each not in t:
            t.append(each)
    return t

def suit_index(b, v):
    delta = 0.6
    r = []
    for each in v:
        r.append(1 - spatial.distance.cosine(b, each))
    m = max(r)
    if m > delta:
        return r.index(m) + 1
    return 0

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
def ssv(t, t1, t2):
    model = word2vec.load('./latents.bin')
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
            print "word not found v1 ssv " + t1[i]
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 99
            baset2.append(0.001)
            print "word not found v2 ssv " + t2[i]
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
                print "word not found t[i] ssv " + t[i]
            s1.append(suit_sim(baset, v1))
        if t[i] in t2:
            s2.append(1)
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 99
                baset.append(0.001)
                print "word not found t[i] ssv " + t[i]
            s2.append(suit_sim(baset, v2))
    #print 'sss ',s1, s2
    similarity = 1 - spatial.distance.cosine(s1, s2)
    return similarity

#
# Li, Y., McLean, D., Bandar, Z. A., O'Shea, J. D., and Crockett, K. (2006)
# Sentence Similarity Based on Semantic Nets and Corpus Statistics.
# IEEE Transactions on Knowledge and Data Engineering 18, 8, 1138-1150.
#
def wo(t, t1, t2):
    model = word2vec.load('./latents.bin')
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
            print "word not found v1 wo " + t1[i]
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 99
            baset2.append(0.001)
            print "word not found v2 wo " + t2[i]
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
                print "word not found t[i] wo " + t[i]
            r1.append(suit_index(baset, v1))
        if t[i] in t2:
            r2.append(t2.index(t[i])+1)
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 99
                baset.append(0.001)
                print "word not found t[i] wo " + t[i]
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

def norm(r):
    total = 0
    for each in r:
        total = total + each*each
    return math.sqrt(total)


def test():
    # ------------ common between two measurments ---------------------------- #
    t1 = "a quick brown dog jumps over the lazy fox"
    t2 = "a quick brown fox jumps over the lazy dog"
    #t1 = "Amrozi accused his brother, whom he called the witness, of deliberately distorting his evidence.".lower()
    #t2 = "Referring to him as only the witness, Amrozi accused his brother of deliberately distorting his evidence.".lower()
    t1 = getWords(t1)
    t2 = getWords(t2)
    t = union(t1, t2)
    #t = ["a", "brown", "jumps", "the", "fox", "dog", "quick", "over", "lazy"]
    print t

    # -------------- sementic similarity between two sentences --------------- #
    similarity = ssv(t, t1, t2)
    print 'ssv ', similarity

    # ----------------- word similarity between sentences -------------------- #
    similarity = wo(t, t1, t2)
    print 'wo ', similarity


test()

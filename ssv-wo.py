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
#
# Li, Y., McLean, D., Bandar, Z. A., O'Shea, J. D., and Crockett, K. (2006)
# Sentence Similarity Based on Semantic Nets and Corpus Statistics.
# IEEE Transactions on Knowledge and Data Engineering 18, 8, 1138-1150.
#
def ssv(t, t1, t2):
    model = word2vec.load('./latents.bin')
    s1 = []
    s2 = []
    for i in range(len(t)):
        try:
            baset = model[t[i]]
        except Exception, e:
            baset = [0] * 100
            print "word not found " + t[i]
        try:
            baset1 = model[t1[i]]
        except Exception, e:
            baset1 = [0] * 100
            print "word not found " + t1[i]
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 100
            print "word not found " + t2[i]
        result1 = 1 - spatial.distance.cosine(baset, baset1)
        result2 = 1 - spatial.distance.cosine(baset, baset2)
        s1.append(result1)
        s2.append(result2)
        similarity = 1 - spatial.distance.cosine(s1, s2)
    return similarity

def suit(b, v):
    delta = 0.6
    r = []
    for each in v:
        r.append(1 - spatial.distance.cosine(b, each))
    m = max(r)
    if m > delta:
        return r.index(m) + 1
    return 0
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
            baset1 = [0] * 100
            print "word not found " + t1[i]
        v1.append(baset1)
    for i in range(len(t2)):
        try:
            baset2 = model[t2[i]]
        except Exception, e:
            baset2 = [0] * 100
            print "word not found " + t2[i]
        v2.append(baset2)

    for i in range(len(t)):
        if t[i] in t1:
            r1.append(t1.index(t[i])+1)
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 100
                print "word not found " + t[i]
            r1.append(suit(baset, v1))
        if t[i] in t2:
            r2.append(t2.index(t[i])+1)
        else:
            try:
                baset = model[t[i]]
            except Exception, e:
                baset = [0] * 100
                print "word not found " + t[i]
            r1.append(suit(baset, v2))
    print r1, r2
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
    t1 = getWords(t1)
    t2 = getWords(t2)
    t = union(t1, t2)
    t = ["a", "brown", "jumps", "the", "fox", "dog", "quick", "over", "lazy"]
    print t

    # -------------- sementic similarity between two sentences --------------- #
    similarity = ssv(t, t1, t2)
    print similarity

    # ----------------- word similarity between sentences -------------------- #
    similarity = wo(t, t1, t2)
    print similarity


test()

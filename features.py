import re, word2vec
from dpss import ssv, wo, dp, flex, agreg, union, parse_text, getWords, intersection, getWordsX
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.translate import bleu_score

lmtzr = WordNetLemmatizer()

def lemma(t1):
    t = [str(lmtzr.lemmatize(each)) for each in t1]
    return t

def BLEU(t1,t2):
    precision = bleu_score.sentence_bleu([t1], t2)
    recall = bleu_score.sentence_bleu([t2], t1)
    return precision, recall

def unigrap_precision_recall(t1,t2):
    t = intersection(t1, t2)
    precision = float(len(t)) / float(len(t1))
    recall = float(len(t)) / float(len(t2))
    return precision, recall

def dp_precision_recall(d1,d2):
    for i in range(len(d1)):
        d1[i][1] = 'DEP'
        d1[i][3] = []
        d1[i][4] = []
    for i in range(len(d2)):
        d2[i][1] = 'DEP'
        d2[i][3] = []
        d2[i][4] = []
    d = intersection(d1, d2)
    cnt = len(d)
    precision = float(cnt) / len(d1)
    recall = float(cnt) / len(d2)
    return precision, recall

def dp_precision_recall_lemma(d1,d2):
    for i in range(len(d1)):
        d1[i][1] = 'DEP'
        d1[i][3] = []
        d1[i][4] = []
        d1[i][0] = str(lmtzr.lemmatize(d1[i][0]))
        d1[i][2] = str(lmtzr.lemmatize(d1[i][2]))
    for i in range(len(d2)):
        d2[i][1] = 'DEP'
        d2[i][3] = []
        d2[i][4] = []
        d2[i][0] = str(lmtzr.lemmatize(d2[i][0]))
        d2[i][2] = str(lmtzr.lemmatize(d2[i][2]))
    cnt = 0
    for i in range(len(d2)):
            if d2[i] in d1:
                cnt = cnt + 1
    d = intersection(d1, d2)
    cnt = len(d)
    precision = float(cnt) / len(d1)
    recall = float(cnt) / len(d2)
    return precision, recall

def parent():
    from spacy.en import English
    parser = English()
    t1 = "a quick brown dog jumps over the lazy fox"
    t1 = getWords(t1)
    #print lemma(t1)
    t2 = "a fast brown fox jumps over the lazy dog"
    t2 = getWords(t2)
    #t2 = "he is a brown fox"
    t2 = "jumps over the lazy fox is a quick brown dog"
    t1 = "Many consider Maradona as the best player in soccer history"
    t2 = "Maradona is one of the best soccer players"
    t1="The DVD-CCA then appealed decisions to the state Supreme Court albert. hdujhuju".lower()
    t2="The DVD CCA appealed that decision to the U.S. Supreme Court albert einstein bhijjnjd.".lower()
    sentence_1 = unicode(t1, "utf-8")
    p1, d1 = parse_text(parser, sentence_1, 1)
    sentence_2 = unicode(t2, "utf-8")
    p2, d2 = parse_text(parser, sentence_2, 1)
    diff1 = len(t1) - len(t2)
    t1 = getWords(t1)
    t2 = getWords(t2)
    t1 = flex(t1)
    t2 = flex(t2)
    t1_l = lemma(t1)
    t2_l = lemma(t2)
    t = union(t1, t2)

    print '--- unigram ---'
    prec, rec = unigrap_precision_recall(t1,t2)
    prec_lemma, rec_lemma = unigrap_precision_recall(t1_l, t2_l)
    print 'regular ', prec, rec
    print 'lemmaed ', prec_lemma, rec_lemma

    print '--- BLEU ---'
    prec, rec = BLEU(t1,t2)
    prec_lemma, rec_lemma = BLEU(t1_l, t2_l)
    print 'regular ', prec, rec
    print 'lemmaed ', prec_lemma, rec_lemma

    print '--- dependency based ---'
    prec, rec = dp_precision_recall(d1,d2)
    prec_lemma, rec_lemma = dp_precision_recall_lemma(d1,d2)
    print 'regular ', prec, rec
    print 'lemmaed ', prec_lemma, rec_lemma

    print '--- absolute ---'
    diff2 = len(t1) - len(t2)
    if diff1 < 0:
        diff1 = -1*diff1
    if diff2 < 0:
        diff2 = -1*diff2
    print diff1, diff2



#parent()
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
        t1_l = lemma(t1)
        t2_l = lemma(t2)
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
        # ---- unigram feartures ------------------------ #
        f1, f2 = unigrap_precision_recall(t1,t2)
        f3, f4 = unigrap_precision_recall(t1_l, t2_l)

        # ---- bleu features ------------------------ #
        f5, f6 = BLEU(t1,t2)
        f7, f8 = BLEU(t1_l, t2_l)

        # ---- dependency based features ------------------------ #
        f9, f10 = dp_precision_recall(d1,d2)
        f11, f12 = dp_precision_recall_lemma(d1,d2)

        # ---- absolute features ------------------------ #
        diff1 = len(s1) - len(s2)
        diff2 = len(t1) - len(t2)
        if diff1 < 0:
            diff1 = -1*diff1
        if diff2 < 0:
            diff2 = -1*diff2
        with open('testdata/features-output.txt','a') as f:
            f.write(str(similarity_ssv)+'\t'+str(similarity_wo)+'\t'+str(similarity_dp)+'\t'+str(similarity_dp_cnze)+'\t'+str(c1)
            +'\t'+str(f1)+'\t'+str(f2)+'\t'+str(f3)+'\t'+str(f4)+'\t'+str(f5)+'\t'+str(f6)+'\t'+str(f7)+'\t'+str(f8)+'\t'+str(f9)
            +'\t'+str(f10)+'\t'+str(f11)+'\t'+str(f12)+'\t'+str(diff1)+'\t'+str(diff2)+'\t'+str(block[i][0])+'\t'+str(block[i][1])+'\n')
        z = 0
        i=i+1

predict()

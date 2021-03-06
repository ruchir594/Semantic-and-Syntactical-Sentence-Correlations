import re, word2vec, numpy, pyter
from dpss import ssv, wo, dp, flex, agreg, union, parse_text, getWords, intersection, getWordsX, polish
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

def BLEU3(t1,t2):
    precision = bleu_score.sentence_bleu([t1], t2,  weights=(0.33333,0.33333,0.33333))
    recall = bleu_score.sentence_bleu([t2], t1,  weights=(0.33333,0.33333,0.33333))
    return precision, recall

def BLEU2(t1,t2):
    precision = bleu_score.sentence_bleu([t1], t2,  weights=(0.5, 0,5))
    recall = bleu_score.sentence_bleu([t2], t1,  weights=(0.5, 0.5))
    return precision, recall

def BLEU1(t1,t2):
    precision = bleu_score.sentence_bleu([t1], t2,  weights=(1,0))
    recall = bleu_score.sentence_bleu([t2], t1,  weights=(1,0))
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

def per(t1, t2):
    cnt = 0
    cnt2 = 0
    for each in t1:
        if each not in t2:
            cnt = cnt + 1
    for each in t2:
        if each not in t1:
            cnt2 = cnt2 + 1
    cnt = max(cnt, cnt2)
    return 1 - float(cnt)/float(max(len(t1), len(t2)))

def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    #build the matrix
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0: d[0][j] = j
            elif j == 0: d[i][0] = i
    for i in range(1,len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    result = float(d[len(r)][len(h)]) / max(len(r), len(h))
    return 1 - result

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

    print '--- BLEU3 ---'
    prec, rec = BLEU3(t1,t2)
    prec_lemma, rec_lemma = BLEU3(t1_l, t2_l)
    print 'regular ', prec, rec
    print 'lemmaed ', prec_lemma, rec_lemma

    print '--- BLEU2 ---'
    prec, rec = BLEU2(t1,t2)
    prec_lemma, rec_lemma = BLEU2(t1_l, t2_l)
    print 'regular ', prec, rec
    print 'lemmaed ', prec_lemma, rec_lemma

    print '--- BLEU1 ---'
    prec, rec = BLEU1(t1,t2)
    prec_lemma, rec_lemma = BLEU1(t1_l, t2_l)
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
        #t1 = polish(p1)
        #t2 = polish(p2)
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

        # ---- BLEU features ------------------------ #
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

        # ----- word error rate (WER) & position independent WER ---#
        f13 = wer(t1, t2)
        f14 = per(t1, t2)

        # ------ BLEU3 -----#
        f15, f16 = BLEU3(t1,t2)
        f17, f18 = BLEU3(t1_l, t2_l)

        # ------ BLEU2 -----#
        f19, f20 = BLEU2(t1,t2)
        f21, f22 = BLEU2(t1_l, t2_l)

        # ---- F score ----#
        f23 = 2*f5*f6/(f5+f6)

        # ---- TER ---- #
        f24 = pyter.ter(t1,t2)
        f25 = pyter.ter(t2,t1)
        f26 = pyter.ter(t1_l, t2_l)
        f27 = pyter.ter(t2_l, t1_l)

        with open('testdata/features-output.txt','a') as f:
            f.write(str(similarity_ssv)+'\t'+str(similarity_wo)+'\t'+str(similarity_dp)+'\t'+str(similarity_dp_cnze)+'\t'+str(c1)
            +'\t'+str(f1)+'\t'+str(f2)+'\t'+str(f3)+'\t'+str(f4)+'\t'+str(f5)+'\t'+str(f6)+'\t'+str(f7)+'\t'+str(f8)+'\t'+str(f9)
            +'\t'+str(f10)+'\t'+str(f11)+'\t'+str(f12)+'\t'+str(diff1)+'\t'+str(diff2)+'\t'+str(f13)+'\t'+str(f14)+'\t'+str(f15)
            +'\t'+str(f16)+'\t'+str(f17)+'\t'+str(f18)+'\t'+str(f19)+'\t'+str(f20)+'\t'+str(f21)+'\t'+str(f22)+'\t'+str(f23)
            +'\t'+str(f24)+'\t'+str(f25)+'\t'+str(f26)+'\t'+str(f27)+'\t'+str(block[i][0])+'\t'+str(block[i][1])+'\n')
        z = 0
        i=i+1

parent()
predict()

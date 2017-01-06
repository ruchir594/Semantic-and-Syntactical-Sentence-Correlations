from collections import Counter
def cross():
    print '-----test------'
    with open('test-pred-test-rf-1.txt') as f:
        zz = f.readlines()
    with open('test-pred-test-rf-2.txt') as f:
        zz2 = f.readlines()
    with open('test-pred-test-svm-1.txt') as f:
        zz3 = f.readlines()
    with open('test-pred-test-svm-2.txt') as f:
        zz4 = f.readlines()
    with open('test-pred-test-svm-3.txt') as f:
        zz5 = f.readlines()
    with open('test-pred-test-svm-4.txt') as f:
        zz6 = f.readlines()
    with open('test-pred-test-svm-5.txt') as f:
        zz7 = f.readlines()
    with open('test-pred-test-svm-6.txt') as f:
        zz8 = f.readlines()
    z = []
    vote = []
    for w1,w2,w3,w4,w5,w6,w7,w8 in zip(zz, zz2, zz3, zz4, zz5, zz6, zz7, zz8):
        c = Counter([w3,w8,w5,w6])
        value, count = c.most_common()[0]
        vote.append(value)
    zz = vote
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
cross()

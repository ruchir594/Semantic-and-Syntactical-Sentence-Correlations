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

import re

def get_postagging(parsedData):
    full_pos = []
    sent = []
    for span in parsedData.sents:
        sent = sent + [parsedData[i] for i in range(span.start, span.end)]
        #break

    for token in sent:
        full_pos.append([token.orth_, token.pos_])
    return full_pos

def get_dependency(parsedEx):
    # Let's look at the dependencies of this example:
    # shown as: original token, dependency tag, head word, left dependents, right dependents
    full_dep = []
    for token in parsedEx:
        full_dep.append([token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights]])
    return full_dep

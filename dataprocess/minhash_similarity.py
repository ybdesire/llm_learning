# reference
# https://ekzhu.com/datasketch/minhash.html
# https://charent.github.io/p/mini-hash%E6%96%87%E6%A1%A3%E5%8E%BB%E9%87%8D/


from datasketch import MinHash
# data should like this
data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']
# num_perm for accuracy
m1, m2 = MinHash(num_perm=128), MinHash(num_perm=128)
# update() to calculate minhash
for d in data1:
    m1.update(d.encode('utf8'))
for d in data2:
    m2.update(d.encode('utf8'))
# Estimated Jaccard similarity
print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))#0.7109375
# Actual Jaccard similarity
s1 = set(data1)
s2 = set(data2)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and data2 is", actual_jaccard)#0.7142857142857143
# print
h = m1.digest()
print(type(h))# <class 'numpy.ndarray'>
print(h.shape)#(128,)



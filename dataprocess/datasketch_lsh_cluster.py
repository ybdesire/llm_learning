# reference: https://charent.github.io/p/mini-hash%E6%96%87%E6%A1%A3%E5%8E%BB%E9%87%8D/

from datasketch import MinHashLSH

# data should like this
data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'a']
data3 = ['sdfa', 'asdf', 'dd', 'aa', 'ff', 'ss', 'a']

def getMinHash(data):
    m = MinHash(num_perm=128)
    for d in data:
        m.update(d.encode('utf8'))
    return m

data_lsh = MinHashLSH(threshold=0.7, num_perm=128) 
h1 = getMinHash(data1)
h2 = getMinHash(data2)
h3 = getMinHash(data3)
data_lsh.insert(0,h1)# insert h1 as cluster-0
data_lsh.insert(1,h3)# insert h3 as cluster-1

print(data_lsh.query(h2))# query h2 similarity as cluster-n?
# print: [0], i.e. h2 is more similar as cluster-0

from datasketch import MinHash, MinHashLSH

def ngrams(text, n=2):
    return [text[i:i+n] for i in range(len(text)-n+1)]

sss = ['a a a', 'b b', 'cc', 'd d', 'd d', 'd d']
result_no_dup = []
# init minhash
nperm = 128
lsh = MinHashLSH(threshold=0.4, num_perm=nperm)
# iteration all text
k = 0
for text in sss:
    k+=1
    # calc minhash
    minhash = MinHash(num_perm=nperm)
    for d in ngrams(text, 2):# 2-gram
        minhash.update(d.encode('utf-8'))
    unique_key = k
    if not lsh.query(minhash): # if not dup
        lsh.insert(unique_key, minhash)
        print('no-dup: {0}, minhash={1}'.format(text, minhash))
        result_no_dup.append(text)


'''
no-dup: a a a, minhash=<datasketch.minhash.MinHash object at 0x7f7ffc08f490>
no-dup: b b, minhash=<datasketch.minhash.MinHash object at 0x7f7ffc08d6d0>
no-dup: cc, minhash=<datasketch.minhash.MinHash object at 0x7f7e9381bf10>
no-dup: d d, minhash=<datasketch.minhash.MinHash object at 0x7f7ffc08d6d0>
'''


from FlagEmbedding import BGEM3FlagModel

# pip install FlagEmbedding
# download model  to folder `bge-m3` from : https://huggingface.co/BAAI/bge-m3

model = BGEM3FlagModel('bge-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
s = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."]
v = model.encode(s, batch_size=12, max_length=8192, )['dense_vecs']
print(v[0], len(v[0]), v.shape )



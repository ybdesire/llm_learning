from datasets import *
from transformers import *
from tokenizers import *
import os
import json
# reference: https://thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python



# download data
# ./hfd.sh cc_news --dataset --tool aria2c -x 4
# download from    https://hf-mirror.com/datasets/cc_news

# pip install datasets transformers sentencepiece




# load data
dataset = load_dataset("cc_news", split="train")

# split the dataset into training (90%) and testing (10%)
d = dataset.train_test_split(test_size=0.1)
print(d["train"], d["test"])


'''
Dataset({
    features: ['title', 'text', 'domain', 'date', 'description', 'url', 'image_url'],
    num_rows: 637416
}) Dataset({
    features: ['title', 'text', 'domain', 'date', 'description', 'url', 'image_url'],
    num_rows: 70825
})

'''

# print first 3 articles (Parse)
for t in d["train"]["text"][:3]:
    print(t)
    print("="*50)

'''
The Enugu Electricity Distribution Company (EEDC), ...
...
Eze urged the customers of EEDC not to delay in ...
==================================================
Kim Kardashian and Kanye West stepped out for Valentine's Day Tuesday.
Kim has been extra attentive toward her spouse and ...
==================================================
Words and Feathers
...in exchange for his sponsoring (and titling) this poem. 
while skydiving
==================================================

'''

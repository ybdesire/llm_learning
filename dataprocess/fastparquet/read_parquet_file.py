from fastparquet import ParquetFile
# reference: https://charent.github.io/p/parquet%E6%96%87%E4%BB%B6%E7%9A%84%E8%AF%BB%E5%86%99%E5%92%8C%E5%BE%AA%E7%8E%AF%E9%81%8D%E5%8E%86/
# data file from https://github.com/charent/ChatLM-mini-Chinese/tree/main/data
pf = ParquetFile('example.parquet')
for pf_chunk in pf:
    for rows in pf_chunk.iter_row_groups():
        for prompt, response in zip(rows['prompt'], rows['response']):
            pass
print(prompt)# What is your name?
print(response)# My name is Sam. Nice to meet you

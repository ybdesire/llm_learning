import re

# Keep Chinese, English and underline, no punctuation
rex = re.compile("[^[\u4E00-\u9FA5|A-Za-z_0-9]")
src = 'This is 是。不是Sdf#@4@#$__@3。中文，保留中文和英文、底線，不要標點符號。китайский'
doc = ''.join(rex.split(src))
print(doc)# Thisis是不是Sdf4__3中文保留中文和英文底線不要標點符號

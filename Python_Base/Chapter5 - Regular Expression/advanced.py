import re

# Data
report = '''
If you invested $1 in the year 1801, you would have $18087791.41 today.
This is a 7.967% return on investment. 
But if you invested only $0.25 in 1801, you would end up with $4521947.8525.
'''

# 可以把 [0-9] 换成 \d
pattern = r'(-?\$[0-9]+(\.[0-9]+)?)'

print(re.findall(pattern, report))
dollars = [x[0] for x in re.findall(pattern, report)]
print(dollars)

# find all valid http links
# Data
article = '''
The algorithm has important practical applications
http://blog.finxter.com/applications/
in many basic data structures such as sets, trees,
dictionaries, bags, bag trees, bag dictionaries,
hash sets, https://blog.finxter.com/sets-in-python/
hash tables, maps, and arrays. http://blog.finxter.com/
http://not-a-valid-url
http:/bla.ba.com
http://bo.bo.bo.bo.bo.bo/
http://bo.bo.bo.bo.bo.bo/333483--33343-/
'''

stale_links = re.findall(r'http://[0-9a-zA-Z_\-.]+\.[0-9a-zA-Z_\-/]+', article)
print(stale_links)

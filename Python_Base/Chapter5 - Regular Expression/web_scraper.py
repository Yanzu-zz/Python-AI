import re
import urllib.request as r

search_phrase = 'crypto'

# with r.urlopen('https://www.wired.com') as response:
#     html = response.read().decode("utf8")
#     first_pos = html.find(search_phrase)
#     print(html[first_pos - 10:first_pos + 10])

# Data
page = '''
<!DOCTYPE html>
<html>
<body>
<h1>My Programming Links</h1>
<a href="https://app.finxter.com/">test your Python skills</a>
<a href="https://blog.finxter.com/recursion/">Learn recursion</a>
<a href="https://nostarch.com/">Great books from NoStarchPress</a>
<a href="http://finxter.com/">Solve more Python puzzles</a>
</body>
</html>
'''


def find_hyperlinks_with_keywords(text):
    pattern = r'<a\s+(?:[^>]*?\s+)?href=(?:"|\')(http[s]?://finxter\.com[^"\']+)(?:"|\')[^>]*?>(?:.*?test.*?|.*?puzzle.*?)<\/a>'
    # pattern = r'<a\s+(?:[^>]*?\s+)?href=(?:"|\')(http[s]?://finxter\.com[^"\']+)(?:"|\')[^>]*?>(.*?)<\/a>'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return matches


res1 = find_hyperlinks_with_keywords(page)
print(res1)
res2 = re.findall("(<a.*?finxter.*?(test|puzzle).*?>)", page)
print(res2)

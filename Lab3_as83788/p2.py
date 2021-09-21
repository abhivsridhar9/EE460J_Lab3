import math
import os
from collections import Counter
from random import choices

import bs4
import numpy
import requests
import textract

# part a
url = "http://proceedings.mlr.press/v70/"

folder_location = r'p2files'
if not os.path.exists(folder_location):os.mkdir(folder_location)

response = requests.get(url)
soup= bs4.BeautifulSoup(response.text, "html.parser")

for link in soup.select("a[href$='.pdf']"):
    #Name the pdf files using the last portion of each link which are unique in this case
    filename = os.path.join(folder_location,link['href'].split('/')[-1])
    with open(filename, 'wb') as f:
        f.write(requests.get(urljoin(url,link['href'])).content)


text=''
for filename in os.listdir('p2files'):
    size = os.path.getsize('p2files/'+filename)
    if size>0:
        text+=textract.process('p2files/'+filename).decode('utf-8')

text=text.replace('\x00','')
text = text.split()


Counters_found = Counter(text)
most_occur = Counters_found.most_common(10)
print(most_occur)

# part 2
all_words = list(Counter(text).items())


total_word_count=sum(Counters_found.values())
entropy_sum=0

# entropy
for i in all_words:
    entropy_sum-= (i[1] / total_word_count) * math.log((i[1] / total_word_count), 2)

print(entropy_sum)

# part 3
words=[]
weights=[]

for i in Counters_found.keys():
    words.append(i)

for i in Counters_found.values():
    weights.append(i/total_word_count)

random_paragraph=''
for i in range(100):
    random_paragraph+=choices(words,weights)[0]
    random_paragraph+=' '

print(random_paragraph)

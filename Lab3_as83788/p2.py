import math
import os
from collections import Counter
import bs4
import requests
import textract
# part a

url = "http://proceedings.mlr.press/v70/"

folder_location = r'p2files'
if not os.path.exists(folder_location):os.mkdir(folder_location)

response = requests.get(url)
soup= bs4.BeautifulSoup(response.text, "html.parser")

# for link in soup.select("a[href$='.pdf']"):
#     #Name the pdf files using the last portion of each link which are unique in this case
#     filename = os.path.join(folder_location,link['href'].split('/')[-1])
#     with open(filename, 'wb') as f:
#         f.write(requests.get(urljoin(url,link['href'])).content)


text=''
for filename in os.listdir('p2files'):
    size = os.path.getsize('p2files/'+filename)
    if size>0:
        text+=textract.process('p2files/'+filename).decode('utf-8')

text=text.replace('\x00','')
text = text.split()


Counters_found = Counter(text)
most_occur = Counters_found.most_common(10)
all_words = list(Counter(text).items())


total_word_count=sum(Counters_found.values())
entropy_sum=0
# entropy
for i in all_words:
    entropy_sum+= (i[1] / total_word_count) * math.log((i[1] / total_word_count), 2)

entropy_sum=-entropy_sum

print(most_occur)
print(entropy_sum)

# entropy calculation




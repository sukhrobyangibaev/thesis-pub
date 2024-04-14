import json

from bs4 import BeautifulSoup
import requests


items_by_names = {}
item_id_winrate = {}
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
page_link = 'https://www.dotabuff.com/items/winning'

f = open('part_10_generate_train/items.json')
data = json.load(f)

for item in data.keys():
    print(item)
    if 'dname' in data[item]:
        items_by_names[data[item]['dname']] = data[item]['id']

page_response = requests.get(page_link, timeout=5, headers=headers)
soup = BeautifulSoup(page_response.content, "html.parser")
trs = soup.find_all('tr')
for tr in trs[1:]:
    item_name, winrate = tr.find_all('td')[1:3]
    item_id_winrate[items_by_names.get(item_name.a.string)] = float(winrate['data-value'])

print(item_id_winrate)
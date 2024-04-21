import json
import time

from bs4 import BeautifulSoup
import requests


print("Fetching hero winrates...")
heros_by_names = {}
heroid_winrate = {}
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
page_link = 'https://www.dotabuff.com/heroes/winning'

f = open('part_10_generate_train/heroes.json')
data = json.load(f)

for hero in data:
    heros_by_names[hero['localized_name']] = hero['id']

page_response = requests.get(page_link, timeout=5, headers=headers)
soup = BeautifulSoup(page_response.content, "html.parser")
trs = soup.find_all('tr')
for tr in trs[1:]:
    hero_name, winrate = tr.find_all('td')[1:3]
    heroid_winrate[heros_by_names.get(hero_name.a.string)] = float(winrate['data-value'])

print(heroid_winrate)
time.sleep(2)
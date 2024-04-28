import json
import time
from bs4 import BeautifulSoup
import numpy as np
import pymongo
import pandas as pd
from gensim.models import Word2Vec
import requests


MONGO_CLIENT = pymongo.MongoClient("mongodb://192.168.1.7:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
PT_COL = SDA_DB["predict_timelines"]

match_id = 7702817490

# ---------------------------
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
# ----------------------------------------
print("Fetching item winrates...")
items_by_names = {}
item_id_winrate = {}
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
page_link = 'https://www.dotabuff.com/items/winning'

f = open('part_10_generate_train/items.json')
data = json.load(f)

for item in data.keys():
    if 'dname' in data[item]:
        items_by_names[data[item]['dname']] = data[item]['id']

page_response = requests.get(page_link, timeout=5, headers=headers)
soup = BeautifulSoup(page_response.content, "html.parser")
trs = soup.find_all('tr')
for tr in trs[1:]:
    item_name, winrate = tr.find_all('td')[1:3]
    item_id_winrate[items_by_names.get(item_name.a.string)] = float(winrate['data-value'])
print(item_id_winrate)
time.sleep(2)

# Load the Word2Vec model
# heroes_model = Word2Vec.load("part_9_embedding/heroes/winner_hero_embeddings.model")
# items_model = Word2Vec.load("part_9_embedding/items/dota2_items_embeddings.model")

matches = PT_COL.find({"match_id": match_id})

matches_for_pd = []

for entry in matches:
    try:
        if "radiant" not in entry["scoreboard"]:
            continue

        tmp = {}

        tmp["duration"] = entry["scoreboard"]["duration"]

        tmp["radiant_series_wins"] = entry["radiant_series_wins"]
        tmp["dire_series_wins"] = entry["dire_series_wins"]

        tmp["score"] = (
            entry["scoreboard"]["radiant"]["score"]
            - entry["scoreboard"]["dire"]["score"]
        )

        rts = entry["scoreboard"]["radiant"]["tower_state"]
        for i, t in enumerate(format(rts, "b").zfill(11)):
            tmp[f"{i}_rts"] = t
        dts = entry["scoreboard"]["dire"]["tower_state"]
        for i, t in enumerate(format(dts, "b").zfill(11)):
            tmp[f"{i}_dts"] = t

        rbs = entry["scoreboard"]["radiant"]["barracks_state"]
        for i, t in enumerate(format(rbs, "b").zfill(6)):
            tmp[f"{i}_rbs"] = t
        dbs = entry["scoreboard"]["dire"]["barracks_state"]
        for i, t in enumerate(format(dbs, "b").zfill(6)):
            tmp[f"{i}_dbs"] = t

        radiant_net_worth = 0
        dire_net_worth = 0

        radiant_assissts = 0
        dire_assissts = 0

        radiant_last_hits = 0
        dire_last_hits = 0

        radiant_gold = 0
        dire_gold = 0

        radiant_level = 0
        dire_level = 0

        radiant_gpm = 0
        dire_gpm = 0

        radiant_xpm = 0
        dire_xpm = 0

        radiant_hero_winrates = []
        dire_hero_winrates = []

        radiant_item_winrates = []
        dire_item_winrates = []

        if len(entry["scoreboard"]["radiant"]["players"]) != 5:
            continue

        for i, player in enumerate(entry["scoreboard"]["radiant"]["players"]):
            radiant_item_winrates.append(item_id_winrate.get(player["item0"], 50))
            radiant_item_winrates.append(item_id_winrate.get(player["item1"], 50))
            radiant_item_winrates.append(item_id_winrate.get(player["item2"], 50))
            radiant_item_winrates.append(item_id_winrate.get(player["item3"], 50))
            radiant_item_winrates.append(item_id_winrate.get(player["item4"], 50))
            radiant_item_winrates.append(item_id_winrate.get(player["item5"], 50))

            radiant_hero_winrates.append(heroid_winrate.get(player["hero_id"], 50))

            radiant_net_worth += player["net_worth"]
            radiant_assissts += player["assists"]
            radiant_last_hits += player["last_hits"]
            radiant_gold += player["gold"]
            radiant_level += player["level"]
            radiant_gpm += player["gold_per_min"]
            radiant_xpm += player["xp_per_min"]

        if len(entry["scoreboard"]["dire"]["players"]) != 5:
            continue

        for i, player in enumerate(entry["scoreboard"]["dire"]["players"]):
            dire_item_winrates.append(item_id_winrate.get(player["item0"], 50))
            dire_item_winrates.append(item_id_winrate.get(player["item1"], 50))
            dire_item_winrates.append(item_id_winrate.get(player["item2"], 50))
            dire_item_winrates.append(item_id_winrate.get(player["item3"], 50))
            dire_item_winrates.append(item_id_winrate.get(player["item4"], 50))
            dire_item_winrates.append(item_id_winrate.get(player["item5"], 50))

            dire_hero_winrates.append(heroid_winrate.get(player["hero_id"], 50))

            dire_net_worth += player["net_worth"]
            dire_assissts += player["assists"]
            dire_last_hits += player["last_hits"]
            dire_gold += player["gold"]
            dire_level += player["level"]
            dire_gpm += player["gold_per_min"]
            dire_xpm += player["xp_per_min"]


        tmp['hero_winrate_diff'] = np.mean(radiant_hero_winrates) - np.mean(dire_hero_winrates)
        tmp['item_winrate_diff'] = np.mean(radiant_item_winrates) - np.mean(dire_item_winrates)

        tmp["net_worth"] = radiant_net_worth - dire_net_worth
        tmp["assissts"] = radiant_assissts - dire_assissts
        tmp["last_hits"] = radiant_last_hits - dire_last_hits
        tmp["gold"] = radiant_gold - dire_gold
        tmp["level"] = radiant_level - dire_level
        tmp["gpm"] = radiant_gpm - dire_gpm
        tmp["xpm"] = radiant_xpm - dire_xpm

        tmp["winner"] = entry["winner"]

        matches_for_pd.append(tmp)
    except Exception as e:
        print(e, entry["match_id"])

df = pd.DataFrame(matches_for_pd)
print("shape:", df.shape)

df = df.dropna()
print("shape after dropna:", df.shape)

df = df.drop_duplicates()
print("shape after drop_duplicates:", df.shape)

df.to_csv(
    f"various_experiments/predict_in_timelines/results/{match_id}/{df.shape[0]}x{df.shape[1]}_samples.csv",
    index=False,
)

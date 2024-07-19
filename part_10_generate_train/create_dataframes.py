import json
import time
from bs4 import BeautifulSoup
import numpy as np
import pymongo
import pandas as pd
# from gensim.models import Word2Vec
import requests

# Start the timer
start_time = time.time()

MONGO_CLIENT = pymongo.MongoClient("mongodb://localhost:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
LGC_COL = SDA_DB["league_games_col"]

# ---------------------------
# print("Fetching hero winrates...")
# heros_by_names = {}
# heroid_winrate = {}
# headers = {
#     'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
# page_link = 'https://www.dotabuff.com/heroes/winning'

# f = open('part_10_generate_train/heroes.json')
# data = json.load(f)

# for hero in data:
#     heros_by_names[hero['localized_name']] = hero['id']

# page_response = requests.get(page_link, timeout=5, headers=headers)
# soup = BeautifulSoup(page_response.content, "html.parser")
# trs = soup.find_all('tr')
# for tr in trs[1:]:
#     hero_name, winrate = tr.find_all('td')[0:2]
#     heroid_winrate[heros_by_names.get(hero_name.get_text())] = float(winrate.get_text()[:-1])

# print(heroid_winrate)
# time.sleep(2)
# ----------------------------------------
# print("Fetching item winrates...")
# items_by_names = {}
# item_id_winrate = {}
# headers = {
#     'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'}
# page_link = 'https://www.dotabuff.com/items/winning'

# f = open('part_10_generate_train/items.json')
# data = json.load(f)

# for item in data.keys():
#     if 'dname' in data[item]:
#         items_by_names[data[item]['dname']] = data[item]['id']

# page_response = requests.get(page_link, timeout=5, headers=headers)
# soup = BeautifulSoup(page_response.content, "html.parser")
# trs = soup.find_all('tr')
# for tr in trs[1:]:
#     item_name, winrate = tr.find_all('td')[1:3]
#     item_id_winrate[items_by_names.get(item_name.a.string)] = float(winrate['data-value'])
# print(item_id_winrate)
# time.sleep(2)
# ----------------------------------------

# print("Loading embedding models...")
# heroes_model = Word2Vec.load("trained_models/embedding_heroes.model")
# items_model = Word2Vec.load("trained_models/embedding_items.model")


print("Fetching matches...")
min_10 = LGC_COL.find(
    {"winner": {"$exists": True}, "scoreboard.duration": {"$lte": 600}}
)
min_20 = LGC_COL.find(
    {"winner": {"$exists": True}, "scoreboard.duration": {"$gt": 600, "$lte": 1200}}
)
min_30 = LGC_COL.find(
    {"winner": {"$exists": True}, "scoreboard.duration": {"$gt": 1200}}
)

filenames = ["10min", "20min", "30min"]
timelines = [min_10, min_20, min_30]

print("Processing matches...")
for i_timeline, timeline in enumerate(timelines):
    matches_for_pd = []
    print(f"Processing {filenames[i_timeline]} matches...")
    for entry in timeline:
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
                # radiant_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item0"])])
                # )
                # radiant_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item1"])])
                # )
                # radiant_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item2"])])
                # )
                # radiant_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item3"])])
                # )
                # radiant_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item4"])])
                # )
                # radiant_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item5"])])
                # )

                # radiant_hero_embeddings.append(
                #     np.mean(heroes_model.wv[str(player["hero_id"])])
                # )
                # radiant_item_winrates.append(item_id_winrate.get(player["item0"], 50))
                # radiant_item_winrates.append(item_id_winrate.get(player["item1"], 50))
                # radiant_item_winrates.append(item_id_winrate.get(player["item2"], 50))
                # radiant_item_winrates.append(item_id_winrate.get(player["item3"], 50))
                # radiant_item_winrates.append(item_id_winrate.get(player["item4"], 50))
                # radiant_item_winrates.append(item_id_winrate.get(player["item5"], 50))

                # radiant_hero_winrates.append(heroid_winrate.get(player["hero_id"], 50))

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
                # dire_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item0"])])
                # )
                # dire_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item1"])])
                # )
                # dire_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item2"])])
                # )
                # dire_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item3"])])
                # )
                # dire_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item4"])])
                # )
                # dire_item_embeddings.append(
                #     np.mean(items_model.wv[str(player["item5"])])
                # )

                # dire_hero_embeddings.append(
                #     np.mean(heroes_model.wv[str(player["hero_id"])])
                # )
                # dire_item_winrates.append(item_id_winrate.get(player["item0"], 50))
                # dire_item_winrates.append(item_id_winrate.get(player["item1"], 50))
                # dire_item_winrates.append(item_id_winrate.get(player["item2"], 50))
                # dire_item_winrates.append(item_id_winrate.get(player["item3"], 50))
                # dire_item_winrates.append(item_id_winrate.get(player["item4"], 50))
                # dire_item_winrates.append(item_id_winrate.get(player["item5"], 50))

                # dire_hero_winrates.append(heroid_winrate.get(player["hero_id"], 50))

                dire_net_worth += player["net_worth"]
                dire_assissts += player["assists"]
                dire_last_hits += player["last_hits"]
                dire_gold += player["gold"]
                dire_level += player["level"]
                dire_gpm += player["gold_per_min"]
                dire_xpm += player["xp_per_min"]
            
            # np.random.shuffle(radiant_item_embeddings)
            # for i, item_embedding in enumerate(radiant_item_embeddings):
            #     tmp[f"radiant_item_{i}_embedding"] = item_embedding

            # np.random.shuffle(dire_item_embeddings)
            # for i, item_embedding in enumerate(radiant_item_embeddings):
            #     tmp[f"dire_item_{i}_embedding"] = item_embedding

            # tmp["radiant_item_embeddings"] = np.mean(radiant_item_embeddings)
            # tmp["dire_item_embeddings"] = np.mean(dire_item_embeddings)

            # np.random.shuffle(radiant_hero_embeddings)
            # for i, item_embedding in enumerate(radiant_hero_embeddings):
            #     tmp[f"radiant_hero_{i}_embedding"] = item_embedding

            # np.random.shuffle(dire_hero_embeddings)
            # for i, item_embedding in enumerate(dire_hero_embeddings):
            #     tmp[f"dire_hero_{i}_embedding"] = item_embedding

            # tmp["radiant_embeddings"] = np.mean(radiant_hero_embeddings)
            # tmp["dire_embeddings"] = np.mean(dire_hero_embeddings)

            # tmp['radiant_hero_winrates'] = np.mean(radiant_hero_winrates)
            # tmp['dire_hero_winrates'] = np.mean(dire_hero_winrates)

            # tmp['hero_winrate_diff'] = np.mean(radiant_hero_winrates) - np.mean(dire_hero_winrates)
            # tmp['item_winrate_diff'] = np.mean(radiant_item_winrates) - np.mean(dire_item_winrates)

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
            # time.sleep(2)

    print("Creating DataFrame for", filenames[i_timeline])
    df = pd.DataFrame(matches_for_pd)
    print("shape:", df.shape)

    df = df.dropna()
    print("shape after dropna:", df.shape)

    df = df.drop_duplicates()
    print("shape after drop_duplicates:", df.shape)

    df.to_csv(
        f"dataframes/{filenames[i_timeline]}_{df.shape[0]}x{df.shape[1]}.csv",
        index=False,
    )

# End the timer and calculate the execution time
end_time = time.time()
execution_time = end_time - start_time

# Convert execution time to minutes and seconds
minutes, seconds = divmod(execution_time, 60)

# Print the execution time in minutes and seconds
print(f"Execution time: {int(minutes)} minutes and {seconds} seconds")
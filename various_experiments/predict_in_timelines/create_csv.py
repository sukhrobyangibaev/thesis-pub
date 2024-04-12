import numpy as np
import pymongo
import pandas as pd
from gensim.models import Word2Vec


MONGO_CLIENT = pymongo.MongoClient("mongodb://192.168.1.7:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
PT_COL = SDA_DB["predict_timelines"]

# Load the Word2Vec model
heroes_model = Word2Vec.load("part_9_embedding/heroes/winner_hero_embeddings.model")
items_model = Word2Vec.load("part_9_embedding/items/dota2_items_embeddings.model")

matches = PT_COL.find()

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

        radiant_hero_embeddings = []
        dire_hero_embeddings = []

        radiant_item_embeddings = []
        dire_item_embeddings = []

        if len(entry["scoreboard"]["radiant"]["players"]) != 5:
            continue

        for i, player in enumerate(entry["scoreboard"]["radiant"]["players"]):
            radiant_item_embeddings.append(np.mean(items_model.wv[str(player["item0"])]))
            radiant_item_embeddings.append(np.mean(items_model.wv[str(player["item1"])]))
            radiant_item_embeddings.append(np.mean(items_model.wv[str(player["item2"])]))
            radiant_item_embeddings.append(np.mean(items_model.wv[str(player["item3"])]))
            radiant_item_embeddings.append(np.mean(items_model.wv[str(player["item4"])]))
            radiant_item_embeddings.append(np.mean(items_model.wv[str(player["item5"])]))

            radiant_hero_embeddings.append(np.mean(heroes_model.wv[str(player["hero_id"])]))

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
            dire_item_embeddings.append(np.mean(items_model.wv[str(player["item0"])]))
            dire_item_embeddings.append(np.mean(items_model.wv[str(player["item1"])]))
            dire_item_embeddings.append(np.mean(items_model.wv[str(player["item2"])]))
            dire_item_embeddings.append(np.mean(items_model.wv[str(player["item3"])]))
            dire_item_embeddings.append(np.mean(items_model.wv[str(player["item4"])]))
            dire_item_embeddings.append(np.mean(items_model.wv[str(player["item5"])]))

            dire_hero_embeddings.append(np.mean(heroes_model.wv[str(player["hero_id"])]))

            dire_net_worth += player["net_worth"]
            dire_assissts += player["assists"]
            dire_last_hits += player["last_hits"]
            dire_gold += player["gold"]
            dire_level += player["level"]
            dire_gpm += player["gold_per_min"]
            dire_xpm += player["xp_per_min"]


        tmp['radiant_item_embeddings'] = np.mean(radiant_item_embeddings)
        tmp['dire_item_embeddings'] = np.mean(dire_item_embeddings)

        tmp['radiant_embeddings'] = np.mean(radiant_hero_embeddings)
        tmp['dire_embeddings'] = np.mean(dire_hero_embeddings)

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
    f"various_experiments/predict_in_timelines/{df.shape[0]}x{df.shape[1]}_samples.csv",
    index=False,
)

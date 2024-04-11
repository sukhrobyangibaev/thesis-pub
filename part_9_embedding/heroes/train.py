from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import pymongo
import pandas as pd

MONGO_CLIENT = pymongo.MongoClient("mongodb://192.168.1.7:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
LGC_COL = SDA_DB["league_games_col"]

unique_match_ids = LGC_COL.distinct("match_id")
print('Number of unique matches:', len(unique_match_ids))

matchups = []

print("Processing matches...")
for match_id in unique_match_ids:
    match = LGC_COL.find_one({"match_id": match_id})
    radiant_matchup = []
    dire_matchup = []
    
    for i, player in enumerate(match["scoreboard"]["radiant"]["players"]):
        radiant_matchup.append(str(player["hero_id"]))

    for i, player in enumerate(match["scoreboard"]["dire"]["players"]):
        dire_matchup.append(str(player["hero_id"]))

    matchups.append(radiant_matchup)
    matchups.append(dire_matchup)

# Train a Word2Vec model
print("Training Word2Vec model...")
model = Word2Vec(matchups, min_count=1, vector_size=100, window=5)

# Save the model for later use
print("Saving model...")
model.save("dota2_hero_embeddings.model")
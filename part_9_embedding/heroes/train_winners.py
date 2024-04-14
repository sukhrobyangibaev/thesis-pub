from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import pymongo
import pandas as pd

MONGO_CLIENT = pymongo.MongoClient("mongodb://192.168.1.7:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
LGC_COL = SDA_DB["league_games_col"]

LGC_COL.create_index("match_id")

unique_match_ids = LGC_COL.distinct("match_id")
print('Number of unique matches:', len(unique_match_ids))

# Fetch all matches at once based on unique_match_ids
matches_cursor = LGC_COL.find({"match_id": {"$in": unique_match_ids}})

matchups = []

print("Processing matches...")
for match in matches_cursor:
    matchup = []

    if match['winner'] == 'radiant':
        for player in match["scoreboard"]["radiant"]["players"]:
            matchup.append(str(player["hero_id"]))
    else:
        for player in match["scoreboard"]["dire"]["players"]:
            matchup.append(str(player["hero_id"]))

    matchups.append(matchup)

# Train a Word2Vec model
print("Training Word2Vec model...")
model = Word2Vec(matchups, min_count=1, vector_size=100, window=5)

# Save the model for later use
print("Saving model...")
model.save("part_9_embedding/heroes/winner_hero_embeddings.model")
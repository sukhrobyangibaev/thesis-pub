from gensim.models import Word2Vec
import pymongo

MONGO_CLIENT = pymongo.MongoClient("mongodb://192.168.1.7:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
LGC_COL = SDA_DB["league_games_col"]

LGC_COL.create_index("match_id")

unique_match_ids = LGC_COL.distinct("match_id")
print("Number of unique matches:", len(unique_match_ids))

print("Fetching matches...")
matches_cursor = LGC_COL.find({"match_id": {"$in": unique_match_ids}})

matchups = []

print("Processing matches...")
for match in matches_cursor:
    if match["winner"] == "radiant":
        matchup = [
            str(player["hero_id"])
            for player in match["scoreboard"]["radiant"]["players"]
        ]
    else:
        matchup = [
            str(player["hero_id"]) for player in match["scoreboard"]["dire"]["players"]
        ]

    matchups.append(matchup)

print("Training Word2Vec model...")
model = Word2Vec(matchups, min_count=1, vector_size=100, window=5)

print("Saving model...")
model.save("trained_models/embedding_heroes.model")

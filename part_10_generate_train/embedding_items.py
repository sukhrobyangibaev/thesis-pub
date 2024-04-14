from gensim.models import Word2Vec
import pymongo

MONGO_CLIENT = pymongo.MongoClient("mongodb://192.168.1.7:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
LGC_COL = SDA_DB["league_games_col"]


def append_items(player, item_set):
    try:
        item_set.extend(str(player[f"item{i}"]) for i in range(6) if f"item{i}" in player)
    except Exception as e:
        print(e)


item_set_list = []
counter = 0

print("Processing matches...")
for i in range(0, 6001, 300):
    matches = LGC_COL.find({"scoreboard.duration": {"$gt": i, "$lte": i + 55}})
    for match in matches:
        counter += 1
        item_set = []

        if match["winner"] == "radiant":
            for player in match["scoreboard"]["radiant"]["players"]:
                append_items(player, item_set)
        else:
            for player in match["scoreboard"]["dire"]["players"]:
                append_items(player, item_set)

        item_set_list.append(item_set)

print(f"Processed {counter} matches.")

print("Training Word2Vec model...")
model = Word2Vec(item_set_list, min_count=1, vector_size=100, window=5)

print("Saving model...")
model.save("trained_models/embedding_items.model")

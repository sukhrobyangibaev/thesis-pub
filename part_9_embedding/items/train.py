from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import pymongo
import pandas as pd

MONGO_CLIENT = pymongo.MongoClient("mongodb://192.168.1.7:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
LGC_COL = SDA_DB["league_games_col"]


item_set_list = []
counter = 0

print("Processing matches...")
for i in range(300, 3601, 300):
    matches = LGC_COL.find(
        {"scoreboard.duration": {"$gt": i, "$lte": i + 100}}
    )
    for match in matches:
        counter += 1
        item_set = []
        
        if match['winner'] == 'radiant':
            for i, player in enumerate(match["scoreboard"]["radiant"]["players"]):
                item_set.append(str(player['item0']))
                item_set.append(str(player['item1']))
                item_set.append(str(player['item2']))
                item_set.append(str(player['item3']))
                item_set.append(str(player['item4']))
                item_set.append(str(player['item5']))            
        else:
            for i, player in enumerate(match["scoreboard"]["dire"]["players"]):
                item_set.append(str(player['item0']))
                item_set.append(str(player['item1']))
                item_set.append(str(player['item2']))
                item_set.append(str(player['item3']))
                item_set.append(str(player['item4']))
                item_set.append(str(player['item5']))

        item_set_list.append(item_set)

print(f"Processed {counter} matches.")

# Train a Word2Vec model
print("Training Word2Vec model...")
model = Word2Vec(item_set_list, min_count=1, vector_size=100, window=5)

# Save the model for later use
print("Saving model...")
model.save("part_9_embedding/items/dota2_items_embeddings.model")
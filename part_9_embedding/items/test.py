from gensim.models import Word2Vec

# Load the model from the file
model = Word2Vec.load("part_9_embedding/items/dota2_items_embeddings.model")

# Use the model
# Get the vector for a specific item_id
item_id = "6"
vector = model.wv[item_id]
print(f"Vector for item_id {item_id}: {vector}")

# Find the most similar item_ids to a given item_id
similar_items = model.wv.most_similar(item_id)
print(f"Item_ids most similar to {item_id}: {similar_items}")

# Find the similarity between two item_ids
item_id1 = "1"
item_id2 = "2"
similarity = model.wv.similarity(item_id1, item_id2)
print(f"Similarity between item_id {item_id1} and item_id {item_id2}: {similarity}")

# TODO: Use the model to find the most similar items to a list of items
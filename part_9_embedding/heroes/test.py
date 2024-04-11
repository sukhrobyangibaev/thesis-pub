from gensim.models import Word2Vec

# Load the model from the file
model = Word2Vec.load("dota2_hero_embeddings.model")

# Use the model
# Get the vector for a specific hero_id
hero_id = "1"
vector = model.wv[hero_id]
print(f"Vector for hero_id {hero_id}: {vector}")

# Find the most similar hero_ids to a given hero_id
similar_heroes = model.wv.most_similar(hero_id)
print(f"Hero_ids most similar to {hero_id}: {similar_heroes}")

# Find the similarity between two hero_ids
hero_id1 = "1"
hero_id2 = "2"
similarity = model.wv.similarity(hero_id1, hero_id2)
print(f"Similarity between hero_id {hero_id1} and hero_id {hero_id2}: {similarity}")
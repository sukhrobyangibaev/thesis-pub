import pymongo
import pandas as pd

MONGO_CLIENT = pymongo.MongoClient("mongodb://192.168.1.7:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
LGC_COL = SDA_DB["league_games_col"]

matches_for_pd = []

matches = LGC_COL.find({"winner": {"$exists": True}})

for entry in matches:
    try:
        tmp = {}

        tmp['duration'] = entry['scoreboard']['duration']
        tmp['radiant_series_wins'] = entry['radiant_series_wins']
        tmp['dire_series_wins'] = entry['dire_series_wins']

        tmp['score'] = entry["scoreboard"]['radiant']["score"] - entry["scoreboard"]['dire']["score"]
        tmp['tower_state'] = entry["scoreboard"]['radiant']["tower_state"] - entry["scoreboard"]['dire']["tower_state"]
        tmp['barracks_state'] = entry["scoreboard"]['radiant']["barracks_state"] - entry["scoreboard"]['dire']["barracks_state"]
        

        radiant_net_worth = 0
        dire_net_worth = 0

        if 'radiant' not in entry["scoreboard"]:
            continue

        for i, player in enumerate(entry["scoreboard"]['radiant']['players']):
            tmp[f'radiant_player_{i}_kills'] = player['kills']
            tmp[f'radiant_player_{i}_death'] = player['death']
            tmp[f'radiant_player_{i}_assists'] = player['assists']
            tmp[f'radiant_player_{i}_last_hits'] = player['last_hits']
            tmp[f'radiant_player_{i}_gold'] = player['gold']
            tmp[f'radiant_player_{i}_level'] = player['level']
            tmp[f'radiant_player_{i}_gold_per_min'] = player['gold_per_min']
            tmp[f'radiant_player_{i}_xp_per_min'] = player['xp_per_min']
            tmp[f'radiant_player_{i}_item0'] = player['item0']
            tmp[f'radiant_player_{i}_item1'] = player['item1']
            tmp[f'radiant_player_{i}_item2'] = player['item2']
            tmp[f'radiant_player_{i}_item3'] = player['item3']
            tmp[f'radiant_player_{i}_item4'] = player['item4']
            tmp[f'radiant_player_{i}_item5'] = player['item5']
            tmp[f'radiant_player_{i}_net_worth'] = player['net_worth']

            radiant_net_worth += player['net_worth']

        for i, player in enumerate(entry["scoreboard"]['dire']['players']):
            i = player['player_slot']
            tmp[f'dire_player_{i}_kills'] = player['kills']
            tmp[f'dire_player_{i}_death'] = player['death']
            tmp[f'dire_player_{i}_assists'] = player['assists']
            tmp[f'dire_player_{i}_last_hits'] = player['last_hits']
            tmp[f'dire_player_{i}_gold'] = player['gold']
            tmp[f'dire_player_{i}_level'] = player['level']
            tmp[f'dire_player_{i}_gold_per_min'] = player['gold_per_min']
            tmp[f'dire_player_{i}_xp_per_min'] = player['xp_per_min']
            tmp[f'dire_player_{i}_item0'] = player['item0']
            tmp[f'dire_player_{i}_item1'] = player['item1']
            tmp[f'dire_player_{i}_item2'] = player['item2']
            tmp[f'dire_player_{i}_item3'] = player['item3']
            tmp[f'dire_player_{i}_item4'] = player['item4']
            tmp[f'dire_player_{i}_item5'] = player['item5']
            tmp[f'dire_player_{i}_net_worth'] = player['net_worth']

            dire_net_worth += player['net_worth']

        tmp['net_worth'] = radiant_net_worth - dire_net_worth

        tmp["winner"] = entry["winner"]

        matches_for_pd.append(tmp)
    except Exception as e:
        print(e, entry['match']['match_id'])

df = pd.DataFrame(matches_for_pd)
print('shape:', df.shape)

df = df.dropna()
print('shape after dropna:', df.shape)

df = df.drop_duplicates()
print('shape after drop_duplicates:', df.shape)

df.to_csv(f'{df.shape[0]}x{df.shape[1]}_samples.csv', index=False)
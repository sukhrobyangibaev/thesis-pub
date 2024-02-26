import pymongo
import pandas as pd

MONGO_CLIENT = pymongo.MongoClient("mongodb://192.168.1.7:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
LGC_COL = SDA_DB["league_games_col"]

matches_for_pd = []

# matches = LGC_COL.find({"winner": {"$exists": True}, 'scoreboard.duration': {"$gt": 600, "$lte": 1200}})
matches = LGC_COL.find({"winner": {"$exists": True}, 'scoreboard.duration': {"$gt": 1200}})

for entry in matches:
    try:
        if 'radiant' not in entry["scoreboard"]:
            continue

        tmp = {}

        tmp['duration'] = entry['scoreboard']['duration']

        tmp['radiant_series_wins'] = entry['radiant_series_wins']
        tmp['dire_series_wins'] = entry['dire_series_wins']

        tmp['score'] = entry["scoreboard"]['radiant']["score"] - entry["scoreboard"]['dire']["score"]

        rts = entry["scoreboard"]['radiant']["tower_state"]
        for i, t in enumerate(format(rts, 'b').zfill(11)):
            tmp[f'{i}_rts'] = t
        dts = entry["scoreboard"]['dire']["tower_state"]
        for i, t in enumerate(format(dts, 'b').zfill(11)):
            tmp[f'{i}_dts'] = t

        rbs = entry["scoreboard"]['radiant']["barracks_state"]
        for i, t in enumerate(format(rbs, 'b').zfill(6)):
            tmp[f'{i}_rbs'] = t
        dbs = entry["scoreboard"]['dire']["barracks_state"]
        for i, t in enumerate(format(dbs, 'b').zfill(6)):
            tmp[f'{i}_dbs'] = t

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

        for i, player in enumerate(entry["scoreboard"]['radiant']['players']):
            # tmp[f'radiant_player_{i}_kills'] = player['kills']
            # tmp[f'radiant_player_{i}_death'] = player['death']
            # tmp[f'radiant_player_{i}_assists'] = player['assists']
            # tmp[f'radiant_player_{i}_last_hits'] = player['last_hits']
            # tmp[f'radiant_player_{i}_gold'] = player['gold']
            # tmp[f'radiant_player_{i}_level'] = player['level']
            # tmp[f'radiant_player_{i}_gold_per_min'] = player['gold_per_min']
            # tmp[f'radiant_player_{i}_xp_per_min'] = player['xp_per_min']
            # tmp[f'radiant_player_{i}_net_worth'] = player['net_worth']
            # tmp[f'radiant_player_{i}_item0'] = player['item0']
            # tmp[f'radiant_player_{i}_item1'] = player['item1']
            # tmp[f'radiant_player_{i}_item2'] = player['item2']
            # tmp[f'radiant_player_{i}_item3'] = player['item3']
            # tmp[f'radiant_player_{i}_item4'] = player['item4']
            # tmp[f'radiant_player_{i}_item5'] = player['item5']

            radiant_net_worth += player['net_worth']
            radiant_assissts += player['assists']
            radiant_last_hits += player['last_hits']
            radiant_gold += player['gold']
            radiant_level += player['level']
            radiant_gpm += player['gold_per_min']
            radiant_xpm += player['xp_per_min']

        for i, player in enumerate(entry["scoreboard"]['dire']['players']):
            # tmp[f'dire_player_{i}_kills'] = player['kills']
            # tmp[f'dire_player_{i}_death'] = player['death']
            # tmp[f'dire_player_{i}_assists'] = player['assists']
            # tmp[f'dire_player_{i}_last_hits'] = player['last_hits']
            # tmp[f'dire_player_{i}_gold'] = player['gold']
            # tmp[f'dire_player_{i}_level'] = player['level']
            # tmp[f'dire_player_{i}_gold_per_min'] = player['gold_per_min']
            # tmp[f'dire_player_{i}_xp_per_min'] = player['xp_per_min']
            # tmp[f'dire_player_{i}_net_worth'] = player['net_worth']
            # tmp[f'dire_player_{i}_item0'] = player['item0']
            # tmp[f'dire_player_{i}_item1'] = player['item1']
            # tmp[f'dire_player_{i}_item2'] = player['item2']
            # tmp[f'dire_player_{i}_item3'] = player['item3']
            # tmp[f'dire_player_{i}_item4'] = player['item4']
            # tmp[f'dire_player_{i}_item5'] = player['item5']

            dire_net_worth += player['net_worth']
            dire_assissts += player['assists']
            dire_last_hits += player['last_hits']
            dire_gold += player['gold']
            dire_level += player['level']
            dire_gpm += player['gold_per_min']
            dire_xpm += player['xp_per_min']

        tmp['net_worth'] = radiant_net_worth - dire_net_worth
        tmp['assissts'] = radiant_assissts - dire_assissts
        tmp['last_hits'] = radiant_last_hits - dire_last_hits
        tmp['gold'] = radiant_gold - dire_gold
        tmp['level'] = radiant_level - dire_level
        tmp['gpm'] = radiant_gpm - dire_gpm
        tmp['xpm'] = radiant_xpm - dire_xpm

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

df.to_csv(f'part_7/no_items/30min_{df.shape[0]}x{df.shape[1]}_samples.csv', index=False)
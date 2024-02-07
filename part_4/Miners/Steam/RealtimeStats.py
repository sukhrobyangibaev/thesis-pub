import requests
import time
import pymongo
import logging
import sys
from random import randint
from dotenv import load_dotenv
import os

load_dotenv()

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
steam_dota_api_db = mongo_client["steam_dota_api"]
real_time_stats_col = steam_dota_api_db["multi_real_time_stats"]
league_games_col = steam_dota_api_db["league_games_col"]

TOP_LIVE_GAMES_URL = 'https://api.steampowered.com/IDOTA2Match_570/GetTopLiveGame/v1/'
REAL_TIME_STATS_URL = 'https://api.steampowered.com/IDOTA2MatchStats_570/GetRealtimeStats/v1'
LEAGUE_GAMES_URL = 'https://api.steampowered.com/IDOTA2Match_570/getLiveLeagueGames/v1/'

STEAM_KEY = os.getenv("STEAM_KEY")

logging.basicConfig(format="%(asctime)s > %(levelname)s > %(message)s",
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

request_counter = 0
games_observed = []

while True:
    # try:
    #     request_counter += 1
    #     response = requests.get(LEAGUE_GAMES_URL, params={"key": STEAM_KEY})
    # except requests.exceptions.ConnectionError as e:
    #     logger.exception(e)
    #     time.sleep(1)
    #     continue
    # if response.status_code != 200:
    #     logger.warning('LEAGUE_GAMES_URL, status code: {}, content: {}'.format(
    #             response.status_code, response.content))
    #     continue
    
    # res_json = response.json()

    # if not res_json:
    #     logger.warning(
    #         'LEAGUE_GAMES_URL, content: {}'.format(response.content))
    #     continue

    # for game in res_json['result']['games']:
    #     if 'scoreboard' not in game or game['scoreboard']['duration'] == 0:
    #         continue
    #     league_games_col.insert_one(game)

    #     logger.info('added {} League Match, Duration  {}'.format(
    #             game['match_id'],
    #             game['scoreboard']['duration']))

    for partner in range(2):
        time.sleep(1)
        try:
            request_counter += 1
            response = requests.get(TOP_LIVE_GAMES_URL, params={
                                    'key': STEAM_KEY, 'partner': partner})

        except requests.exceptions.ConnectionError as e:
            logger.exception(e)
            time.sleep(5)
            continue

        if response.status_code != 200:
            logger.warning('TOP_LIVE_GAMES_URL, status code: {}, partner: {}, content: {}'.format(
                response.status_code, partner, response.content))
            continue

        res_json = response.json()

        if not res_json:
            logger.warning(
                'TOP_LIVE_GAMES_URL, content: {}'.format(response.content))
            continue

        for game in res_json['game_list']:
            if game['game_time'] < 100:
                continue

            cur_game = {
                'match_id': game['match_id'],
                'server_steam_id': game['server_steam_id']}

            if cur_game in games_observed:
                continue

            time.sleep(1)
            try:
                request_counter += 1
                response = requests.get(
                    REAL_TIME_STATS_URL, params={'key': STEAM_KEY,
                                                 'server_steam_id': cur_game['server_steam_id']})
            except requests.exceptions.ConnectionError as e:
                logger.exception(e)
                time.sleep(5)
                continue

            if response.status_code != 200:
                logger.warning('REAL_TIME_STATS_URL, status code: {}, match_id: {}, game_time: {}'.format(
                    response.status_code, cur_game['match_id'], game['game_time']))
                continue

            res_json = response.json()

            if not res_json:
                logger.warning(
                    'REAL_TIME_STATS_URL, content: {}'.format(response.content))

            real_time_stats_col.insert_one(res_json)

            # games_observed.append(cur_game)

            logger.info('added {}, TOP game_time {}, REAL game_time {}'.format(
                res_json['match']['match_id'],
                game['game_time'],
                res_json['match']['game_time']))

    delay = randint(30, 35)
    logger.info(
        '-------------- sleep {} sec, requests: {}'.format(delay, request_counter))
    time.sleep(delay)

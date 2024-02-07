import requests
import logging
import pymongo
import time
import sys
from dotenv import load_dotenv
import os

load_dotenv()

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
steam_dota_api_db = mongo_client["steam_dota_api"]
league_games_col = steam_dota_api_db["league_games_col"]

LEAGUE_GAMES_URL = 'https://api.steampowered.com/IDOTA2Match_570/getLiveLeagueGames/v1/'

KEY = os.getenv("STEAM_KEY")

logging.basicConfig(format="%(asctime)s > %(levelname)s > %(message)s",
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

request_counter = 0

while True:
    try:
        request_counter += 1
        response = requests.get(LEAGUE_GAMES_URL, params={"key": KEY})
    except requests.exceptions.ConnectionError as e:
        logger.exception(e)
        time.sleep(1)
        continue

    if response.status_code != 200:
        logger.warning('LEAGUE_GAMES_URL, status code: {}, content: {}'.format(
                response.status_code, response.content))
        continue
    
    res_json = response.json()

    if not res_json:
        logger.warning(
            'LEAGUE_GAMES_URL, content: {}'.format(response.content))
        continue

    for game in res_json['result']['games']:
        if 'scoreboard' not in game or game['scoreboard']['duration'] == 0:
            continue

        team_radiant, team_dire = '-', '-'

        if 'radiant_team' in game:
            team_radiant = game['radiant_team']['team_name']
            game['radiant_team']['team_logo'] = 0
        if 'dire_team' in game:
            team_dire = game['dire_team']['team_name']
            game['dire_team']['team_logo'] = 0

        league_games_col.insert_one(game)

        logger.info('added {}, {} vs {}, Duration  {} min'.format(
                game['match_id'],
                team_radiant,
                team_dire,
                round(game['scoreboard']['duration'] / 60)))
    
    delay = 57
    logger.info(
        '-------------- sleep {} sec, requests: {}'.format(delay, request_counter))
    time.sleep(delay)
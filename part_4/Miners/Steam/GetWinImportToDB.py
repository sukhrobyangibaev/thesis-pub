import requests
import time
import pymongo
import logging
import sys
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_CLIENT = pymongo.MongoClient("mongodb://localhost:27017/")
SDA_DB = MONGO_CLIENT["steam_dota_api"]
MLRS_COL = SDA_DB["multi_real_time_stats"]

GET_MATCH_DETAILS_URL = (
    "https://api.steampowered.com/IDOTA2Match_570/GetMatchDetails/v1/"
)
KEY = os.getenv("STEAM_KEY")

logging.basicConfig(
    format="%(asctime)s > %(levelname)s > %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

unique_match_ids_unprocessed = MLRS_COL.distinct(
    "match.match_id", {"match.winner": {"$exists": False}}
)

logger.info("found {} ids".format(len(unique_match_ids_unprocessed)))

error_ids = []
try:
    for match_id in unique_match_ids_unprocessed:
        try:
            time.sleep(2)
            response = requests.get(
                GET_MATCH_DETAILS_URL,
                params={"key": KEY, "match_id": match_id},
            )
            if response.status_code != 200:
                logger.warning("failed getting math details, id = {}".format(match_id))

            res_json = response.json()

            if "radiant_win" not in res_json["result"]:
                logger.info("match_id: {}, radiant_win not in res_json[result]".format(match_id))
                error_ids.append(match_id)
                continue

            try:
                winner = "radiant" if res_json["result"]["radiant_win"] else "dire"
            except Exception as e:
                logger.exception(e)
                time.sleep(5)
                continue

            try:
                update_result = MLRS_COL.update_many(
                    {"match.match_id": match_id},
                    {"$set": {"match.winner": winner}},
                )
            except Exception as e:
                logger.exception(e)
                time.sleep(5)
                continue

            logger.info("match_id: {}, {}".format(match_id, update_result))

        except Exception as e:
            logger.exception(e)
            time.sleep(5)
            continue

    for err_match_id in error_ids:
        delete_result = MLRS_COL.delete_many({"match.match_id": err_match_id})
        logger.info('deleted matches with id {}, {}'.format(err_match_id, delete_result))

except Exception as e:
    logger.exception(e)
    time.sleep(5)

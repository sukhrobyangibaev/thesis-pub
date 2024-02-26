# Applying Decision tree models to solve real-life problems

## Understanding Decision Tree Models: Theory and Methodology.
- [CART](part_1/CART.md)
- [ID3](part_1/ID3.md)
- [Regression](part_1/Regression.md)
- [С45](part_1/С45.md)
- [Random Forest](part_1/Random%20Forest.md)
- [CHAID](part_1/CHAID.md)
- [Gradient Boosting](part_1/Gradient%20Boosting.md)
- [Adaboost](part_1/Adaboost.md)

## Implementing a Universal Decision Tree Framework and Experimental Validation.
- [Annual Profit](part_2/Annual%20Profit/models.ipynb)
- [Breast Cancer](part_2/Breast%20Cancer/models.ipynb)
- [Contraceptive Method Choice](part_2/Contraceptive%20Method%20Choice/models.ipynb)
- [Credit Approval](part_2/Credit%20Approval/models.ipynb)
- [Energy Efficency](part_2/Energy%20Efficency/models.ipynb)
- [Glass Identification](part_2/Glass%20Identification/models.ipynb)
- [Ionosphere](part_2/Ionosphere/models.ipynb)
- [Iris Plants](part_2/Iris%20Plants/models.ipynb)
- [Nursery](part_2/Nursery/models.ipynb)
- [Spam](part_2/Spam/models.ipynb)
- [Zoo](part_2/Zoo/models.ipynb)

| Data                 | CART   | C4.5     | Extra Trees Classifier | Gradient Boosting | Hist Gradient Boosting | Random Forest | Adaboost |
| -------------------- | ------ | -------- | ---------------------- | ----------------- | ---------------------- | ------------- | -------- |
| Annual Profit        | 81.6   | 81.3     | 85.5                   | 87.3              | 87.4                   | 86.4          | 86.9     |
| Breast Cancer        | 93.4   | 92.7     | 96.4                   | 95.6              | 94.9                   | 94.9          | 95.6     |
| Credit Approval      | 87.8   | 78.6     | 87.8                   | 91.6              | 89.3                   | 90.0          | 90.8     |
| Energy Efficiency    | 98.7   | 98.0     | 99.4                   | 98.7              | 98.7                   | 99.4          | 84.4     |
| Glass Identification | 69.8   | 72.1     | 79.1                   | 72.1              | 72.1                   | 76.7          | 67.4     |
| Ionosphere           | 87.1   | 90.0     | 95.7                   | 95.7              | 94.3                   | 95.7          | 97.1     |
| Iris Plants          | 96.7   | 96.7     | 96.7                   | 100.0             | 96.7                   | 96.7          | 96.7     |
| Nursery              | 99.5   | 99.4     | 97.8                   | 100.0             | 98.5                   | 98.4          | 87.9     |
| Spam                 | 90.4   | 92.2     | 95.7                   | 95.7              | 95.2                   | 95.5          | 94.8     |
| Zoo                  | 95.0   | 95.0     | 100.0                  | 95.0              | 100.0                  | 100.0         | 95.0     |
|                      | **90** | **89.6** | **93.4**               | **85.6**          | **92.7**               | **93.3**      | **89.6** |

## Exploring Open Dota API and Steam API for Data Acquisition.
- [Open Dota API](part_3/Open%20Dota%20API/open_dota_api.md)
- [Steam API](part_3/Steam%20API/steam_api.md)

## Data Retrieval from Steam API: Extracting Gaming Insights.
- [Collecting Current Top Live Games Data with GetTopLiveGame API and Gathering Real-time Match Statistics using GetRealtimeStats API](part_4/Miners/Steam/RealtimeStats.py)
- [Retrieving Live Professional Match Statistics with getLiveLeagueGames API](part_4/Miners/Steam/LiveLeagueGames.py)
- [Determining Match Winners: Utilizing GetMatchDetails API](part_4/Miners/Steam/GetWinImportToDB.py)

## Training Models (Public Matches)
- [Training Decision Tree Models with 1,000 Samples of Mined Data](New_Data/1000/decision_trees.ipynb)
- [Training Decision Tree Models with 15,000 Samples of Mined Data](New_Data/multi/15k/all_game_times/decision_trees.ipynb)
- [Training Decision Tree Models with 20,000 Samples of Mined Data](New_Data/multi/20k/all_game_times/decision_trees.ipynb)
- [Training Decision Tree Models with 30,000 Samples of Mined Data](New_Data/multi/30k/decision_trees.ipynb)
- [Training Decision Tree Models with 40,000 Samples of Mined Data](New_Data/multi/40k/decision_trees.ipynb)

## Training Models (League Matches)
- [Training Decision Tree Models with 20,000 Samples of Mined Data](part_6/20k/decision_trees.ipynb)
- [Training Decision Tree Models with 50,000 Samples of Mined Data](part_6/53k/model_scores.txt)
- [Training Decision Tree Models with 100,000 Samples of Mined Data](part_6/100k/model_scores.txt)

## Training Models (League Matches, 3 stages of the game)
- [Training Decision Tree Models with 100,000 Samples of Mined Data](part_7/model_scores.txt)
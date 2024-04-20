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

## Implementing Data Miners for Real-Time Data Acquisition

### Miners For Public Matches

#### [`RealtimeStats.py`](part_4/Miners/Steam/RealtimeStats.py)

This Python script continuously fetches real-time statistics for top live Dota 2 games from the Dota 2 API and stores them in a MongoDB collection for further analysis.

#### [`GetWinImportToDB.py`](part_4/Miners/Steam/GetWinImportToDB.py)

This Python script retrieves match details for unprocessed matches stored in a MongoDB collection and updates the collection with the winning team information.

### Miners For League Matches

#### [`LiveLeagueGames.py`](part_4/Miners/Steam/LiveLeagueGames.py)

This script continuously fetches live league games data from the Dota 2 API and stores them in a MongoDB collection for real-time analysis.

#### [`GetLeagueWin.py`](part_4/Miners/Steam/GetLeagueWin.py)

This Python script retrieves match details from the Dota 2 API for matches stored in a MongoDB collection. It iterates through unprocessed match IDs, retrieves match details, and updates the database with the winning team information.


## Training Models (Public Matches)

This section presents the performance of various decision tree-based models trained on different sizes of public match datasets.

### Notebooks

- [1,000 samples](part_5/1000/decision_trees.ipynb)
- [15,000 samples](part_5/15k/all_game_times/decision_trees.ipynb)
- [20,000 samples](part_5/20k/all_game_times/decision_trees.ipynb)
- [30,000 samples](part_5/30k/decision_trees.ipynb)
- [40,000 samples](part_5/40k/decision_trees.ipynb)

### Model Performance

| Data           | CART | C4.5 | Extra Trees Classifier | Gradient Boosting | Hist Gradient Boosting | Random Forest | Adaboost |
| -------------- | ---- | ---- | ---------------------- | ----------------- | ---------------------- | ------------- | -------- |
| 1,000 samples  | 65.6 | 61.4 | 66.6                   | 64.5              | 71.8                   | 71.8          | 66.6     |
| 15,000 samples | 89.7 | 89.5 | 93.0                   | 93.6              | 98.7                   | 91.3          | 72.3     |
| 20,000 samples | 77.6 | 76.5 | 85.8                   | 83.1              | 89.1                   | 83.2          | 70.6     |
| 30,000 samples | 75.8 | 75.8 | 85.2                   | 81.2              | 87.9                   | 81.1          | 69.3     |
| 40,000 samples | 75.1 | 74.7 | 85.4                   | 82.1              | 88.0                   | 81.7          | 70.6     |


## Training Models (League Matches)
- [20,000 samples](part_6/20k/decision_trees.ipynb)
- [50,000 samples](part_6/53k/model_scores.txt)
- [100,000 samples](part_6/100k/model_scores.txt)

## Training Models (League Matches, 3 stages of the game)
- [100,000 samples](part_7/model_scores.txt)
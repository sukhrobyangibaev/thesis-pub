# NaN

```python
# Search for NaN
df.isnull().values.any()

# Search for NaN in columnds
pd.isna(df).any()
```

# Taking care of missing data

```python
# if you now exact fill value
df = df.fillna(value=10)

# replace nan with mean value of column
# inplace=True means original object is modified with this change.
# If it is False (default) the function doesn't modify the original object,
# instead it returns a modified copy of it and you have to assign it to the original object to replace it

df['column_name'].fillna(df['column_name'].mean(), inplace=True)
```

# Encoding categorical data

```python
# (for y) [0, 1, 2, 1]
# simple way
df['column_name'] = df['column_name'].astype('category').cat.codes
# with sklearn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# (for X) [1.0, 0.0, 0.0]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

# Separate the feature columns from the target column

```python
# with column names
X = df[['Age', 'Experience', 'Rank', 'Nationality']]
y = df['Go']

# with column numbers
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

# Display tree

```python
from sklearn import tree
features = ['Age', 'Experience', 'Rank', 'Nationality']
tree.plot_tree(dtree, feature_names=features)
```

# Random State

```python
random_state = 1
```

Random state is a model hyperparameter used to control the randomness involved in machine learning models. We can use any integer including 0, but not negative ones, only positive integers. The most popular integers are 0 and 42. When we use an integer for random_state, the function will produce the same results across different executions. The results are only changed if we change the integer value.

### Example

Let’s see how random state works with an example. For this, we use Scikit-learn’s train_test_split() function and LinearRegression() function. The train_test_split() function is used to split the dataset into train and test sets. By default, the function shuffles the data (with shuffle=True) before splitting. The random state hyperparameter in the train_test_split() function controls the shuffling process.

With random_state=None , we get different train and test sets across different executions and the shuffling process is out of control.

With random_state=0 , we get the same train and test sets across different executions. With random_state=42, we get the same train and test sets across different executions, but in this time, the train and test sets are different from the previous case with random_state=0.

The train and test sets directly affect the model’s performance score. Because we get different train and test sets with different integer values for random_state in the train_test_split() function, the value of the random state hyperparameter indirectly affects the model’s performance score.

DecisionTreeClassifier(): The random_state in these algorithms controls the randomness involved during the node splitting process by searching for the best feature. It will define the tree structure.

RandomForestClassifier(): The random_state in these algorithms controls two randomized processes — bootstrapping of the samples when creating tress and getting a random subset of features to search for the best feature during the node splitting process when creating each tree. For more details, read this article written by me.

---

### to dump mongo database

```mongo
mongodump
```

### to copy mongo dump from server

```bash
scp -i digital-ocean -r root@159.89.6.72:/root/dump /c/Users/User/Documents
```

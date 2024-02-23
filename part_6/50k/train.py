import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv('part_6/50k/50406x186_samples.csv')

X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

le = LabelEncoder()
y = le.fit_transform(y)

for i, winner in enumerate(le.classes_):
    print(i, '=', winner)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1)

results = []

# --------------------------------
criterion = 'gini'

gini_classifier = DecisionTreeClassifier(criterion=criterion, random_state=1)
gini_classifier.fit(X_train, y_train)
y_pred = gini_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('CART', acc)
results.append({
    'classifier': 'CART',
    'criterion': criterion,
    'accuracy_score': acc
})

with open('part_6/50k/cart_classifier.pkl', 'wb') as f:
    pickle.dump(gini_classifier, f)

# --------------------------------
criterion = 'entropy'

entropy_classifier = DecisionTreeClassifier(
    criterion=criterion, random_state=1)
entropy_classifier.fit(X_train, y_train)
y_pred = entropy_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('C4.5', acc)
results.append({
    'classifier': 'C4.5',
    'criterion': criterion,
    'accuracy_score': acc
})

with open('part_6/50k/c4.5_classifier.pkl', 'wb') as f:
    pickle.dump(entropy_classifier, f)

# --------------------------------
etc_results = []

et_classifier = ExtraTreesClassifier(
    criterion='entropy', n_estimators=150, random_state=1)
et_classifier.fit(X_train, y_train)
y_pred = et_classifier.predict(X_test)
etc_results.append({
    'criterion': 'entropy',
    'n_estimators': 150,
    'accuracy_score': accuracy_score(y_test, y_pred)})

result = max(etc_results, key=lambda x: x['accuracy_score'])
print('Extra Trees Classifier', result['accuracy_score'])
result['classifier'] = 'Extra Trees Classifier'
results.append(result)

with open('part_6/50k/et_classifier.pkl', 'wb') as f:
    pickle.dump(et_classifier, f)

# --------------------------------
gb_results = []

gb_classifier = GradientBoostingClassifier(
    loss='log_loss', n_estimators=50, learning_rate=1, criterion='friedman_mse', max_depth=4, random_state=1)
gb_classifier.fit(X_train, y_train)
y_pred = gb_classifier.predict(X_test)
gb_results.append({
    'loss': 'log_loss',
    'learning_rate': 1,
    'n_estimators': 50,
    'criterion': 'friedman_mse',
    'max_depth': 4,
    'accuracy_score': accuracy_score(y_test, y_pred)
})

result = max(gb_results, key=lambda x: x['accuracy_score'])
print('Gradient Boosting', result['accuracy_score'])
result['classifier'] = 'Gradient Boosting'
results.append(result)

with open('part_6/50k/gb_classifier.pkl', 'wb') as f:
    pickle.dump(gb_classifier, f)

# --------------------------------
hgb_results = []

hgb_classifier = HistGradientBoostingClassifier(
    learning_rate=0.2, max_iter=100, random_state=1)
hgb_classifier.fit(X_train, y_train)
y_pred = hgb_classifier.predict(X_test)
hgb_results.append({
    'learning_rate': 0.2,
    'max_iter': 100,
    'accuracy_score': accuracy_score(y_test, y_pred)
})

result = max(hgb_results, key=lambda x: x['accuracy_score'])
print('Hist Gradient Boosting', result['accuracy_score'])
result['classifier'] = 'Hist Gradient Boosting'
results.append(result)

with open('part_6/50k/hgb_classifier.pkl', 'wb') as f:
    pickle.dump(hgb_classifier, f)

# --------------------------------
rf_results = []

rf_classifier = RandomForestClassifier(
    n_estimators=50, criterion='gini', random_state=1)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
rf_results.append({
    'n_estimators': 50,
    'criterion': 'gini',
    'accuracy_score': accuracy_score(y_test, y_pred)
})

result = max(rf_results, key=lambda x: x['accuracy_score'])
print('Random Forest', result['accuracy_score'])
result['classifier'] = 'Random Forest'
results.append(result)

with open('part_6/50k/rf_classifier.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

# --------------------------------
ada_results = []

ab_classifier = AdaBoostClassifier(
    n_estimators=50, learning_rate=0.1, algorithm='SAMME', random_state=0)
ab_classifier.fit(X_train, y_train)
y_pred = ab_classifier.predict(X_test)
ada_results.append({
    'n_estimators': 50,
    'learning_rate': 0.1,
    'algorithm': 'SAMME.R',
    'accuracy_score': accuracy_score(y_test, y_pred)
})

result = max(ada_results, key=lambda x: x['accuracy_score'])
print('Adaboost', result['accuracy_score'])
result['classifier'] = 'Adaboost'
results.append(result)

with open('part_6/50k/ab_classifier.pkl', 'wb') as f:
    pickle.dump(ab_classifier, f)
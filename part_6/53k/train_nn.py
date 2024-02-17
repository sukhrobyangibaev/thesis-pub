import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('part_6/53k/53169x158_samples.csv', delimiter=',')

label_encoder = LabelEncoder()
dataset['winner'] = label_encoder.fit_transform(dataset['winner'])

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=100, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs=100)

y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5

acc = accuracy_score(y_test, y_pred)
print(acc)

import pickle

with open('part_6/53k/ann.pkl','wb') as f:
    pickle.dump(ann, f)
with open('part_6/53k/ann_standart_scaler.pkl','wb') as f:
    pickle.dump(sc, f)
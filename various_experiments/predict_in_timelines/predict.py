import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler


with open('part_9_embedding/trained_models/10min/et_classifier.pkl', 'rb') as f:
    M10_ET_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/10min/rf_classifier.pkl', 'rb') as f:
    M10_RF_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/10min/hgb_classifier.pkl', 'rb') as f:
    M10_HGB_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/10min/gb_classifier.pkl', 'rb') as f:
    M10_GB_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/10min/cart_classifier.pkl', 'rb') as f:
    M10_CART_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/10min/cart_classifier.pkl', 'rb') as f:
    M10_C45_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/10min/cart_classifier.pkl', 'rb') as f:
    M10_AB_CLASSIFIER = pickle.load(f)  

with open('part_9_embedding/trained_models/20min/et_classifier.pkl', 'rb') as f:
    M20_ET_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/20min/rf_classifier.pkl', 'rb') as f:
    M20_RF_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/20min/hgb_classifier.pkl', 'rb') as f:
    M20_HGB_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/20min/gb_classifier.pkl', 'rb') as f:
    M20_GB_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/20min/cart_classifier.pkl', 'rb') as f:
    M20_CART_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/20min/cart_classifier.pkl', 'rb') as f:
    M20_C45_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/20min/cart_classifier.pkl', 'rb') as f:
    M20_AB_CLASSIFIER = pickle.load(f) 

with open('part_9_embedding/trained_models/30min/et_classifier.pkl', 'rb') as f:
    M30_ET_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/30min/rf_classifier.pkl', 'rb') as f:
    M30_RF_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/30min/hgb_classifier.pkl', 'rb') as f:
    M30_HGB_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/30min/gb_classifier.pkl', 'rb') as f:
    M30_GB_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/30min/cart_classifier.pkl', 'rb') as f:
    M30_CART_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/30min/cart_classifier.pkl', 'rb') as f:
    M30_C45_CLASSIFIER = pickle.load(f)
with open('part_9_embedding/trained_models/30min/cart_classifier.pkl', 'rb') as f:
    M30_AB_CLASSIFIER = pickle.load(f) 


df = pd.read_csv('various_experiments/predict_in_timelines/27x50_samples.csv')

X = df.iloc[:, 0:-1].values

durations = []

net_worth = []
xpm = []

et_predictions = []
rf_predictions = []
hgb_predictions = []
gb_predictions = []


for row in X:
    durations.append(row[0])
    net_worth.append(row[-7])
    xpm.append(row[-1])

    min, sec = divmod(int(row[0]), 60)

    if float(row[0]) <= 600:
        et_pred = M10_ET_CLASSIFIER.predict_proba([row])[0]
        if et_pred[0] > et_pred[1]:
            et_pred_str = f"🔴 {round(et_pred[0] * 100)}%"
        else:
            et_pred_str = f"🟢 {round(et_pred[1] * 100)}%"

        rf_pred = M10_RF_CLASSIFIER.predict_proba([row])[0]
        if rf_pred[0] > rf_pred[1]:
            rf_pred_str = f"🔴 {round(rf_pred[0] * 100)}%"
        else:
            rf_pred_str = f"🟢 {round(rf_pred[1] * 100)}%"

        hgb_pred = M10_HGB_CLASSIFIER.predict_proba([row])[0]
        if hgb_pred[0] > hgb_pred[1]:
            hgb_pred_str = f"🔴 {round(hgb_pred[0] * 100)}%"
        else:
            hgb_pred_str = f"🟢 {round(hgb_pred[1] * 100)}%"

        gb_pred = M10_GB_CLASSIFIER.predict_proba([row])[0]
        if gb_pred[0] > gb_pred[1]:
            gb_pred_str = f"🔴 {round(gb_pred[0] * 100)}%"
        else:
            gb_pred_str = f"🟢 {round(gb_pred[1] * 100)}%"

        avg_dire = (et_pred[0] + rf_pred[0] + hgb_pred[0] + gb_pred[0]) * 25
        avg_radiant = (et_pred[1] + rf_pred[1] + hgb_pred[1] + gb_pred[1]) * 25
        if avg_dire > avg_radiant:
            avg_str = f"🔴 {round(avg_dire)}%"
        else:
            avg_str = f"🟢 {round(avg_radiant)}%"

        cart_pred = M10_CART_CLASSIFIER.predict_proba([row])[0]
        if cart_pred[0] > cart_pred[1]:
            cart_pred_str = f"🔴 {round(cart_pred[0] * 100)}%"
        else:
            cart_pred_str = f"🟢 {round(cart_pred[1] * 100)}%"

        c45_pred = M10_C45_CLASSIFIER.predict_proba([row])[0]
        if c45_pred[0] > c45_pred[1]:
            c45_pred_str = f"🔴 {round(c45_pred[0] * 100)}%"
        else:
            c45_pred_str = f"🟢 {round(c45_pred[1] * 100)}%"

        ab_pred = M10_AB_CLASSIFIER.predict_proba([row])[0]
        if ab_pred[0] > ab_pred[1]:
            ab_pred_str = f"🔴 {round(ab_pred[0] * 100)}%"
        else:
            ab_pred_str = f"🟢 {round(ab_pred[1] * 100)}%"

    elif float(row[0]) > 600 and float(row[0]) <= 1200:
        et_pred = M20_ET_CLASSIFIER.predict_proba([row])[0]
        if et_pred[0] > et_pred[1]:
            et_pred_str = f"🔴 {round(et_pred[0] * 100)}%"
        else:
            et_pred_str = f"🟢 {round(et_pred[1] * 100)}%"

        rf_pred = M20_RF_CLASSIFIER.predict_proba([row])[0]
        if rf_pred[0] > rf_pred[1]:
            rf_pred_str = f"🔴 {round(rf_pred[0] * 100)}%"
        else:
            rf_pred_str = f"🟢 {round(rf_pred[1] * 100)}%"

        hgb_pred = M20_HGB_CLASSIFIER.predict_proba([row])[0]
        if hgb_pred[0] > hgb_pred[1]:
            hgb_pred_str = f"🔴 {round(hgb_pred[0] * 100)}%"
        else:
            hgb_pred_str = f"🟢 {round(hgb_pred[1] * 100)}%"

        gb_pred = M20_GB_CLASSIFIER.predict_proba([row])[0]
        if gb_pred[0] > gb_pred[1]:
            gb_pred_str = f"🔴 {round(gb_pred[0] * 100)}%"
        else:
            gb_pred_str = f"🟢 {round(gb_pred[1] * 100)}%"

        avg_dire = (et_pred[0] + rf_pred[0] + hgb_pred[0] + gb_pred[0]) * 25
        avg_radiant = (et_pred[1] + rf_pred[1] + hgb_pred[1] + gb_pred[1]) * 25
        if avg_dire > avg_radiant:
            avg_str = f"🔴 {round(avg_dire)}%"
        else:
            avg_str = f"🟢 {round(avg_radiant)}%"

        cart_pred = M20_CART_CLASSIFIER.predict_proba([row])[0]
        if cart_pred[0] > cart_pred[1]:
            cart_pred_str = f"🔴 {round(cart_pred[0] * 100)}%"
        else:
            cart_pred_str = f"🟢 {round(cart_pred[1] * 100)}%"

        c45_pred = M20_C45_CLASSIFIER.predict_proba([row])[0]
        if c45_pred[0] > c45_pred[1]:
            c45_pred_str = f"🔴 {round(c45_pred[0] * 100)}%"
        else:
            c45_pred_str = f"🟢 {round(c45_pred[1] * 100)}%"

        ab_pred = M20_AB_CLASSIFIER.predict_proba([row])[0]
        if ab_pred[0] > ab_pred[1]:
            ab_pred_str = f"🔴 {round(ab_pred[0] * 100)}%"
        else:
            ab_pred_str = f"🟢 {round(ab_pred[1] * 100)}%"

    else:
        et_pred = M30_ET_CLASSIFIER.predict_proba([row])[0]
        if et_pred[0] > et_pred[1]:
            et_pred_str = f"🔴 {round(et_pred[0] * 100)}%"
        else:
            et_pred_str = f"🟢 {round(et_pred[1] * 100)}%"

        rf_pred = M30_RF_CLASSIFIER.predict_proba([row])[0]
        if rf_pred[0] > rf_pred[1]:
            rf_pred_str = f"🔴 {round(rf_pred[0] * 100)}%"
        else:
            rf_pred_str = f"🟢 {round(rf_pred[1] * 100)}%"

        hgb_pred = M30_HGB_CLASSIFIER.predict_proba([row])[0]
        if hgb_pred[0] > hgb_pred[1]:
            hgb_pred_str = f"🔴 {round(hgb_pred[0] * 100)}%"
        else:
            hgb_pred_str = f"🟢 {round(hgb_pred[1] * 100)}%"

        gb_pred = M30_GB_CLASSIFIER.predict_proba([row])[0]
        if gb_pred[0] > gb_pred[1]:
            gb_pred_str = f"🔴 {round(gb_pred[0] * 100)}%"
        else:
            gb_pred_str = f"🟢 {round(gb_pred[1] * 100)}%"

        avg_dire = (et_pred[0] + rf_pred[0] + hgb_pred[0] + gb_pred[0]) * 25
        avg_radiant = (et_pred[1] + rf_pred[1] + hgb_pred[1] + gb_pred[1]) * 25
        if avg_dire > avg_radiant:
            avg_str = f"🔴 {round(avg_dire)}%"
        else:
            avg_str = f"🟢 {round(avg_radiant)}%"

        cart_pred = M30_CART_CLASSIFIER.predict_proba([row])[0]
        if cart_pred[0] > cart_pred[1]:
            cart_pred_str = f"🔴 {round(cart_pred[0] * 100)}%"
        else:
            cart_pred_str = f"🟢 {round(cart_pred[1] * 100)}%"

        c45_pred = M30_C45_CLASSIFIER.predict_proba([row])[0]
        if c45_pred[0] > c45_pred[1]:
            c45_pred_str = f"🔴 {round(c45_pred[0] * 100)}%"
        else:
            c45_pred_str = f"🟢 {round(c45_pred[1] * 100)}%"

        ab_pred = M30_AB_CLASSIFIER.predict_proba([row])[0]
        if ab_pred[0] > ab_pred[1]:
            ab_pred_str = f"🔴 {round(ab_pred[0] * 100)}%"
        else:
            ab_pred_str = f"🟢 {round(ab_pred[1] * 100)}%"


    et_predictions.append(et_pred[1] * 100)
    rf_predictions.append(rf_pred[1] * 100)
    hgb_predictions.append(hgb_pred[1] * 100)
    gb_predictions.append(gb_pred[1] * 100)


    # prediction = '{}:{} {}'.format(min, sec, et_pred_str)
    # print(prediction)

durations = np.array(durations) / 60

average_predictions = np.mean([et_predictions, rf_predictions, hgb_predictions, gb_predictions], axis=0)
spl = UnivariateSpline(durations, average_predictions)
smooth_durations = np.linspace(np.min(durations), np.max(durations), 500)
smooth_predictions = spl(smooth_durations)

scaler = MinMaxScaler(feature_range=(0, 100))

net_worth = np.array(net_worth).reshape(-1, 1)
net_worth_scaled = scaler.fit_transform(net_worth).flatten()

xpm = np.array(xpm).reshape(-1, 1)
xpm_scaled = scaler.fit_transform(xpm).flatten()

# plt.plot(durations, et_predictions, label='ET')
# plt.plot(durations, rf_predictions, label='RF')
# plt.plot(durations, hgb_predictions, label='HGB')
# plt.plot(durations, gb_predictions, label='GB')
plt.plot(smooth_durations, smooth_predictions, label='Average Prediction', color='green', linestyle='-')
plt.plot(durations, net_worth_scaled, label='Net Worth Difference (scaled)', linestyle='-', color='blue')
plt.plot(durations, xpm_scaled, label='XPM Difference (scaled)', linestyle='-', color='red')
plt.axhline(50, color='black', linestyle='--')  # Add horizontal line at y=50

plt.legend()
plt.xlabel('Duration (minutes)')
plt.ylabel('Prediction')
plt.title('Prediction vs Duration')
plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))


plt.show()

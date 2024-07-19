import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler


with open('trained_models/10min/et_classifier.pkl', 'rb') as f:
    M10_ET_CLASSIFIER = pickle.load(f)
with open('trained_models/10min/rf_classifier.pkl', 'rb') as f:
    M10_RF_CLASSIFIER = pickle.load(f)
with open('trained_models/10min/hgb_classifier.pkl', 'rb') as f:
    M10_HGB_CLASSIFIER = pickle.load(f)
with open('trained_models/10min/gb_classifier.pkl', 'rb') as f:
    M10_GB_CLASSIFIER = pickle.load(f)
with open('trained_models/10min/cart_classifier.pkl', 'rb') as f:
    M10_CART_CLASSIFIER = pickle.load(f)
with open('trained_models/10min/cart_classifier.pkl', 'rb') as f:
    M10_C45_CLASSIFIER = pickle.load(f)
with open('trained_models/10min/cart_classifier.pkl', 'rb') as f:
    M10_AB_CLASSIFIER = pickle.load(f)  

with open('trained_models/20min/et_classifier.pkl', 'rb') as f:
    M20_ET_CLASSIFIER = pickle.load(f)
with open('trained_models/20min/rf_classifier.pkl', 'rb') as f:
    M20_RF_CLASSIFIER = pickle.load(f)
with open('trained_models/20min/hgb_classifier.pkl', 'rb') as f:
    M20_HGB_CLASSIFIER = pickle.load(f)
with open('trained_models/20min/gb_classifier.pkl', 'rb') as f:
    M20_GB_CLASSIFIER = pickle.load(f)
with open('trained_models/20min/cart_classifier.pkl', 'rb') as f:
    M20_CART_CLASSIFIER = pickle.load(f)
with open('trained_models/20min/cart_classifier.pkl', 'rb') as f:
    M20_C45_CLASSIFIER = pickle.load(f)
with open('trained_models/20min/cart_classifier.pkl', 'rb') as f:
    M20_AB_CLASSIFIER = pickle.load(f) 

with open('trained_models/30min/et_classifier.pkl', 'rb') as f:
    M30_ET_CLASSIFIER = pickle.load(f)
with open('trained_models/30min/rf_classifier.pkl', 'rb') as f:
    M30_RF_CLASSIFIER = pickle.load(f)
with open('trained_models/30min/hgb_classifier.pkl', 'rb') as f:
    M30_HGB_CLASSIFIER = pickle.load(f)
with open('trained_models/30min/gb_classifier.pkl', 'rb') as f:
    M30_GB_CLASSIFIER = pickle.load(f)
with open('trained_models/30min/cart_classifier.pkl', 'rb') as f:
    M30_CART_CLASSIFIER = pickle.load(f)
with open('trained_models/30min/cart_classifier.pkl', 'rb') as f:
    M30_C45_CLASSIFIER = pickle.load(f)
with open('trained_models/30min/cart_classifier.pkl', 'rb') as f:
    M30_AB_CLASSIFIER = pickle.load(f) 


df = pd.read_csv('various_experiments/predict_in_timelines/results/7702650896/41x46_samples.csv')

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

average_predictions = np.mean([et_predictions, rf_predictions, hgb_predictions], axis=0)
# average_predictions = np.mean([et_predictions, rf_predictions, hgb_predictions, gb_predictions], axis=0)
spl = UnivariateSpline(durations, average_predictions)
smooth_durations = np.linspace(np.min(durations), np.max(durations), 500)
smooth_predictions = spl(smooth_durations)

scaler = MinMaxScaler(feature_range=(0, 100))

net_worth = np.array(net_worth).reshape(-1, 1)
net_worth_scaled = scaler.fit_transform(net_worth).flatten()

xpm = np.array(xpm).reshape(-1, 1)
xpm_scaled = scaler.fit_transform(xpm).flatten()

# Plotting
# --------------------------- Extra Tree Classifier
plt.figure()
plt.plot(durations, et_predictions, label='Extra Tree Classifier', linestyle='-', color='red')

plt.axhline(50, color='black', linestyle='--')  # Add horizontal line at y=50
plt.legend(loc='upper right')
plt.xlabel('Duration (minutes)')
plt.ylabel('Prediction (%)')
plt.title('Extra Tree Classifier')
plt.ylim(0, 100)
plt.yticks([0, 25, 50, 75, 100], [100, 75, 50, 75, 100])
plt.text(0, 2, 'Dire Win', ha='left', va='bottom')
plt.text(0, 98, 'Radiant Win', ha='left', va='top')

# --------------------------- Random Forest Classifier
plt.figure()
plt.plot(durations, rf_predictions, label='Random Forest Classifier', linestyle='-', color='blue')

plt.axhline(50, color='black', linestyle='--')  # Add horizontal line at y=50
plt.legend(loc='upper right')
plt.xlabel('Duration (minutes)')
plt.ylabel('Prediction (%)')
plt.title('Random Forest Classifier')
plt.ylim(0, 100)
plt.yticks([0, 25, 50, 75, 100], [100, 75, 50, 75, 100])
plt.text(0, 2, 'Dire Win', ha='left', va='bottom')
plt.text(0, 98, 'Radiant Win', ha='left', va='top')

# --------------------------- Hist Gradient Boosting Classifier
plt.figure()
plt.plot(durations, hgb_predictions, label='Hist Gradient Boosting Classifier', linestyle='-', color='green')

plt.axhline(50, color='black', linestyle='--')  # Add horizontal line at y=50
plt.legend(loc='upper right')
plt.xlabel('Duration (minutes)')
plt.ylabel('Prediction (%)')
plt.title(' Hist Gradient Boosting Classifier')
plt.ylim(0, 100)
plt.yticks([0, 25, 50, 75, 100], [100, 75, 50, 75, 100])
plt.text(0, 2, 'Dire Win', ha='left', va='bottom')
plt.text(0, 98, 'Radiant Win', ha='left', va='top')

# --------------------------- Gradient Boosting Classifier
# plt.figure()
# plt.plot(durations, gb_predictions, label='GB') # not good enough

# plt.axhline(50, color='black', linestyle='--')  # Add horizontal line at y=50
# plt.legend(loc='upper right')
# plt.xlabel('Duration (minutes)')
# plt.ylabel('Prediction (%)')
# plt.title('Gradient Boosting Classifier')
# plt.ylim(0, 100)
# plt.yticks([0, 25, 50, 75, 100], [100, 75, 50, 75, 100])
# plt.text(0, 2, 'Dire Win', ha='left', va='bottom')
# plt.text(0, 98, 'Radiant Win', ha='left', va='top')

# --------------------------- Average of all classifiers
plt.figure()
plt.plot(smooth_durations, smooth_predictions, label='Average (ET, RF, HGB)', linestyle='-', color='black')

plt.axhline(50, color='black', linestyle='--')  # Add horizontal line at y=50
plt.legend(loc='upper right')
plt.xlabel('Duration (minutes)')
plt.ylabel('Prediction (%)')
plt.title('Average (ET, RF, HGB)')
plt.ylim(0, 100)
plt.yticks([0, 25, 50, 75, 100], [100, 75, 50, 75, 100])
plt.text(0, 2, 'Dire Win', ha='left', va='bottom')
plt.text(0, 98, 'Radiant Win', ha='left', va='top')

# --------------------------- ET, RF, HGB and Average
plt.figure()
plt.plot(durations, et_predictions, label='Extra Tree Classifier', linestyle='-', color='red')
plt.plot(durations, rf_predictions, label='Random Forest Classifier', linestyle='-', color='blue')
plt.plot(durations, hgb_predictions, label='Hist Gradient Boosting Classifier', linestyle='-', color='green')
plt.plot(smooth_durations, smooth_predictions, label='Average (ET, RF, HGB)', linestyle='-', color='black')

plt.axhline(50, color='black', linestyle='--')  # Add horizontal line at y=50
plt.legend(loc='upper right')
plt.xlabel('Duration (minutes)')
plt.ylabel('Prediction (%)')
plt.title('ET, RF, HGB and Average')
plt.ylim(0, 100)
plt.yticks([0, 25, 50, 75, 100], [100, 75, 50, 75, 100])
plt.text(0, 2, 'Dire Win', ha='left', va='bottom')
plt.text(0, 98, 'Radiant Win', ha='left', va='top')

# --------------------------- Net Worth and XPM
# plt.figure()
# plt.plot(durations, net_worth_scaled, label='Net Worth Difference (scaled)', linestyle='-', color='blue')

# plt.axhline(50, color='black', linestyle='--')  # Add horizontal line at y=50
# plt.legend(loc='upper right')
# plt.xlabel('Duration (minutes)')
# plt.ylabel('Prediction (%)')
# plt.title('Net Worth Difference (scaled)')
# plt.ylim(0, 100)
# plt.yticks([0, 25, 50, 75, 100], [100, 75, 50, 75, 100])
# plt.text(0, 2, 'Dire Win', ha='left', va='bottom')
# plt.text(0, 98, 'Radiant Win', ha='left', va='top')

# --------------------------- Net Worth and XPM
# plt.figure()
# plt.plot(durations, xpm_scaled, label='XPM Difference (scaled)', linestyle='-', color='red')

# plt.axhline(50, color='black', linestyle='--')  # Add horizontal line at y=50
# plt.legend(loc='upper right')
# plt.xlabel('Duration (minutes)')
# plt.ylabel('Prediction (%)')
# plt.title('XPM Difference (scaled)')
# plt.ylim(0, 100)
# plt.yticks([0, 25, 50, 75, 100], [100, 75, 50, 75, 100])
# plt.text(0, 2, 'Dire Win', ha='left', va='bottom')
# plt.text(0, 98, 'Radiant Win', ha='left', va='top')

plt.show()

# CART algorithm

- Used both for classification and regression tree
- It uses Gini index as metric/cost function to evaluate split in feature selection in case of classification tree
- it is used for binary classification
- it uses least square as a metric to select features in case of Regression tree

# Example

| Day | Outlook  | Temp. | Humidity | Wind   | Decision |
| --- | -------- | ----- | -------- | ------ | -------- |
| 1   | Sunny    | Hot   | High     | Weak   | No       |
| 2   | Sunny    | Hot   | High     | Strong | No       |
| 3   | Overcast | Hot   | High     | Weak   | Yes      |
| 4   | Rain     | Mild  | High     | Weak   | Yes      |
| 5   | Rain     | Cool  | Normal   | Weak   | Yes      |
| 6   | Rain     | Cool  | Normal   | Strong | No       |
| 7   | Overcast | Cool  | Normal   | Strong | Yes      |
| 8   | Sunny    | Mild  | High     | Weak   | No       |
| 9   | Sunny    | Cool  | Normal   | Weak   | Yes      |
| 10  | Rain     | Mild  | Normal   | Weak   | Yes      |
| 11  | Sunny    | Mild  | Normal   | Strong | Yes      |
| 12  | Overcast | Mild  | High     | Strong | Yes      |
| 13  | Overcast | Hot   | Normal   | Weak   | Yes      |
| 14  | Rain     | Mild  | High     | Strong | No       |

## Outlook

> Outlook is a nominal feature. It can be sunny, overcast or rain. I will summarize the final decisions for outlook feature.

| Outlook  | Yes | No  | Number of instances |
| -------- | --- | --- | ------------------- |
| Sunny    | 2   | 3   | 5                   |
| Overcast | 4   | 0   | 4                   |
| Rain     | 3   | 2   | 5                   |

<br/>

$$Gini_i = 1 – \sum_{i=1}^{n} (P_i)^2$$

<br/>

$Gini(Outlook=Sunny) = 1 – (2/5)^2 – (3/5)^2 = 1 – 0.16 – 0.36 = 0.48$

$Gini(Outlook=Overcast) = 1 – (4/4)^2 – (0/4)^2 = 0$

$Gini(Outlook=Rain) = 1 – (3/5)^2 – (2/5)^2 = 1 – 0.36 – 0.16 = 0.48$

> Then, we will calculate weighted sum of gini indexes for outlook feature.

<br/>

$$Gini = \sum_{i=1}^{n} P_i * G_i$$

<br/>

$Gini(Outlook) = (5/14) * 0.48 + (4/14) * 0 + (5/14) * 0.48 = 0.171 + 0 + 0.171 = 0.342$

> We’ve calculated gini index values for each feature.

| Feature     | Gini index |
| ----------- | ---------- |
| Outlook     | 0.342      |
| Temperature | 0.439      |
| Humidity    | 0.367      |
| Wind        | 0.428      |

> The winner will be **outlook** feature because its cost is the lowest.

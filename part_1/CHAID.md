# CHAID algorithm

CHAID - chi-square automatic interaction detection.

Chi-square is a metric to find the significance of a feature. The higher the value, the higher the statistical significance.

- CHAID builds decision trees for classification problems.
- It expects data sets having a categorical target variable.

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

---

> We need to find most dominant feature in this data set.

## **Outlook**

| Outlook  | Yes | No  | Total |
| -------- | --- | --- | ----- |
| Sunny    | 2   | 3   | 5     |
| Overcast | 4   | 0   | 4     |
| Rain     | 3   | 2   | 5     |

> Expected values are the half of total column because there are 2 classes in the decision. It is easy to calculate the chi-squared values based on this table.

<br/>

$$D = 2 (Yes, No)$$

$$ý = \frac{Total(Yes, No)}{D}$$

<br/>

$Expected(sunny) = \frac{5}{2} = 2.5$

$Expected(overcast) = \frac{4}{2} = 2$

$Expected(rain) = \frac{5}{2} = 2.5$

<br/>

$$ChiSquare = \sqrt{\frac{(y - ý)^2}{ý}}$$

<br/>

$ChiSquare(Yes, outlook=sunny) = \sqrt{\frac{(2 - 2.5)^2}{2.5}} = 0.316 $

$ChiSquare(Yes, outlook=overcast) = \sqrt{\frac{(4 - 2)^2}{2}} = 1.414 $

$ChiSquare(Yes, outlook=rain) = \sqrt{\frac{(3 - 2.5)^2}{2.5}} = 0.316 $

<br/>

$ChiSquare(No, outlook=sunny) = \sqrt{\frac{(3 - 2.5)^2}{2.5}} = 0.316 $

$ChiSquare(No, outlook=overcast) = \sqrt{\frac{(0 - 2)^2}{2}} = 1.414 $

$ChiSquare(No, outlook=rain) = \sqrt{\frac{(2 - 2.5)^2}{2.5}} = 0.316 $

<br/>

> Chi-square value of outlook is the sum of chi-square yes and no columns.
> <br/>

$$ChiSquare = \sum{ChiSquare(Yes, No)}$$

<br/>

$ChiSquare(outlook) = 0.316 + 0.316 + 1.414 + 1.414 + 0.316 + 0.316 = 4.092$

> Now, we will find chi-square values for other features. The feature having the maximum chi-square value will be the decision point.

$ChiSquare(temperature ) = 0 + 0 + 0.577 + 0.577 + 0.707 + 0.707 = 2.569$

$ChiSquare(humidity) = 0.267 + 0.267 + 1.336 + 1.336 = 3.207$

$ChiSquare(wind) = 0.802 + 0.802 + 0 + 0 = 1.604$

> We’ve found the chi square values of all features. Let’s see them all in a table.

| Feature     | Chi-square value |
| ----------- | ---------------- |
| Outlook     | 4.092            |
| Temperature | 2.569            |
| Humidity    | 3.207            |
| Wind        | 1.604            |

> Outlook feature has the highest chi-square value. This means that it is the most significant feature. So, we will put this feature to the root node.

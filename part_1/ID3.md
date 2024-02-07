# ID3 algorithm

- Used only for classification
- It uses Entropy as metric/cost function to evaluate split in feature selection in case of classification tree

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

> We need to calculate the global entropy first. Decision column consists of 14 instances and includes two labels: yes and no. There are 9 decisions labeled yes, and 5 decisions labeled no.

<br/>

$$Entropy(Decision) = – P(Yes) * \log_{2}^{P(Yes)} – P(No) * \log_{2}^{P(No)}$$

<br/>

$Entropy(Decision) = – (9/14) * \log_{2}^{9/14} – (5/14) * \log_{2}^{5/14} = 0.940$

---

> Now, we need to find the most dominant factor for decisioning.

### **Wind** factor on decision

> There are 8 instances for weak wind. Decision of 2 items are no and 6 items are yes as illustrated below.

<br/>

$$Entropy(S) = \sum_{i=1}^{n} – P_i * \log_{2}^{P_i}$$

<br/>
$Entropy(Decision|Wind=Weak) = – P(Yes) * \log_{2}^{P(Yes)} – P(No) * \log_{2}^{P(No)}$

$Entropy(Decision|Wind=Weak) = – (2/8) * \log_{2}^{2/8} – (6/8) * \log_{2}^{6/8} = 0.811$

> Here, there are 6 instances for strong wind. Decision is divided into two equal parts.

$Entropy(Decision|Wind=Strong) = – (3/6) * \log_{2}^{3/6} – (3/6) * \log_{2}^{3/6} = 1$

> Now, we can turn back to Gain(Decision, Wind) equation.

<br/>

$$Gain(S, A) = Entropy(S)  – \sum_{i=1}^{n}  P(S|A) * Entropy(S|A) $$

<br/>

$Gain(Decision, Wind) = Entropy(Decision) – [ P(Decision|Wind=Weak) * Entropy(Decision|Wind=Weak) ] – [ P(Decision|Wind=Strong) * Entropy(Decision|Wind=Strong) ]$

$Gain(Decision, Wind) = 0.940 – [ (8/14) * 0.811 ] – [ (6/14) * 1] = 0.048$

---

### **Other** factors on decision

> We have applied similar calculation on the other columns.

$Gain(Decision, Outlook) = 0.246$

$Gain(Decision, Temperature) = 0.029$

$Gain(Decision, Humidity) = 0.151$

$Gain(Decision, Wind) = 0.048$

> As seen, outlook factor on decision produces the highest score. That’s why, outlook decision will appear in the root node of the tree.

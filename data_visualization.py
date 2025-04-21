import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the dataset here (prolly not the most efficient)
df = pd.read_csv('data/winequality-red.csv', delimiter=';') # relative for now
# heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm'), plt.title('Heatmap')
plt.show()

# show wine quality here (?)
sns.countplot(x='quality', data=df)
plt.title('wine quality distribution')
plt.xlabel('quality')
plt.ylabel('count')
plt.show()
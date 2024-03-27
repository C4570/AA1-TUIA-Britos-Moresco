import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path= 'weatherAUS.csv'
df = pd.read_csv(file_path, sep=',',engine='python')

# Graficamos los boxplots de las columnas numéricas para ver si hay outliers
plt.figure(figsize=(12,8))
sns.sns.scatterplot(x = 'Unname: 0', y = 'MaxTemp', hue = "grupo",
                data = df)
plt.tight_layout()
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

 

df = pd.read_csv('Zjazd5\\dane\\weight-height (1).csv')
print(df.head(10))
print(df.Gender.value_counts())
df.Height *= 2.54
df.Weight /=2.2
print(f'Po zmianie jednostek: \n{df.head(10)}')

plt.hist(df.query("Gender=='Male'").Weight, bins=50)
plt.hist(df.query("Gender=='Female'").Weight, bins=50)
#plt.show()

df = pd.get_dummies(df)
print(df.head())
del df['Gender_Male']
print(df.head())
df.rename(columns={'Gender_Female': 'Gender'})
print(df.head())

model = LinearRegression()
model.fit( df[['Height', 'Gender']]          , df['Weight']          )

print(f'Współczynnik kierunkowy: {model.coef_}\nWyraz wolny: {model.intercept}')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Tworzenie modelu
model = Sequential()

# Dodanie warstw
model.add(Dense(1, activation='relu'))
model.add(Dense(1, activation='linear'))
model.add(Dense(1, activation='linear'))
model.add(Dense(1, activation='linear'))

# Kompilacja modelu
model.compile(optimizer='rmsprop', loss='mse')

# Wczytanie danych
df = pd.read_csv('Zjazd5\\dane\\f-c.csv', usecols=[1, 2])
print(df)

# Wizualizacja danych
plt.scatter(df.F, df.C)
plt.show()

# Przygotowanie danych do modelu
X = df.F.values.reshape(-1, 1)  # Przekształcenie do odpowiedniego kształtu
y = df.C.values

# Trenowanie modelu
result = model.fit(X, y, epochs=10, verbose=2)

# Przeanalizowanie wyników
df1 = pd.DataFrame(result.history)
print(df1)
df1.plot()
plt.show()

# Predykcje modelu
C_pred = model.predict(X)

# Wizualizacja wyników
plt.scatter(df.F, df.C)
plt.scatter(df.F, C_pred, c='r')
plt.show()
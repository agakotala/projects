import numpy as np
import matplotlib.pyplot as plt

# Definicje funkcji
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Zakres wartości x
x = np.linspace(-5, 5, 400)

# Obliczanie wartości funkcji
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

# Tworzenie wykresów
plt.figure(figsize=(12, 4))

# Wykres Sigmoid
plt.subplot(1, 3, 1)
plt.plot(x, y_sigmoid, label="Sigmoid", color='blue')
plt.title("Sigmoid Function")
plt.grid()
plt.legend()

# Wykres ReLU
plt.subplot(1, 3, 2)
plt.plot(x, y_relu, label="ReLU", color='red')
plt.title("ReLU Function")
plt.grid()
plt.legend()

# Wykres Tangens Hiperboliczny
plt.subplot(1, 3, 3)
plt.plot(x, y_tanh, label="Tanh", color='green')
plt.title("Hyperbolic Tangent (Tanh) Function")
plt.grid()
plt.legend()

# Wyświetlenie wykresów
plt.tight_layout()
#plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Definicje funkcji
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Zakres wartości x
x = np.linspace(-5, 5, 400)

# Obliczanie wartości funkcji
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

# Tworzenie wykresów
plt.figure(figsize=(12, 4))

# Wykres Sigmoid
plt.subplot(1, 3, 1)
plt.plot(x, y_sigmoid, label="Sigmoid", color='blue')
plt.title("Sigmoid Function")
plt.grid()
plt.legend()

# Wykres ReLU
plt.subplot(1, 3, 2)
plt.plot(x, y_relu, label="ReLU", color='red')
plt.title("ReLU Function")
plt.grid()
plt.legend()

# Wykres Tangens Hiperboliczny
plt.subplot(1, 3, 3)
plt.plot(x, y_tanh, label="Tanh", color='green')
plt.title("Hyperbolic Tangent (Tanh) Function")
plt.grid()
plt.legend()

# Wyświetlenie wykresów
plt.tight_layout()
#zplt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definicja funkcji Softmax
def softmax(x1, x2):
    exp_values = np.exp([x1, x2])
    return exp_values / np.sum(exp_values, axis=0)

# Generowanie siatki punktów
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)

# Obliczanie wartości Softmax dla dwóch wejść
Z1, Z2 = softmax(X1, X2)

# Tworzenie wykresu 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Rysowanie powierzchni dla obu wyjść Softmax
ax.plot_surface(X1, X2, Z1, cmap="Blues", alpha=0.6, edgecolor='k')
ax.plot_surface(X1, X2, Z2, cmap="Oranges", alpha=0.6, edgecolor='k')

# Opisy osi
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("Softmax Probability")
ax.set_title("Softmax Function Visualization for Two Inputs")

#plt.show()
import pandas as pd

# Tworzenie tabeli funkcji aktywacji
activation_functions = pd.DataFrame({
    "Funkcja": ["Sigmoid", "ReLU", "Tanh", "Softmax"],
    "Zakres wartości": ["(0,1)", "[0,∞)", "(-1,1)", "(0,1) (dla każdej klasy)"],
    "Zalety": [
        "Interpretacja jako prawdopodobieństwo",
        "Szybkie obliczenia, brak zanikania gradientu",
        "Symetryczność względem zera",
        "Klasyfikacja wieloklasowa"
    ],
    "Wady": [
        "Zanikanie gradientu",
        "Martwe neurony",
        "Zanikanie gradientu",
        "Wrażliwość na duże wartości wejściowe"
    ],
    "Zastosowanie": [
        "Klasyfikacja binarna",
        "Warstwy ukryte w CNN, MLP",
        "Warstwy ukryte w RNN",
        "Warstwa wyjściowa w klasyfikacji wieloklasowej"
    ]
})

# Wyświetlenie tabeli
#print(activation_functions)


import matplotlib.pyplot as plt
import numpy as np

# Dane do wykresu
functions = ["Sigmoid", "ReLU", "Tanh", "Softmax"]
applications = ["Klasyfikacja binarna", "Warstwy ukryte w CNN, MLP", 
                "Warstwy ukryte w RNN", "Warstwa wyjściowa w klasyfikacji wieloklasowej"]

# Wykres słupkowy
plt.figure(figsize=(10, 5))
plt.barh(functions, np.arange(len(functions)), color=['blue', 'red', 'green', 'orange'])

# Opisy osi
plt.xlabel("Zastosowanie")
plt.ylabel("Funkcja aktywacji")
plt.title("Porównanie funkcji aktywacji")
plt.yticks(ticks=np.arange(len(functions)), labels=applications)

#plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Tworzenie danych w formie binarnej do wizualizacji
data = pd.DataFrame({
    "Funkcja": ["Sigmoid", "ReLU", "Tanh", "Softmax"],
    "Zakres (0-1)": [1, 0, 0, 1],
    "Symetryczność": [0, 0, 1, 0],
    "Brak zanikania gradientu": [0, 1, 0, 0],
    "Martwe neurony": [0, 1, 0, 0],
    "Klasyfikacja binarna": [1, 0, 0, 0],
    "Klasyfikacja wieloklasowa": [0, 0, 0, 1],
    "Warstwy CNN/MLP": [0, 1, 0, 0],
    "Warstwy RNN": [0, 0, 1, 0]
})

# Ustawienie "Funkcja" jako indeks
data.set_index("Funkcja", inplace=True)

# Tworzenie heatmapy
plt.figure(figsize=(10, 5))
sns.heatmap(data, annot=True, cmap="coolwarm", linewidths=0.5, fmt="d", cbar=False)

# Ustawienia etykiet
plt.title("Porównanie funkcji aktywacji")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

#plt.show()


def colorize(val):
    """Kolorowanie tekstu w tabeli"""
    if val == "Tak":
        return "background-color: lightgreen"
    elif val == "Nie":
        return "background-color: lightcoral"
    return ""

# Tworzenie DataFrame z opisowymi wartościami
df = pd.DataFrame({
    "Funkcja": ["Sigmoid", "ReLU", "Tanh", "Softmax"],
    "Symetryczność względem 0": ["Nie", "Nie", "Tak", "Nie"],
    "Zanikanie gradientu": ["Tak", "Nie", "Tak", "Nie"],
    "Szybkie obliczenia": ["Nie", "Tak", "Nie", "Tak"],
    "Zastosowanie": ["Klasyfikacja binarna", "Warstwy ukryte w CNN, MLP", 
                     "Warstwy ukryte w RNN", "Warstwa wyjściowa w klasyfikacji wieloklasowej"]
})

# Stylowanie tabeli
styler_map = df.style.map(colorize)
styler_map


def colorize(val):
    """Kolorowanie tekstu w tabeli"""
    if val == "Tak":
        return "background-color: lightgreen"
    elif val == "Nie":
        return "background-color: lightcoral"
    return ""

# Tworzenie DataFrame z opisowymi wartościami
df = pd.DataFrame({
    "Funkcja": ["Sigmoid", "ReLU", "Tanh", "Softmax"],
    "Symetryczność względem 0": ["Nie", "Nie", "Tak", "Nie"],
    "Zanikanie gradientu": ["Tak", "Nie", "Tak", "Nie"],
    "Szybkie obliczenia": ["Nie", "Tak", "Nie", "Tak"],
    "Zastosowanie": ["Klasyfikacja binarna", "Warstwy ukryte w CNN, MLP", 
                     "Warstwy ukryte w RNN", "Warstwa wyjściowa w klasyfikacji wieloklasowej"]
})

import pandas as pd

# Funkcja do kolorowania komórek
def colorize(val):
    if val == "Tak":
        return "background-color: lightgreen"
    elif val == "Nie":
        return "background-color: lightcoral"
    return ""

# Tworzenie DataFrame z opisowymi wartościami
df = pd.DataFrame({
    "Funkcja": ["Sigmoid", "ReLU", "Tanh", "Softmax"],
    "Symetryczność względem 0": ["Nie", "Nie", "Tak", "Nie"],
    "Zanikanie gradientu": ["Tak", "Nie", "Tak", "Nie"],
    "Szybkie obliczenia": ["Nie", "Tak", "Nie", "Tak"],
    "Zastosowanie": ["Klasyfikacja binarna", "Warstwy ukryte w CNN, MLP", 
                     "Warstwy ukryte w RNN", "Warstwa wyjściowa w klasyfikacji wieloklasowej"]
})

# Stylowanie tabeli
styled_df = df.style.map(colorize)
styled_df

import pandas as pd

# Tworzenie DataFrame
df = pd.DataFrame({
    "Funkcja": ["Sigmoid", "ReLU", "Tanh", "Softmax"],
    "Symetryczność względem 0": ["Nie", "Nie", "Tak", "Nie"],
    "Zanikanie gradientu": ["Tak", "Nie", "Tak", "Nie"],
    "Szybkie obliczenia": ["Nie", "Tak", "Nie", "Tak"],
    "Zastosowanie": ["Klasyfikacja binarna", "Warstwy ukryte w CNN, MLP", 
                     "Warstwy ukryte w RNN", "Warstwa wyjściowa w klasyfikacji wieloklasowej"]
})

# Funkcja mapująca kolory
color_map = {
    "Tak": "background-color: lightgreen",
    "Nie": "background-color: lightcoral"
}

# Stylowanie tabeli
styler_map = df.style.map(lambda x: color_map.get(x, ""))

# Wyświetlenie w Jupyter Notebook
styled_df

import pandas as pd

# Tworzenie DataFrame
df = pd.DataFrame({
    "Funkcja": ["Sigmoid", "ReLU", "Tanh", "Softmax"],
    "Symetryczność względem 0": ["Nie", "Nie", "Tak", "Nie"],
    "Zanikanie gradientu": ["Tak", "Nie", "Tak", "Nie"],
    "Szybkie obliczenia": ["Nie", "Tak", "Nie", "Tak"],
    "Zastosowanie": ["Klasyfikacja binarna", "Warstwy ukryte w CNN, MLP", 
                     "Warstwy ukryte w RNN", "Warstwa wyjściowa w klasyfikacji wieloklasowej"]
})

# Funkcja mapująca kolory
color_map = {
    "Tak": "background-color: lightgreen",
    "Nie": "background-color: lightcoral"
}

# Stylowanie tabeli
styled_df = df.style.map(lambda x: color_map.get(x, ""))

# Wyświetlenie w Jupyter Notebook
styled_df
df.to_excel("funkcje_aktywacji.xlsx", index=False)
import pandas as pd

df = pd.read_excel("funkcje_aktywacji.xlsx", engine="openpyxl")
print(df)


import pandas as pd

# Tworzenie danych
data = {
    "Funkcja": ["Sigmoid", "ReLU", "Tanh", "Softmax"],
    "Zakres wartości": ["(0, 1)", "[0, ∞)", "(-1, 1)", "(0, 1) (dla każdej klasy)"],
    "Zalety": [
        "Interpretacja jako prawdopodobieństwo",
        "Szybkie obliczenia, brak zanikania gradientu",
        "Symetryczność względem zera",
        "Klasyfikacja wieloklasowa"
    ],
    "Wady": [
        "Zanikanie gradientu",
        "Martwe neurony",
        "Zanikanie gradientu",
        "Wrażliwość na duże wartości wejściowe"
    ],
    "Zastosowanie": [
        "Klasyfikacja binarna",
        "Warstwy ukryte w CNN, MLP",
        "Warstwy ukryte w RNN",
        "Warstwa wyjściowa w klasyfikacji wieloklasowej"
    ]
}

# Tworzenie DataFrame
df = pd.DataFrame(data)

# Wyświetlenie tabeli
print(df)
df.to_excel('funkcje_aktywacji.xlsx', index=False)
df.to_excel('funkcje_aktywacji.xlsx', index=False)

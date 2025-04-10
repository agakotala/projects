import pandas as pd

# Wczytanie danych z pliku CSV
df = pd.read_csv('plik.csv')

# Wczytanie z pliku Excel (xlsx)
df_excel = pd.read_excel('plik.xlsx', sheet_name='Arkusz1')

df.head()      # Domyślnie wyświetla 5 pierwszych wierszy
df.head(10)    # Wyświetla 10 pierwszych wierszy

df.tail()      # Domyślnie wyświetla 5 ostatnich wierszy
df.tail(10)    # 10 ostatnich wierszy

df.info()      # Wyświetla liczbę wierszy, kolumn, typy danych oraz liczbę niepustych rekordów w każdej kolumnie

df.describe()  # Opis statyczny

df['nazwa kolumny']  # Zwraca Series z danej kolumny 
df[['kolumna1', 'kolumna2']] # Zwraca DataFrame z wybranych kolumn 

df[df['kolumna'] == 'wartość']
df[df['kolumna'] > 10]
df[(df['kolumna1'] > 5 ) & (df['kolumna2'] == 'abc')]

df.loc[0:10, ['kolumna1', 'kolumna2']] # wiersze od 0 do 10, tylko dwie kolumny
df.iloc[0:10, 0:2] # wiersze od 0 do 10, kolumny od 0 do 2

df['kolumna_numeryczna'].sum()
df.sum() # domyślnie wykona sume dla każdej kolumny numerycznej 
df['kolumna'].count() # zwraca liczbę niepustych (nie-nan) wartości w kolumnie
df.count()
df['kolumna_numeryczna'].mean()
df['kolumna_numeryczna'].median()
df['kolumna_numeryczna'].std()
df['kolumna_kategoryczna'].unique() #lista unikalnych wartości 
df['kolumna_kategoryczna'].nunique() #lisa unikalnych wartosci 

grupa = df.grouphy('kolumna_kategoryczna')
grypa['kolumna_numeryczna'].mean()
#zwraca średnie wartości kolumny numerycznej w przedziale na unikalne wartości w kolumnie kategorycznej 



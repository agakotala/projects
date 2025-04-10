#import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#df = pd.read_csv(r'C:\Users\kstor\Downloads\Big_data_2425-main (1)\Big_data_2425-main\diabetes.csv')
#print(f'wpisuje wszystko albo i nie wszystko:\n{df}')
#print(f'ilość danych: {df.shape}\nliczba kolumn: {df.shape[1]}')
#print(f'wpisuję tak, jak chcę:\n{df.head(3).to_string()}')
#print(f'opis danych:\n{df.describe()}')
#print(f'ilość pustych komórek: \n{df.isna().to_string}')


#df['bmi'] +=1000
#df['nowa_testowa'] = df['bmi'] / df['glucose'] - 50 * df.shape[1]
#print(f'opis danych:\n{df.describe().T.to_string}')

#df['bmi'] = df['bmi'].replace(0, np.nan)
#for col in ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'diabetespedigreefunction', 'age']:
    #df[col].replace(0, np.nan, inplace=True) 
    #mean_ = df[col].mean()
    #df[col].replace(np.nan, mean_, inplace=True)


#print('po czyszczeniu danych')
#print(df.describe().T.to_string())
#print(df.isna().sum())


X = df.iloc[:, :-1]
y = df.outcome


X_train, X_test, y_trian, y_test = train_test_split(:X, y, test_size=0.2)
model = LogisticRegression
model.fit(X_trian, y_train)
print(f'dokładność modelu {model.score(X_testm y_test)}')






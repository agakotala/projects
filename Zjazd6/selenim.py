import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import pandas as pd

X, y = make_circles(n_samples=1000, factor=0.8, noise=0.2)
print(X)
print(y)
plt.scatter(X[:, 0], X[:, 1]   , c=y  )
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  #clasifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

print('Regresja logistyczna')
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

print('Drzewo decyzyjne')
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

print('kNeiborsClassifier')
model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

print('SVC')
model = SVC(kernel='rbf')  #tylko radial base function, więcej wymiarów
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

from selenium import webdriver
from selenium.webdriver import Keys
from time import sleep

okno1_chrome = webdriver.Chrome()
okno2_chrome = webdriver.Chrome()

okno1_chrome.get('https://www.google.com/')
okno2_chrome.get('https://allegro.pl')

sleep(3)
okno1_chrome.find_element('id','L2AGLb').click()
sleep(3)

search_field = okno1_chrome.find_element('name','q')
search_field.clear()
search_field.send_keys('Czy chat GPT opanuje świat?')
search_field.send_keys(Keys.ENTER)

sleep(3)


okno1_chrome.close()
okno2_chrome.close()
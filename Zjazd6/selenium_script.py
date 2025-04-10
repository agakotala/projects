from selenium import webdriver
from selenium.webdriver import Keys
from time import sleep

from selenium.webdriver.chrome.service import Service

service = Service(r'C:\Users\kstor\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe')
driver = webdriver.Chrome(service=service)


okno1_chrome = webdriver.Chrome()
okno2_chrome = webdriver.Chrome()

okno1_chrome.get('https://www.google.com/')
okno2_chrome.get('https://allegro.pl')

sleep(3)
okno1_chrome.find_element('id','L2AGLb').click()git init
sleep(3)

search_field = okno1_chrome.find_element('name','q')
search_field.clear()
search_field.send_keys('Czy chat GPT opanuje świat?')
search_field.send_keys(Keys.ENTER)

sleep(3)


okno1_chrome.close()
okno2_chrome.close()
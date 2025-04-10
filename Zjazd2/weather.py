from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from datetime import datetime


PATH = r"C:\Users\kstor\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
service = Service(PATH)

def get_temperature(city="Warszawa"):
    try:
        driver = webdriver.Chrome(service=service)
        driver.get("https://weather.com")
        
       
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "LocationSearch_input")))

        search = driver.find_element(By.ID, "LocationSearch_input")
        search.send_keys(city)
        search.send_keys(Keys.RETURN)

        
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//span[contains(@data-testid, 'TemperatureValue')]")))

        temp_element = driver.find_element(By.XPATH, "//span[contains(@data-testid, 'TemperatureValue')]")
        temperature = temp_element.text
    except Exception as e:
        print(f"Błąd: {e}")
        temperature = "Nie znaleziono"
    finally:
        driver.quit()
    
    return temperature

def zapisz_do_pliku(miasto, temperatura):
    with open("pogoda_log.txt", "a", encoding="utf-8") as f:
        teraz = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{teraz}] {miasto}: {temperatura}\n")


while True:
    miasto = "Warszawa"
    temp = get_temperature(miasto)
    zapisz_do_pliku(miasto, temp)
    print(f"Zapisano: {miasto} - {temp}")
    
   
    time.sleep(3600)

def get_temperature(city="Warszawa"):
        
    try:
        driver = webdriver.Chrome(service=service)
        driver.get("https://weather.com")
        
        print("Oczekiwanie na załadowanie strony...")
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "LocationSearch_input")))

        print("Wyszukiwanie miasta...")
        search = driver.find_element(By.ID, "LocationSearch_input")
        search.send_keys(city)
        search.send_keys(Keys.RETURN)

        print("Oczekiwanie na załadowanie temperatury...")
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//span[@data-testid='TemperatureValue']")))

        temp_element = driver.find_element(By.XPATH, "//span[@data-testid='TemperatureValue']")
        temperature = temp_element.text
    except Exception as e:
        print(f"Błąd: {e}")
        temperature = "Nie znaleziono"
    finally:
        driver.quit()
    
    return temperature

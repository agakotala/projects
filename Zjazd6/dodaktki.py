import os
from time import sleep

print(os.getcwd())
os.chdir(r"c:/Users/kstor/Downloads/Big_data_2425-main (1)")
print(os.getcwd())
os.mkdir('BigData')
sleep(2)
os.rename('Bigdata', 'Nowy')
sleep(2)
os.rmdir('Nowy')

os.system("cmd \c 'dir'")
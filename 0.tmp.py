for i in range(5):
    print('siema')
print()
for i in range(3,7):
    print(f'siema{i}')

word = 'piesek'
for i in word:
    print(i)

dogs_names = ['Luna', 'Tosia', 'Max', 'Barley']
for dog_name in dogs_names:
    print(f'imię to: {dog_name}')
    if dog_name[-1] == 'a':
        print(f'Pani {dog_name}')
    else:
        print(f'Pan {dog_name}')
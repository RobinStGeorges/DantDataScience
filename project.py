import random
#lire le tab et reecire le tab ds un fichier avec les lignes melangés
def my_shuffle(file):
    lines = open(file).readlines()
    random.shuffle(lines)
    open('test.txt', 'w').writelines(lines)

my_shuffle('original.txt')
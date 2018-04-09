# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:56:38 2018

@author: 3200107
"""
import numpy

def lectureFichiertab(s):
    monFichier = open(s, "r") 
    contenu = monFichier.readlines() 
    monFichier.close() 
    for i in range(0,len(contenu)):
        contenu[i]=contenu[i].split(',')
        contenu[i][len(contenu[i])-1]=contenu[i][len(contenu[i])-1][0] #enléve le \n de la derniére valeur
    return contenu

def splitTrainTest(tab,percen):
    taille = len(tab)
    nbTrain = int(taille*percen)
    trainTab=tab[:nbTrain]
    testTab=tab[nbTrain:]
    return trainTab,testTab
    
def delColone(tab,liste):
    liste.sort()
    liste.reverse()
    for i in tab:
        for j in liste:
            del i[j]
    return tab


    
test = lectureFichiertab("spambase.data")
print(len(test))
tabTrain,tabTest = splitTrainTest(test,0.1)
print(tabTrain[0])
print(delColone(tabTrain,[3,2,1])[0])


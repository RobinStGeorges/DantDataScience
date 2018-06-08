# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:56:38 2018

@author: 3200107
"""
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
import pandas as panda
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection


def lectureFichiertab(s):
    monFichier = open(s, "r") 
    contenu = monFichier.readlines() 
    monFichier.close() 
    for i in range(0,len(contenu)):
        contenu[i] = contenu[i].split(',')
        contenu[i][len(contenu[i])-1] = contenu[i][len(contenu[i])-1][0] #enléve le \n de la derniére valeur
    return panda.DataFrame(contenu)


def splitTrainTest(tab,percen):
    taille = len(tab)
    nbTrain = int(taille*percen)
    trainTab = tab.iloc[:nbTrain]
    testTab = tab.iloc[nbTrain:]
    return trainTab,testTab
    
def getColone(tab,liste):
    return tab[liste]

def knnation(K,tab,percent):
    tabTrain, tabTest = splitTrainTest(tab,percent)
    YTrain=tabTrain[57]
    YTest = tabTest[57]
    XTest = getColone(tabTest,[8,15,16,19,23,39,52,51,54])
    XTrain = getColone(tabTrain,[8,15,16,19,23,39,52,51,54])
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(XTrain,YTrain)
    nbBonResultat = 0

    for i in range(0,XTest.shape[0]):
        if neigh.predict([XTest.iloc[i]]) == YTest.iloc[i]:
            nbBonResultat = nbBonResultat + 1


    percentResult = nbBonResultat/XTest.shape[0]
    return percentResult

def plotknnKVariation(tab):
    plotab = [0] * 50
    for i in range(0,50):
        plotab[i] = knnation(i+1,tab,0.7)
        print(i)
        print(plotab[i])
    tabinator = [0] * 50
    for i in range(0,50):
        tabinator[i] = i+1
    plt.plot(tabinator, plotab)
    plt.show()

def plotknnPercentVariation(tab):
    plotab = [0] * 49
    j = 0
    for i in range(2,100,2):
        plotab[j] = knnation(30, tab, i/100)
        print(j,i)
        print(plotab[j])
        j=j+1
    tabinator = [0] * 49
    j = 0
    for i in range(2, 100, 2):
        tabinator[j] = i + 1
        j = j + 1
    plt.plot(tabinator, plotab)
    plt.show()


def kmeannation(tab,percent,a):
    tabTrain, tabTest = splitTrainTest(tab, percent)
    YTrain = tabTrain[57]
    YTest = tabTest[57]
    XTest = getColone(tabTest,[8,15,16,19,23,38,53,20,25])
    XTrain = getColone(tabTrain,[8,15,16,19,23,38,53,20,25])

    kmeans = KMeans(n_clusters=2,init='k-means++',n_init=1,max_iter=10, random_state=0).fit(XTrain)
    # print(kmeans.cluster_centers_)
    nbBonResultat = 0
    for i in range(0,XTest.shape[0]):

        if kmeans.predict([XTest.iloc[i]])[0] == np.int32(YTest.iloc[i]):
            nbBonResultat = nbBonResultat + 1

    percentResult = nbBonResultat / XTest.shape[0]
    return percentResult

def graphKmean(test):
        a = 2
        d = 0
        liste1 = []
        liste2 = []
        pourcent = 0.3
        for d in range(20):
            x = d
            y = kmeannation(test, pourcent, a)
            print("--------")
            print(x)
            print(y)
            print("--------")
            liste1.append(x)
            liste2.append(y)
            a = a + 2
            if (pourcent + 0.030) < 1:
                pourcent = pourcent + 0.0305
            print(pourcent)
        print("---Les liste---")
        print(liste1)
        print(liste2)
        plt.plot(liste1, liste2)
        plt.show()




def svmation(tab, percent):
    tabTrain, tabTest = splitTrainTest(tab, percent)
    YTrain = tabTrain[57]
    YTest = tabTest[57]
    XTest = getColone(tabTest, [0, 3, 4, 5, 6, 7, 8])
    XTrain = getColone(tabTrain, [0, 3, 4, 5, 6, 8, 9])

    clf = svm.SVC()
    clf.fit(XTrain, YTrain)
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    nbBonResultat = 0
    for i in range(0, XTest.shape[0]):
        if clf.predict([XTest.iloc[i]]) == YTest.iloc[i]:
            nbBonResultat = nbBonResultat + 1

    percentResult = nbBonResultat / XTest.shape[0]
    return percentResult



def plotsvmPercentVariation(tab):
    plotab = [0] * 49
    j = 0
    for i in range(2, 100, 2):
        plotab[j] = svmation( tab, i / 100)
        print(j, i)
        print(plotab[j])
        j = j + 1
    tabinator = [0] * 49
    j = 0
    for i in range(2, 100, 2):
        tabinator[j] = i + 1
        j = j + 1
    plt.plot(tabinator, plotab)
    plt.show()

def randomForest(tab,percent):
    tabTrain, tabTest = splitTrainTest(tab, percent)
    YTrain = tabTrain[57]
    YTest = tabTest[57]
    XTest = getColone(tabTest,[8,15,16,19,23,39,52,51,54])
    XTrain = getColone(tabTrain,[8,15,16,19,23,39,52,51,54])

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(XTrain, YTrain)
    nbBonResultat = 0
    for i in range(0, XTest.shape[0]):
        if clf.predict([XTest.iloc[i]]) == YTest.iloc[i]:
            nbBonResultat = nbBonResultat + 1

    percentResult = nbBonResultat / XTest.shape[0]
    return percentResult

def percentRandomForest(tab):
    plotab = [0] * 49
    j = 0
    for i in range(2, 100, 2):
        plotab[j] = randomForest(tab, i / 100)
        print(j, i)
        print(plotab[j])
        j = j + 1
    tabinator = [0] * 49
    j = 0
    for i in range(2, 100, 2):
        tabinator[j] = i + 1
        j = j + 1
    plt.plot(tabinator, plotab)
    plt.show()


def adaBoostamination(tab,percent,a):
    print("def")
    tabTrain, tabTest = splitTrainTest(tab, percent)
    YTrain = tabTrain[57]
    YTest = tabTest[57]
    XTest = getColone(tabTest,[8,15,16,19,23,38,53,20,25])
    XTrain = getColone(tabTrain,[8,15,16,19,23,38,53,20,25])

    seed = 7
    num_trees = 30
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, XTest, YTest, cv=kfold)
    print(results.mean())
    model =  AdaBoostClassifier.fit(model,XTest,YTest)

    nbBonResultat = 0
    for i in range(0,XTest.shape[0]):
        print("||||||PREDICTIONS||||||")
        print(([XTest.iloc[i]])[0])
        print("VS")
        print(np.int32(YTest.iloc[i]))
        print("|||||||||||||||||||||||")
        if model.predict([XTest.iloc[i]])[0] == (YTest.iloc[i]):
            print("if")
            nbBonResultat = nbBonResultat + 1

    percentResult = nbBonResultat / XTest.shape[0]
    return percentResult

test = lectureFichiertab("mixed.txt")


def graphAdda():
    a=2
    d=0
    liste1=[]
    liste2=[]
    pourcent=0.3
    for d in range (20):
        x=d
        y=adaBoostamination(test,pourcent,a)
        print("--------")
        print(x)
        print(y)
        print("--------")
        liste1.append(x)
        liste2.append(y)
        a=a+2
        if (pourcent+0.030) < 1 :
            pourcent=pourcent+0.0305
        print(pourcent)
    print("---Les liste---")
    print(liste1)
    print(liste2)
    plt.plot(liste1,liste2)
    plt.show()




test = lectureFichiertab("mixed.data")
percentRandomForest(test)
#print(randomForest(test,0.8))


#print( knnation(10,test,0.91) )
#plotsvmPercentVariation(test)
#plotknnPercentVariation(test)
#plotknnKVariation(test)


#print(knnation(1,test,0.5) )
#print(kmeannation(test,0.3) )
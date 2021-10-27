# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:52:45 2021

@author: 33781
"""
import csv
import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt

def LoadDataSet(split, trainingSet, testSet):
    with open('data.csv','r') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines) 
        #print(dataset) #matrice 803x7
        #print(len(dataset)) #=803 
        for x in range (len(dataset)):
            for y in range(6):
                dataset [x][y]=float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
        #print(len(testSet)+len(trainingSet))
       #sépare dataset entre trainingSet et testSet au taux de split dans trainingSet et (1-split) dans testsSet
#LoadDataSet(0.5,[],[])       

def LoadDataSetfinal(trainingSet, testSet):
    with open('data.csv','r') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        #print(len(dataset))
        for x in range (len(dataset)):
            for y in range(6):
                dataset [x][y]=float(dataset[x][y])
            trainingSet.append(dataset[x])
        #print(len(trainingSet))    
    with open('preTest.csv','r') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range (len(dataset)):
            for y in range(6):
                dataset [x][y]=float(dataset[x][y])
            trainingSet.append(dataset[x])
        #print(len(trainingSet))    
    with open('finalTest.csv','r') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range (0,len(dataset)):
            for y in range(6):
                dataset [x][y]=float(dataset[x][y])
            testSet.append(dataset[x])
        #print(len(testSet))    


#LoadDataSetfinal([],[])                
     
       
def Distance(instance1, instance2, length): #distance euclidienne
        dist=0
        for i in range(length):
            calcul= instance1[i] - instance2[i]
            dist= dist + pow(calcul, 2)
        return math.sqrt(dist)


def Distancepond(instance1, instance2, length):
        dist=0
        for i in range(length):
            if i==0:
                calcul= (instance1[i] - instance2[i])/25.5
            elif i==1:
                calcul= (instance1[i] - instance2[i])/5.713
            elif i==2:
                calcul= (instance1[i] - instance2[i])
            elif i==1:
                calcul= (instance1[i] - instance2[i])/7.4  
            else:
                calcul=( instance1[i] - instance2[i])/1.36
            dist= dist + pow(calcul, 2)
        return math.sqrt(dist)
    
def Distancepond2(instance1, instance2, length): 
        dist=0
        for i in range(length):
            if i==0:
                calcul= (instance1[i] - instance2[i])/6.863
            elif i==1:
                calcul= (instance1[i] - instance2[i])/1.489
            elif i==2:
                calcul= (instance1[i] - instance2[i])
            elif i==1:
                calcul= (instance1[i] - instance2[i])/2.014 
            else:
                calcul=( instance1[i] - instance2[i])/0.0969
            dist= dist + pow(calcul, 2)
        return math.sqrt(dist)


def Knn(trainingSet, testInstance, k): 
    list_distances=[]
    length=len(testInstance)-1
    for i in range(len(trainingSet)):
        dist=Distance(testInstance, trainingSet[i], length)
        list_distances.append((trainingSet[i],dist))
    list_distances.sort(key=operator.itemgetter(1)) 
    list_neighbors=[]
    for i in range(k):
        list_neighbors.append(list_distances[i][0])
    return list_neighbors
    

def getResponse(list_neighbors):
    classVotes={}
    for i in range(len(list_neighbors)):
        resp= list_neighbors[i][-1]
        if resp in classVotes:
            classVotes[resp]+=1
        else:
            classVotes[resp] = 1 
    sortedVotes= sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    #print(classVotes)
    return sortedVotes[0][0]

    
    
def Accuracy(testSet, predictions): #précision des prédictions
    correct=0
    for i in range(len(testSet)):
        #if testSet[i][-1] is predictions[i]:
        if testSet[i][-1]== predictions[i]:
            correct= correct + 1
    return(correct/float(len(testSet)))*100.0


def Main(k):
    # prepare data
    trainingSet=[]
    testSet=[]
    split=0.90
    LoadDataSet(split, trainingSet, testSet)
    predictions=[]   
    for x in range(len(testSet)):
        list_neighbors=Knn(trainingSet, testSet[x], k)
        resultat= getResponse(list_neighbors)
        predictions.append(resultat) 
    accuracy= Accuracy(testSet, predictions)
    print('Accuracy: ' + str(accuracy) + '%')
    return accuracy
    

def Mainfinal():
    k=5 #k performant
    trainingSet=[]
    testSet=[] 
    LoadDataSetfinal(trainingSet, testSet)
    predictions=[]
    fichier = open('Ourradour_sample.txt','w')
    for x in range(len(testSet)):
        list_neighbors=Knn(trainingSet, testSet[x], k)
        resultat= getResponse(list_neighbors)
        predictions.append(resultat)
    for i in predictions:
        fichier.write(str(i)+'\n')
    #print(len(predictions)) 3000 c'est bon  
    fichier.close()
   
    
Mainfinal()
    
def influencek(kinf,kmax,nbfois,pas):
    nbfois+=1
    y=[]
    x=[x for x in range(kinf,kmax,pas)]
    moy=0
    for i in range(kinf,kmax,pas):
        for j in range(nbfois+1):
            moy+=Main(i)
        y.append(moy/(nbfois+1))    
        print('la précision pour k = ' + str(i)+ ' est de '+ str(moy/(nbfois+1)) + '%')
        moy=0
    #print(x) test
    #print(y) test
    return(x,y)  
    
#graphek(1,20,20,2)

def graphek(kinf,kmax,nbfois,pas):
    x,y=influencek(kinf,kmax,nbfois,pas)    
    plt.plot(x,y,'o')
    plt.ylabel('Présicision des prédictions (en %)')
    plt.xlabel('Valeurs de k')
    plt.title('Influence de k sur la précision des prédictions pour k compris entre '+str(kinf)+' et ' + str(kmax) )
    plt.show

#graphek(1,102,50,5)
def difference(instance1, instance2, length): #distance euclidienne
        diff=[]
        for i in range(length):
            calcul=abs( instance1[i] - instance2[i])
            diff.append(calcul)         
        return diff    
    
def calculdistancemoy():
    moydis=np.array([0.0,0.0,0.0,0.0,0.0])
    moydisfinal=np.array([0.0,0.0,0.0,0.0,0.0])
    dataA=[]
    dataB=[]
    dataC=[]
    dataD=[]
    dataE=[]
    with open('data.csv','r') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines) 
        #print(dataset) matrice 803x7
        #print(len(dataset)) #=803 
        for x in range (len(dataset)-1):
            for y in range(6):
                dataset[x][y]=float(dataset[x][y])
            if  dataset[x][-1]== 'classA':
                dataA.append(dataset[x])
            elif dataset [x][-1]== 'classB':
                dataB.append(dataset[x])
            elif dataset [x][-1]== 'classC':
                dataC.append(dataset[x])
            elif dataset [x][-1]== 'classD':
                dataD.append(dataset[x])
            else:
                dataE.append(dataset[x])
        
#        print('----------------------------------------------------------------------------------------------------------------------------')
#        print(dataA)
#        print('--------------------------------------------------------------------------------------------------------------------------------')
#        print(dataB)
#        print('--------------------------------------------------------------------------------------------------------------------------------')
#        print(dataC)
#        print('--------------------------------------------------------------------------------------------------------------------------------')
#        print(dataD)
#        print('--------------------------------------------------------------------------------------------------------------------------------')
#        print(dataE)
    for i in range(len(dataA)-1):
             for j in range(len(dataB)-1):
                 moydis+=np.array(difference(dataA[i],dataB[j],5))
                 #moydis+=np.array(difference(dataA[i],dataC[j],5))
                 #moydis+=np.array(difference(dataA[i],dataD[j],5))
                 #moydis+=np.array(difference(dataA[i],dataE[j],5))
             moydis /= (len(dataA))        
             print(moydis)
             moydisfinal+=moydis
             moydis=0
    moydisfinal/= len(dataA)    
    print('moydisfinal est de' + str(moydisfinal)) #moydisfinal est de-0.0002762850864049545  dataA/dataA 
                      
calculdistancemoy()
                     
def ponderation2():
     pond=np.array([.0,.0,.0,.0,.0])
     with open('finalTest.csv','r') as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines) 
        #print(dataset) matrice 803x7
        #print(len(dataset)) #=803 
        for x in range (len(dataset)-1):
            for y in range(6):
                dataset[x][y]=float(dataset[x][y])
            pond+=np.array(dataset[x][0:5])
        print(pond/len(dataset))

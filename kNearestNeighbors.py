# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import math
import time as time

#Global varibles (idk if Python actually has those)
kNearestLabels = []



#Takes child node index and returns parent node index.
def Parent(i):
     return math.floor(i  / 2)


#Returns Left child of the parent node whose index is specified by i.
def Left(i):
    return((2*i) + 1);


#Returns Right child of the parent node whose index is specified by i.
def Right(i):
    return((2*i) + 2)


#Max Heapify
def Max_Heapify(A, i, n):
    largest = -1
    max_heap = []
    
    L = Left(i)
    R = Right(i)

        
    if (L <= n) and (A[0][L] > A[0][i]):

      largest = L
    
    else:
      largest = i
      
    
    if (R <= n) and (A[0][R] > A[0][largest]):
      largest = R

    
    #A[0] becuase I created a list of lists...Realistically you just need a list.
    #I'm just bad at Python.
    if largest != i:
      #I have two lists. One for distances and one for classes
      #distances correspond to classes, so have to perform same movement operations.
      #I don't think you need to do it this way, I was just trying to efficient as possible,
      #and indexing in multi-column arrays is a little more time consuming (at least in R)
      tempDistance = A[0][i]
      tempClassLabel = kNearestLabels[0][i]
      
      A[0][i] = A[0][largest]
      kNearestLabels[0][i] = kNearestLabels[0][largest]
      
      A[0][largest] = tempDistance
      kNearestLabels[0][largest] = tempClassLabel
      
      max_heap = Max_Heapify(A, largest, n)
      
      
      return(max_heap)
      

    return(A)




def Build_Max_Heap(A, n):
  
  i = math.floor(n/2) -1

  
  while(i >= 0):
    A = Max_Heapify(A, i, n)
    i = i - 1


  return(A)
  



########## BEGINNING OF MAIN ##############################

#Time execution
start_time = time.time()

##Get the training data set.
trainingSet = pd.read_csv('./Data/TrainData.csv', delimiter=',')

##Testing set
testingSet = pd.read_csv('./Data/TestData.csv', delimiter=',')






#Create a vector which will hold the guessed label for all testing data rows
guessedLabels = []



###I WAS USING THE FOLLOWING COMMENTED-OUT STUFF TO TEST MY ALGORITHM.
###ITS JUST TAKING FIRST ROWS IN EACH QUARTER OF DATA SET SO WE GET A GOOD SAMPLE POOL

#fourthOfTrainData = math.floor( (1/4) * math.floor(len(trainingSet)))

#trainingSamplePool = trainingSet.iloc[0:5000]

#trainingSamplePool = trainingSamplePool.append(trainingSet.iloc[fourthOfTrainData:(fourthOfTrainData+5000)])

#trainingSamplePool = trainingSamplePool.append(trainingSet.iloc[(2*fourthOfTrainData):(2*fourthOfTrainData+5000)])

#trainingSamplePool = trainingSamplePool.append(trainingSet.iloc[(3*fourthOfTrainData):(3*fourthOfTrainData+5000)])


#trainingSamplePool eventually needs to be the entire training set
trainingSamplePool = trainingSet


#Numpy array
trainingSamplePool = trainingSamplePool.to_numpy()

#Numpy array conversion
testingSet = testingSet.to_numpy()






#DECLARING VARIABLES OUTSIDE THE LOOP. I THINK IT HELPS TIME EFFICIENCY BUT IM NOT 
#ACTUALLY SURE IN PYTHON
#-------------------------------------------------------
#Used to hold the euclidean distances between all training instances and 1 testing instance
distances = []


euclideanDistance = 0

count = 1

trainingInstanceNumber = 0

trainingLabel = []


#Will hold the K Nearest Neighbors for 1 testing instance
kNearestNeighbors_MaxHeap = []


k = math.floor(math.sqrt(len(trainingSamplePool))) #square root of n is optimal according to google
#^So we will have sqrt(n) neighbors

#Placing these variables here because declaring variables in the loop takes more time (I think)
trainingInstanceRowNumber = 1

labels = [] 

kNearestLabels = []

kNearestNeighbors_MaxHeap = []

euclideanDistance = 0

n_minus_k_dataSet = []
n_minus_k_labels = []


positiveOne = 0
negativeOne = 0

count <- 1

#-------------------------------------------------------






#I just wanted regular C-style for loops. So I just used while loops
#and made them act like for-loops 
i = 0

while i < len(testingSet):

  print("\nWorking On Test Data Instance #", i, "...")
  print("\n")

  #Pull one data instance from the testing dataset.
  testingInstance = testingSet[i, ]
  
  
  trainingInstanceRowNumber = 0
  
  
  #Have to clear distance, label, kNearestLabels, and kNearestNeighbors at each new iteration.
  distances = []
  labels = []
  kNearestLabels = []
  kNearestNeighbors_MaxHeap = []
  
  #  I don't actually think this does anything anymore. But I don't want to mess up my program somehow
  #so it's staying in.
  count = 1
  
  #ierator for next loop
  j = 0
  
  while j < len(trainingSamplePool):
    euclideanDistance = 0
    
    #Pull 1 training instance from our training sample pool (which by the end is the entire training set)
    trainingInstance = trainingSamplePool[j, ]
    
    #Could use loop, but maybe this is more time efficient? Idk tbh.
    euclideanDistance = math.sqrt((testingInstance[0] - trainingInstance[0])**2 + (testingInstance[1] - trainingInstance[1])**2 + (testingInstance[2] - trainingInstance[2])**2 + 
                                 (testingInstance[3] - trainingInstance[3])**2 + (testingInstance[4] - trainingInstance[4])**2 + (testingInstance[5] - trainingInstance[5])**2 + 
                                 (testingInstance[6] - trainingInstance[6])**2 + (testingInstance[7] - trainingInstance[7])**2 + (testingInstance[8] - trainingInstance[8])**2 + 
                                 (testingInstance[9] - trainingInstance[9])**2 + (testingInstance[10] - trainingInstance[10])**2 + (testingInstance[11] - trainingInstance[11])**2 +       
                                 (testingInstance[12] - trainingInstance[12])**2 + (testingInstance[13] - trainingInstance[13])**2 ) 
    
    
    distances.append(euclideanDistance)
    

    #column 14 is the class lable. 2 separate lists. 1 for distances and 1 for class labels.
    #Can just keep them in one list or dataframe or something, but the less complex the data structure
    #the more time efficient (or atleast I would think)
    labels.append(trainingSamplePool[trainingInstanceRowNumber, 14])

    
    trainingInstanceRowNumber = trainingInstanceRowNumber + 1    
    
    j = j + 1
    
  
  #Now find k-nearest neighbors
  #----------------------------------
  
  #place first k neighbor distances in queue
  kNearestNeighbors_MaxHeap.append(distances[0:k])
  
  #place first k neighbors class labels in this list   
  kNearestLabels.append(labels[0:k])
  

  #Initialize queue to be a max heap
  kNearestNeighbors_MaxHeap = Build_Max_Heap(kNearestNeighbors_MaxHeap, (len(kNearestNeighbors_MaxHeap[0])-1))

  
  #get the remaining n-k dataset.
  n_minus_k_dataSet = distances[k:len(distances)]
  n_minus_k_labels = labels[k:len(labels)]

  
  index = 0
  
  
  while index < len(n_minus_k_dataSet):
    #If a distance in the n-k dataset (let's call it distance A) is less than the largest distance
    #in the max-heap (let's call it distance B), then replace distance B with distance A in the heap.
    if n_minus_k_dataSet[index] < kNearestNeighbors_MaxHeap[0][0]:
      kNearestNeighbors_MaxHeap[0][0] = n_minus_k_dataSet[index]
      kNearestLabels[0][0] = n_minus_k_labels[index] 
      
      #Maintain max-heap property by calling Max-Heapify on the root node
      kNearestNeighbors_MaxHeap = Max_Heapify(kNearestNeighbors_MaxHeap, 0, (len(kNearestNeighbors_MaxHeap[0])-1))

    index = index + 1
      
     
        
  positiveOne = 0 #Counts how many 1 class labels we have in our k nearest neighbors
  negativeOne = 0 #Counts how many -1 class labels we have in our k nearest neighbors
      
  index = 0
  
  #kNearestLabels is a list of lists...but it probably doesn't have to be.
  #I think I'm just bad at Python
  while index < len(kNearestLabels[0]):
    if kNearestLabels[0][index] == 1:
      positiveOne = positiveOne + 1
    else:
      negativeOne = negativeOne + 1
    
    index = index + 1
  
  
 #if more of our k-nearest neighbors have y = 1 than y =-1  
  if positiveOne >= negativeOne:
    guessedLabels.append(1)  #then that's our guess for that testing instance's class label

  else:
    guessedLabels.append(-1)
    
    
  i = i + 1

  


#Outside of loops-----------------------------

i = 0

correct = 0

#See how many guesses we got right
while i < len(guessedLabels):
  if guessedLabels[i] == testingSet[i, 14]:
    correct = correct + 1
    
  i = i + 1

#Find accuracy
accuracy = 100*(correct / len(guessedLabels))

print("\n\n\nAccuracy is {:0.2f}%\n\n".format(accuracy))



end_time = time.time()

print("\n\nK-Nearest-Neighbors tool {:0.2f} seconds ".format(end_time - start_time))






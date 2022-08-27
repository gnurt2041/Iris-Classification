# Project classification iris flower

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import time
'''
calculate the distance between 2 vector n dimension 
i using norm 2 aka eculid distance
 '''
def eculid_dis(vector1,vector2):
    
    distance = 0.0
    
    for index in range(len(vector1)-1):
        distance += (vector1[index]-vector2[index])**2
        
    return np.sqrt(distance)

'''
Find the n(num_neighbor) of a new_data
1. calculate all the distance between 2 vector: each element of the data_train and the new_data
2. push the calculated distance of each element of the data_train into a list
3. sort the list with key is the distance 
4. finally push n first element of the list sorted into the neighbor list and return it 
'''
def find_neighbor(data_train, new_data, num_neighbor):
    
    distance = list()
    
    for i in range(len(data_train.data)-1):
        distance.append((i, eculid_dis(new_data,data_train[i])))
    distance.sort(key=lambda x:x[1])
    
    neighbor = []
    for i in range(1,num_neighbor+1):
        neighbor.append(distance[i][0])
        
    return neighbor


'''
1. find the neighbor of the new_data( I built this function to not input the num_neighbor -> default : 10)
2. find the max of target in all the neighbor(max of count number of each target -> knn algorithm)
3. return the max -> got the predict
'''
def predict(new_data, data_train, data_train_target):
    
    neighbor = find_neighbor(data_train, new_data, 10)
    all_tag = [data_train_target[index] for index in neighbor]
    pred = max(set(all_tag), key=all_tag.count)
    return pred

start_time = time.time()
iris = datasets.load_iris()
for index in range(np.max(iris.target)+1):
    x = iris.data[iris.target == index,:]
    #print(x[:5,:],i,'\n')

# split the data_train and the data_test
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=50)

# predict the data_test
for index in range(len(y_test)):
    
    print(y_test[index],end=' ')
    
print('\n')
for index in range(len(y_test)):
    
    y_pred = predict(X_test[index], X_train, y_train)
    print(y_pred,end=' ')
    
end_time = time.time()
print ("Running time: %.2f (s)" % (end_time - start_time))
#neighbor = find_neighbor(iris, iris.data[0],10)
#print(iris.data[0],neighbor)
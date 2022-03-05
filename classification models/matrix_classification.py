#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:03:18 2022

@author: m231026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
#from scipy import sparse
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import TruncatedSVD#read data with pandas
#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
#from mpl_toolkits import mplot3d
import random
import math

def coordinate_list(data_matrix):
    nz = data_matrix.nonzero()
    print(nz)
    return [ (x,y) for x, y in zip(nz[0], nz[1])]

#loss function for specific point
#row of p and column of q
def loss(d, p, q):
    #print(d, np.dot(p,q))
    return pow(d - (np.dot(p,q)), 2)

#updates values in p and q based on partial deriviatives of loss function
#row of p and column of q
def update_values(d, p, q, lr=0.01):
    p = p.astype('float64')
    q = q.astype('float64')
    #print("P:", p)
    #print("Q:", q)
    p = p - (lr * (-2 * q * (d - (np.dot(p,q)))))
    q = q - (lr * (-2 * p * (d - (np.dot(p,q)))))
    return p, q

def graph_loss(train_loss, test_loss, epochs, file_name="loss.png"):
    x = [i for i in range(epochs)]

    plt.plot(x, test_loss, label="Test Loss")
    #plt.plot(x, train_loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Testing Loss of matrix SGD (lr=0.01)")
    plt.legend()
    plt.savefig(file_name)

#This function is the meat of the program
def matrix_completion_SGD(data_matrix, train_x, test_x, lr=0.01, epochs=10, k=10):
    print("Starting SVD")
    #Data_matrix is the original matrix
    #train_x is the list of points in data_matrix that are in the training set
    #test_x is the list of points in the data_matrix that are in the testing set
    #lr is learning rate
    #epochs is the number of epochs, duh
    #k is the inner dimensionality of P and Q

    #First setp is generating random p and q
    n = data_matrix.shape[0]
    m = data_matrix.shape[1]

    #Create random P and Q and then add 1 so value is reasonably close to root 1-5
    P = np.random.rand(n,k)
    Q = np.random.rand(k,m)

    #P = P.astype('float64')
    #Q = Q.astype('float64')

    #Creating some lists of training error and test error for plotting later
    train_loss = []
    test_loss = []

    #The matrix produced from the multiplication of P and Q
    reconstructed_matrix = P @ Q

    #reconstructed_matrix = reconstructed_matrix.astype('float64')

    #print("P:", P)

    #Looping over training set of points
    #With each one we do a change to a row of p and column of q
    #get the error
    #Update values suing weight function
    #reconstruct and get total error

    #This is the main SGD loop
    print("Starting Epochs")
    for i in range(epochs):
        #Loop over every point in train_x
        #print("Starting Epoch", i)
        for point in train_x:
            #Get error from actual value in data_matrix
            #Get updated value of p and q based on this error
            #update values of P and Q

            #print(point)

            p = point[0]
            q = point[1]

            p_row_updated, q_col_updated = update_values(data_matrix[p,q], P[p,:], Q[:,q], lr)
            P[p,:] = p_row_updated
            Q[:,q] = q_col_updated


        #reconstruct matrix and get total loss on test and train
        reconstructed_matrix = P @ Q
        #reconstructed_matrix = reconstructed_matrix.astype('float64')

        total_loss = 0
        for point in train_x:
            #Calculate aggregate loss
            p = point[0]
            q = point[1]
            #total_loss += loss(reconstructed_matrix[p,q], P[p,:], Q[:,q]) #point d, row of P and column of Q
            total_loss += pow(data_matrix[p,q] - reconstructed_matrix[p,q], 2)
            #print(data_matrix[p,q], reconstructed_matrix[p,q])
            #print(data_matrix[p,q], reconstructed_matrix[p,q])

        total_loss = math.sqrt(total_loss / len(train_x))
        print("Epoch:", i, "training loss:", total_loss, end="")
        train_loss.append(total_loss)

        total_loss = 0
        for point in test_x:
            #Calculate aggregate loss
            p = point[0]
            q = point[1]
            #print(total_loss)
            #total_loss += loss(reconstructed_matrix[p,q], P[p,:], Q[:,q])
            total_loss += pow(data_matrix[p,q] - reconstructed_matrix[p,q], 2)

        total_loss = math.sqrt(total_loss / len(test_x)) #- 39

        print(" test loss:", total_loss)
        test_loss.append(total_loss)

    #Returns the matrix and the two oss lists

    return reconstructed_matrix, train_loss, test_loss

#P[5,:] row
#Q[:,6] column

def filter_low_count_users(coord_list, count=10):

    row_counts = dict()

    print("Size of data list:", len(coord_list))
    #Get the count of how many times each user rated a movie
    for (x,y) in coord_list:
        if x in row_counts:
            row_counts[x] += 1
        else:
            row_counts[x] = 1

    for i,(x,y) in enumerate(coord_list):
        if row_counts[x] < count:
            coord_list.pop(i)

    print("Counts:", row_counts)

    print("Size of data list filtered:", len(coord_list))
    return coord_list

if __name__=="__main__":
    #Import the dataset
    df=pd.read_csv('matrix_dataset.csv', sep=',', low_memory=False)
    df = df.drop(columns=['index']) #get rid of useless numbering from matrix i.e. 0th column
    print(df.head())

    df.info(verbose=True)

    df = df.dropna()

    df.info(verbose=True)

    #exit(0)

    #Now we need to create the data matrix and split into training and testing datasets
    #row = np.array(df.loc[:, 'index'])
    #col = np.array(df.loc[:, 'movieId'])
    #data = np.array(df.loc[:, 'rating'])

    #row = row - 1
    #col = col - 1

    #data_matrix = coo_matrix((data, (row, col))).tocsr()

    #df['actor'] = df['actor'] / 10 , compensated for with updated csv file

    data_matrix = df.to_numpy(dtype='float64')
    data_matrix = data_matrix / 10

    print(data_matrix)


    #Do we need to transpose this matrix?
    #What are we looping over?
    #How do we break this apart into training and testing sets
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    #Use this for splitting test and train
    #train_matrix, text_matrix, train_y, test_y = train_test_split(data_matrix, range(data_matrix.shape[0]), test_size=0.25)
    #print(train_matrix)

    #Fix train test split
    coord_list = coordinate_list(data_matrix)
    #coord_list = filter_low_count_users(coord_list, count=10)
    #print(out[:5])

    #Now that we have a list of coordinates, we can split into test and train
    train_x, test_x = train_test_split(coord_list, test_size=0.25)
    print(len(train_x))

    #exit(0)

    #out_m, train_loss, test_loss = matrix_completion_SGD(data_matrix, train_x, test_x, epochs=30, lr=0.001)

    #graph_loss(train_loss, test_loss, 30)

    #Now we need to test dimensionality of k
    #ks = [10, 20, 30, 40, 50, 60 ,70, 80, 90, 100]


    ##
    ##Bigger K the faster the overfit
    ##Need bigger train
    kd = 15
    #for kd in ks:
    out_m, train_loss, test_loss = matrix_completion_SGD(data_matrix, train_x, test_x, epochs=300, lr=0.01, k=kd) #0.001
    graph_loss(train_loss[10:], test_loss[10:], 290, file_name="loss_"+str(kd)+".png")
    #sanity check for values
    r = random.randint(0, len(test_x))
    rand_point = test_x[r]
    rand_x = rand_point[0]
    rand_y = rand_point[1]

    print("Original Matrix: ", data_matrix[rand_x, rand_y])
    print("Reconstructed Matrix:", out_m[rand_x, rand_y])

    #my_movies = out_m[610, :]
    #my_movies = np.argsort(my_movies)[-10:]
    #print(my_movies)

    #Next steps we need to remove bad people from training (ie people wo have not seen at least 10 movies)
    #remove bad movies as well
    #Convert this graphing into a function and perform the training wit hdifferent values of k to see which works best
    #Upscale to large dataset



    #Now that we have our training and testing sets we can start SGD

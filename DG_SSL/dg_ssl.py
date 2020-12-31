"""
Author: Jianpeng Liu
Copyrights belong to WiSeR Lab 
Distance-aware graph-based semisupervised learning
Reference: https://ieeexplore.ieee.org/abstract/document/8647621
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA
import statistics
"""Followings are functions used to calalated 
differet component of the equation.
"""
def similarity_matrix(fp,row,t):
    w = [] 
    for i in range(row):
        w.append([])
        for j in range(row):
            temp = math.exp(-LA.norm(fp[i] - fp[j], 2)/t)
            w[i].append(temp)
    return w

def distance_matrix(fp,row,col,a,b):
    d = []
    a = 3.5
    b = -30
    for i in range(row):
        d.append([])
        for j in range(row):
            temp = []
            for m in range(col):
                v1 = (fp[i][m] - fp[j][m])/(10*a)
                v2 = abs(pow(10, v1) - 1)
                v3 = v2 * pow(10,(fp[i][m]-b)/(-10*a))
                temp.append(v3)
            #d[i].append(max(temp))
            d[i].append(statistics.median(temp))
    return d

def laplace_matrix(row,w):
    u = []
    for i in range(row):
        u.append(sum(w[i]))
    D = np.diag(u)
    L = D - w
    return L

def k_matrix(row,w,d):
    k = []
    for i in range(row):
        k.append([])
        for j in range(row):
            k[i].append(w[i][j]*d[i][j])
    k = np.array(k)
    return k

def update_g(result,row,G,k):
    for i in range(row):
        g = np.array([0]*2,dtype=float).reshape(2)
        for j in range(row):
            if (i != j): 
                v1 = result[:,i] - result[:,j]
                v2 = LA.norm(v1,1)
                g = g + (k[i][j] * (v1/v2))
        G[:,i] = g
    return G

def dg_ssl_init(labeled, unlabeled):
    labeled_np = labeled
    unlabeled_np = unlabeled
    dimension_ul = unlabeled_np.shape
    dimension_l = labeled_np.shape
    label_and_unlabel = np.append(labeled_np,unlabeled_np,axis=0)
    dimension_t = label_and_unlabel.shape

    loc = label_and_unlabel[:,dimension_t[1]-2:] # location 
    fp = label_and_unlabel[:,:dimension_t[1]-2] # rssi 
    row = dimension_t[0] 
    col = dimension_t[1]-2
    t = dimension_ul[0] # number of unlabeled data

    w = similarity_matrix(fp,row,t)
    a = 3.5 # parameters for distance_matrix function
    b = -30 # parameters for distance_matrix function
    d = distance_matrix(fp, row, col, a, b)
    L = laplace_matrix(row,w)
    k = k_matrix(row,w,d)

    J = [1]*(row-t) + [0]*t
    J = np.diag(J)

    G = np.array([0]*row*2,dtype=float).reshape(2,row)
    loc_transpose = np.transpose(loc)

    # iterate until converge
    r = 1
    np.seterr(all='raise')

    for i in range(200):
        try:
            first_part_eq = np.dot(loc_transpose,J) + (r*G)
            second_part_eq = np.linalg.inv(J + r*L)
            predicted_result = np.dot(first_part_eq,second_part_eq)
            G = update_g(predicted_result,row,G,k)
        except:
            break

    # Saved the propergation result to a csv file 
    predicted_result = np.transpose(predicted_result)
    final_result = np.concatenate((fp, predicted_result), axis=1)
    final_result2 =  np.concatenate((labeled, final_result[:10,:]), axis=0)
    return [final_result, final_result2]



def dg_ssl_inita(labeled, unlabeled):
    labeled_np = labeled
    unlabeled_np = unlabeled
    dimension_ul = unlabeled_np.shape
    dimension_l = labeled_np.shape
    label_and_unlabel = np.append(labeled_np,unlabeled_np,axis=0)
    dimension_t = label_and_unlabel.shape

    loc = label_and_unlabel[:,dimension_t[1]-2:] # location 
    fp = label_and_unlabel[:,:dimension_t[1]-2] # rssi 
    row = dimension_t[0] 
    col = dimension_t[1]-2
    t = dimension_ul[0] # number of unlabeled data


    w = similarity_matrix(fp,row,t)
    a = 3.5 # parameters for distance_matrix function
    b = -30 # parameters for distance_matrix function
    d = distance_matrix(fp, row, col, a, b)
    L = laplace_matrix(row,w)
    k = k_matrix(row,w,d)

    J = [1]*t + [0]*(row-t)
    J = np.diag(J)

    G = np.array([0]*row*2,dtype=float).reshape(2,row)
    loc_transpose = np.transpose(loc)

    # iterate until converge
    r = 1
    np.seterr(all='raise')
    for i in range(300):
        try:
            first_part_eq = np.dot(loc_transpose,J) + (r*G)
            second_part_eq = np.linalg.inv(J + r*L)
            predicted_result = np.dot(first_part_eq,second_part_eq)
            G = update_g(predicted_result,row,G,k)
        except:
            break
            
    # Saved the propergation result to a csv file 
    predicted_result = np.transpose(predicted_result)
    final_result = np.concatenate((fp, predicted_result), axis=1)
    final_result2 =  np.concatenate((labeled, final_result[:10,:]), axis=0)
    return [final_result, final_result2]



def dg_ssl_int(labeled, unlabeled):
    labeled_np = labeled
    unlabeled_np = unlabeled
    dimension_ul = unlabeled_np.shape
    dimension_l = labeled_np.shape
    label_and_unlabel = np.append(labeled_np,unlabeled_np,axis=0)
    dimension_t = label_and_unlabel.shape

    loc = label_and_unlabel[:,dimension_t[1]-2:] # location 
    fp = label_and_unlabel[:,:dimension_t[1]-2] # rssi 
    row = dimension_t[0] 
    col = dimension_t[1]-2
    t = dimension_ul[0] # number of unlabeled data

    w = similarity_matrix(fp,row,t)
    a = 3.5 # parameters for distance_matrix function
    b = -30 # parameters for distance_matrix function
    d = distance_matrix(fp, row, col, a, b)
    L = laplace_matrix(row,w)
    k = k_matrix(row,w,d)

    J = [1]*(t) + [0]*(row-t)
    J = np.diag(J)

    G = np.array([0]*row*2,dtype=float).reshape(2,row)
    loc_transpose = np.transpose(loc)

    # iterate until converge
    r = 1
    for i in range(10):
        first_part_eq = np.dot(loc_transpose,J) + (r*G)
        second_part_eq = np.linalg.inv(J + r*L)
        predicted_result = np.dot(first_part_eq,second_part_eq)
        G = update_g(predicted_result,row,G,k)

    # Saved the propergation result to a csv file 
    predicted_result = np.transpose(predicted_result)
    final_result = np.concatenate((fp, predicted_result), axis=1)
    final_result2 =  np.concatenate((labeled, final_result[:10,:]), axis=0)
    return [final_result, final_result2]

def prop_error(real,estimate):
    errors = real - estimate 
    error = np.median(np.sqrt(errors[:,0]**2 + errors[:,1]**2))
    return error


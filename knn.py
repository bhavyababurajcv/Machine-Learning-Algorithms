import math

import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="File Name")
args = parser.parse_args()
file = args.data


def calculate_eud(x,k):
    dist = (((x[0]-k[0])**2) + ((x[1]-k[1])**2))
    return dist
def calculate_j(k,clust_points):
    j=0
    for i in clust_points:
        d=calculate_eud(k,i)
        j+=d
    return j
def calculate_mean(c):
    a=( [sum(y) / len(y) for y in zip(*c)])
    return a

def k_means(k,df):
    c1=[]
    c2=[]
    c3=[]
    j=[0]*3
    for i in range(0,df.shape[0]):
        x=tuple(df.loc[i,1:2])
        l=[calculate_eud(x,j) for j in k]
        cluster=l.index(min(l))
        if cluster==0:
            c1.append(x)
        elif cluster==1:
            c2.append(x)
        else:
            c3.append(x)
    j[0]=calculate_j(k[0],c1)
    j[1]=calculate_j(k[1],c2)
    j[2]=calculate_j(k[2],c3)
    cur_j=sum(j)
    print(cur_j)
    k1 = calculate_mean(c1)
    k2 = calculate_mean(c2)
    k3 = calculate_mean(c3)
    cal_k=(k1,k2,k3)
   # print(cal_k)
    return cal_k

df= pd.read_csv(file,header=None)
k=[(0,5),(0,4),(0,3)]
ks=[]
while True:
    next_k=k_means(k,df)
    if k==next_k:
        ks.append(k)
        break
    else:
        ks.append(k)
        k=next_k
        continue

for i in ks:
    print(f"{(i[0][0])},{(i[0][1])}\t { ((i[1][0]))},{((i[1][1]))}\t{ ((i[2][0]))} ,{((i[2][1]))}")

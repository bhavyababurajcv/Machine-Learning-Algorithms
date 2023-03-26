import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="File Name")
args = parser.parse_args()
file = args.data


df=pd.read_csv(file,header=None)
total = df.shape[0]
classes=df[0].explode().unique()
fin={}

def calculate_gauss(x,mean,sd):
    mean=float(mean)
    sd=float(sd)
    ans=(1 / (np.sqrt(2 * np.pi * sd))) * (np.exp(-((x - mean) ** 2 / (2 * sd))))
    return ans


for i in classes:
    cur_count=sum(df[0] == i)
    p=cur_count/total
    x1_sum = (df.loc[df[0] == i, 1].sum())
    x2_sum = (df.loc[df[0] == i, 2].sum())
    mean_x1_c = (1/cur_count) * x1_sum
    mean_x2_c = (1/cur_count) * x2_sum
    x1 =(((df.loc[df[0] == i, 1]).subtract(mean_x1_c))**2).sum()
    x2 = (((df.loc[df[0] == i, 2]).subtract(mean_x2_c))**2).sum()
    sd_1_c = (1/(cur_count-1)) * (x1)
    sd_2_c = (1/(cur_count-1)) * (x2)
    str_fin=str(mean_x1_c)+","+str(sd_1_c)+","+str(mean_x2_c)+","+str(sd_2_c)+","+str(p)
    fin.update({i:str_fin})
    print(str_fin)

error_class=0
for i in range(0,total):
    x1=df.iloc[i][1]
    x2=df.iloc[i][2]
    out_class = df.iloc[i][0]
    data_a=fin.get("A").split(",")
    data_b=fin.get("B").split(",")
    p_a = float(data_a[-1]) * calculate_gauss(x1,data_a[0],data_a[1])*calculate_gauss(x2,data_a[2],data_a[3])
    p_b = float(data_b[-1]) * calculate_gauss(x1,data_b[0],data_b[1])*calculate_gauss(x2,data_b[2],data_b[3])

    if(p_a>p_b):
        pred_class = "A"
    elif(p_b>p_a):
        pred_class="B"
    if pred_class!=out_class:
        error_class+=1


print(error_class)

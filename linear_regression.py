import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="File Name")
parser.add_argument("-e", "--eta", help="Learning Rate")
parser.add_argument("-t", "--threshold", help="Threshold")
args = parser.parse_args()
#reading the arguments passed
file_path=args.data
learning_rate = float(args.eta)
threshold=float(args.threshold)
#file_path="random3.csv"
#learning_rate = 0.00005
#threshold = 0.0001
csv_file = open(file_path, "r")
list_row=[line[:-1] for line in csv_file]
step=0
y_exp=[]
for i in range(0,len(list_row)):
    list_row[i] = list(list_row[i].split(","))
    list_row[i] = [1]+[float(elem) for elem in list_row[i]]
    y_exp.append(list_row[i][-1])
weight = [0]*(len(list_row[0])-1)
SSE_list=[]
SSE=0
while(step>=0):
    if(step>1):
        if(SSE_list[-2]-SSE_list[-1]<threshold):
            break
    y_pred=[]
    for i in list_row:
        x=i[:-1]
        y_p=0
        # calculating y'
        for k in range(0,len(x)):
            y_p+=x[k]*weight[k]
        y_pred.append(y_p)
    error_diff=[y_exp[i]-y_pred[i] for i in range(0,len(y_exp))]
    sse_ind = [i**2 for i in error_diff]
    SSE = round(sum(sse_ind),9)

    #output string
    print(str(step),end=',')
    for i in weight:
        print(str(format(float(i),'.9f')),end=',')
    print(str(format(SSE,'.9f')))
    #-------------------

    step+=1
    gradient=[0]*len(x)
    for i in range(0,len(list_row)):
        k=0
        while(k<len(gradient)):
            gradient[k]+=list_row[i][k] * error_diff[i]
            if(i==len(list_row)-1):
                weight[k]=weight[k]+(learning_rate*gradient[k])
            k+=1
    SSE_list.append(SSE)




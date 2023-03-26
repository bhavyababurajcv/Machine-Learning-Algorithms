import math
import argparse
from decimal import Decimal
def backpropagation(training_example):
    input_vector=training_example[0:-1]
    input_vector.insert(0,1)
    target_output=training_example[-1]
    net_hid=[0]*3
    hid_outs=[0]*3
    out=0
    net_h1=0
    net_h2=0
    net_h3=0
    for i in range(0,len(input_vector)):
        net_h1+=input_vector[i]*weights_h1[i]
        net_h2+=input_vector[i]*weights_h2[i]
        net_h3+=input_vector[i]*weights_h3[i]

    for i in range(1,4):
        net_h_unit = "net_h"+str(i)
        hid_outs[i-1] = 1/(1+math.exp((-1)*locals()[net_h_unit]))
    net_out = 1* weights_h_out[0]
    for i in range(0,3):
        net_out += hid_outs[i]*weights_h_out[i+1]
    out = 1/(1+math.exp((-1)* net_out))
    delta_o = out * (1-out) * (target_output - out)
    delta_h=[0]*3
    for i in range(0,3):
        delta_h[i] = hid_outs[i] * (1-hid_outs[i]) * (weights_h_out[i+1]*delta_o)

    for i in range(0,len(weights_h_out)):
        if i==0:
            delta_w = eta * delta_o * 1
        else:
            delta_w = eta * delta_o * hid_outs[i-1]
        weights_h_out[i] += delta_w

    for i in range(0,len(weights_h1)):
        weights_h1[i] += eta * delta_h[0] * input_vector[i]
        weights_h2[i] += eta * delta_h[1] * input_vector[i]
        weights_h3[i] += eta * delta_h[2] * input_vector[i]
    input_vector=input_vector[1:]
    inputs=','.join(str(round(Decimal(i),5)) for i in input_vector)
    hidd_out=','.join(str(round(Decimal(i),5))   for i in hid_outs)
    deltas=','.join(str(round(Decimal(i),5))   for i in delta_h)
    h1_weight = ','.join(str(round(Decimal(i),5)) for i in weights_h1)
    h2_weight = ','.join(str(round(Decimal(i),5))  for i in weights_h2)
    h3_weight = ','.join(str(round(Decimal(i),5))   for i in weights_h3)
    out_weight = ','.join(str(round(Decimal(i),5))   for i in weights_h_out)

    print(inputs,hidd_out,str(round(out,5)),str(round(target_output,5)),deltas,str(round(delta_o,5)),h1_weight,h2_weight,h3_weight,out_weight,sep=",")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="File Name")
parser.add_argument("-e", "--eta", help="Learning Rate")
parser.add_argument("-t", "--iterations", help="Iterations")
args = parser.parse_args()
data = args.data
eta = float(args.eta)
iteration=float(args.iterations)
#weight initilaization
weights_h1=[0.2,-0.3,0.4] #(bias,a_h1,b_h1)
weights_h2=[-0.5,-0.1,-0.4]
weights_h3=[0.3,0.2,0.1]
weights_h_out=[-0.1,0.1,0.3,-0.4]
h_w = ",".join(str(i) for i in weights_h1+weights_h2+weights_h3+weights_h_out)
print(",".join(["-"]*11),h_w,sep=",")

it = 0
f_h = open(data,"r")
while(it<iteration):
    
    for ele in f_h:
        a=ele.split(",")
        training_ex=[float(i) for i in a]
        backpropagation(training_ex)
    f_h.seek(0)
    it+=1

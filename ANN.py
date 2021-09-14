 #=================================   # Soft computing (ME 674)   =====================================#
 #=================================   # Coding Assignment         =====================================#                          
 #=================================   # Name     - Lokender Singh =====================================#
 #=================================   # Roll No. - 204103311      =====================================#                           

import numpy as np
import pandas as pd
import math

data = pd.read_csv('neural.csv')
data = data.to_numpy()

p = 800  #No. of training Patterns
l = 8    #No. of input neurons
m = 9   #No. of hidden neurons  
n = 1    #No. of output neurons
lr = 0.7   #learning rate
mt = 0.5 #momentum term
mse = 1  #Mean square error
desired_error = 0.0001210#error required
delW_mom = 0
delV_mom = 0
count = 0
convergence_tol = 1e-7
convergence = 1

q =  data[0:p,:] #Training pattern


#Normalization
max =np.zeros((p))
min =np.zeros((p))
for i in range(p):
    max[i] = np.max(q[i, :])
    min[i] = np.min(q[i, :])

s = np.zeros(([p,l+1]))
for i in range(p):
    for j in range(l+1):
        norm = 0
        norm = norm  + ((q[i][j]-min[i])/(max[i]-min[i]))
        s[i][j] = (norm*0.8)+ 0.1

training_features = s[:,0:-1]
training_labels  = s[:,-1]

x = np.transpose(training_features)
print(x.shape)
w = np.transpose(training_labels)
TO = w.reshape(n,p)

#Adding Bias
bias = np.ones([p])
I= np.row_stack((bias,x))

V = np.zeros((l+1,m))  #Initialize V weight matrix 
W = np.zeros((m+1,n))  #Initialize W weight matrix

IH = np.zeros((m+1,p)) # input hidden matrix 
OH = np.zeros((m+1,p)) # Output hidden matrix 

IO = np.zeros((n+1,p)) # input hidden matrix 
OO = np.zeros((n+1,p)) # Output hidden matrix

delW = np.zeros((m+1,n)) # updated w matrix
delV = np.zeros((l+1,m)) # updated v matrix

#random matrix weight V
np.random.seed(0)
for i in range(l+1):
    for j in range(m):
        V[i][j] = np.random.rand()

#random matrix weight W
np.random.seed(3)
for i in range(m+1):
    for j in range(n):
         W[i][j] = np.random.rand()

while mse>desired_error :

    #for input and output of hidden neuron
    for i in range(p):
        for j in range(m):
            val = 0
            for k in range(l+1):
                val = val + I[k][i]*V[k][j]
                IH[j][i] = val
                OH[j][i] = (1/(1+np.exp(- val)))
    
    #for input and output of output neuron
    for i in range(p):
        for j in range(n):
            val = 0
            for k in range(m+1):
                val = val + OH[k][i]*W[k][j]
                IO[j][i] = val
                OO[j][i] = (1/(1+np.exp(- val)))
                
    #Error calculation
    error = 0
    for i in range(p):
        for j in range(n):
            er = TO[j][i] - OO[j][i]
            error = error + 0.5*er*er
    
    mse = error/p
   
    #Update weight W
    for i in range(m+1):
        for j in range(n):
            val = 0
            for k in range(p):
                val = val + ((TO[j][k] - OO[j][k])*(1-OO[j][k])*OO[j][k]*OH[i][k]) #log sigmoid
            delW[i][j] = (lr*val)/p

    delW = delW + (mt*delW_mom)
    
    # V weight update
    for i in range(l+1):
        for j in range(m):
            val=0
            for q in range(p):
                val1 = 0
                for k in range(n):
                    val1 = val1 + ((TO[k][q] - OO[k][q]) *(1 - OO[k][q])*OO[k][q] * W[j][k] * OH[j][q] * (1 - OH[j][q]) * I[i][q]) #log sigmoid
                val = val + val1
                delV[i][j] = ((lr)*val)/(p*n)

    delV = delV + (mt*delV_mom)
    
    W = W + delW 
    V = V + delV
    count = count + 1
    # if(abs(convergence - mse)<convergence_tol):
    #     break
    # convergence = mse
    print(count)
    print(mse)

    if (count >= 5000):
        break

print("the final weight W matrix is:")
print(W)
print("the final weight V matrix is:")
print(V)
print("The mean square error is:")
print(mse)
print("Targeted output:")
print(TO)
print("obtained output:")
print(OO)
print("Count :")
print(count)


# ==================================================Testing==============================================================
q1 =  data[p:,:]
print("hahahaha")
p=238
print(q1.shape)
max =np.zeros((p))
min =np.zeros((p))
for i in range(p):
    max[i] = np.max(q1[i, :])
    min[i] = np.min(q1[i, :])

#tested pattern

s1 = np.zeros(([p,l+1]))
for i in range(p):
    for j in range(l+1):
        norm = 0
        norm = norm  + ((q1[i][j]-min[i])/(max[i]-min[i]))
        s1[i][j] = (norm*0.8)+ 0.1

testing_features = s1[:,0:-1]
testing_labels  = s1[:,-1]

x = np.transpose(testing_features)
print(x.shape)
w = np.transpose(testing_labels)
print(w)
TO = w.reshape(n,p)

bias = np.ones([p])
I= np.row_stack((bias,x))
OO = np.zeros((n+1,p))
for i in range(p):
    for j in range(m):
        val = 0
        for k in range(l+1):
            val = val + I[k][i]*V[k][j]
            IH[j][i] = val
            OH[j][i] = (1/(1+np.exp(- val)))
    
#for input and output of output neuron
for i in range(p):
    for j in range(n):
        val = 0
        for k in range(m+1):
            val = val + OH[k][i]*W[k][j]
            IO[j][i] = val
            OO[j][i] = (1/(1+np.exp(- val)))

print("Tested output")
print(w)
print("obtain output")
print(OO)

c = np.zeros(([p,n+1]))
for i in range(p):
    for j in range(n):
        norm = 0
        norm =  (((TO[j][i]-0.1)*(max[i]-min[i]))/0.8) + min[i]          
        c[i][j] = norm
print(c)

d = np.zeros(([p,n+1]))
for i in range(p):
    for j in range(n):
        norm = 0
        norm =  (((OO[j][i]-0.1)*(max[i]-min[i]))/0.8) + min[i]          
        d[i][j] = norm
print(d)

MSE=0
temp =0
for i in range(p):
    error = 0
    for j in range(n):
        error = math.fabs(c[i][j] - d[i][j])
    temp = temp + error

predicted_error= temp/p
print(predicted_error)

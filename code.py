import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from numpy import nan
#from scipy.interpolate import interp1d
#from scipy import interpolate
%matplotlib inline



#import data to a pandas dataframe
data = pd.read_csv(r'C:\Users\nadia\OneDrive\Desktop/thesis/solar_data.csv')
#data = pd.read_csv(r'/home/angelos/Desktop/nadiasthesis/solar_data.csv')
data = pd.read_csv(r'solar_data.csv')


#add colums to our dataframe
data['kt'] = data['GHIobs']/ data['TOA']
data['h']= np.radians(90 - data['sza'])
data['m']= (np.sin(data['h']) + 0.15 * (np.degrees(data['h']) + 3.885)**(-1.253))**(-1)
data['kt_1']= data['kt']/(1.031 * np.exp(-1.4/(0.9+ 9.4/data['m'])) + 0.1)


#divide dataframe according to weather conditions (ανοιχτό διάστημα προς τα αριστερά)
clear = data[data['kt_1'].between(0.65,10,inclusive='right')]
intermediate = data[data['kt_1'].between(0.30,0.65,inclusive='right')]
cloudy = data[data['kt_1'].between(0.00,0.30,inclusive='right')]


#print(clear['kt_1'].head(50))
#print(intermediate.head(15))
data



# plot function 
def graph(raw,obs,z,cond):
    plt.figure(figsize=(10,10))
    plt.plot(raw,obs,"ro",color=z,markersize=1)
    model = LinearRegression().fit(raw.values.reshape(-1,1), obs.values.reshape((-1, 1)))
    print('for', cond ,'weather conditions')
    print('*slope (a) is:', model.coef_)
    print("*intercept(b) is",model.intercept_)
    return (model.coef_,model.intercept_)
 


# MBE function
def MBE(name,raw,obs,c):
    data[name]= raw-obs
    #data[name]=data[name].isna()
    sum_ = data[name].sum()
    n = len(data[name].index)
    MBE_= sum_ / n
    print("the MBE for the",c, "conditions is:",MBE_)
    return MBE_

  
  
# RMSE function
def RMSE(name,raw,obs,c):
    data[name] = (raw-obs)**2
    #data[name]= data[name].isna()
    sum_=data[name].sum()
    n = len(data[name].index)
    RMSE_=(sum_/n)**0.5
    print("the RMSE for the",c, "conditions is:", RMSE_)
    return RMSE_ 
  
  
  
#clear weather conditions 
x1='clr'
x11='clr2'
y1=clear['GHIraw']
z1=clear['GHIobs']

c1='clear'
r1='pink'

p1,p11 = graph(z1,y1,r1,c1)
MBE(x1,y1,z1,c1)
RMSE(x11,y1,z1,c1)


#intermediate weather conditions 
x2='int'
x22='int2'
y2=intermediate['GHIraw']
z2=intermediate['GHIobs']

c2='intermediate'
r2='purple'

p2,p22 = graph(z2,y2,r2,c2)
MBE(x2,y2,z2,c2)
RMSE(x22,y2,z2,c2)


#cloudy weather conditions 
x3='cld'
x33='cld2'
y3=cloudy['GHIraw']
z3=cloudy['GHIobs']

c3='cloudy'
r3='grey'

p3,p33 = graph(z3,y3,r3,c3)
MBE(x3,y3,z3,c3)
RMSE(x33,y3,z3,c3)



data.head(100)


#is the kt_1 index okay for the cloudy conditions?
print(cloudy[['timestamp','GHIobs','GHIraw','kt','kt_1']].head(20))



#διόρθωση τιμών
#by now we have created 3 dataframes according to the weather conditions
clear


#μέθοδος 1η: γραμμική μέθοδος
# GHImod = aGHIobs + β : already calculated above (GHImod=GHIraw)
# GHImod,cor = GHImod - [(a-1)GHIobs +b]

GHIcclr = clear['GHIraw'] - ((p1[0][0] - 1)*clear['GHIobs'] +  p11)
GHIcint = intermediate['GHIraw'] - ((p2[0][0] - 1)*intermediate['GHIobs'] +  p22)
GHIccld = cloudy['GHIraw'] - ((p3[0][0] - 1)*cloudy['GHIobs'] +  p33)


# GHImod,cor = a'GHImod + b' : προσδιορισμός των α', β'


#corrected clear conditions
cx1 = 'cclr'
cx11 = 'cclr2'
cy1 = GHIcclr
cz1 = clear['GHIobs']

#c1, r1 same as before

graph(cz1,cy1,r1,c1)
MBE(cx1, cy1, cz1, c1)
RMSE(cx11,cy1,cz1,c1)


#corrected intermediate conditions
cx2 = 'cint'
cx22 = 'cint2'
cy2 = GHIcint
cz2 = intermediate['GHIobs']

graph(cz2,cy2,r2,c2)
MBE(cx2, cy2, cz2, c2)
RMSE(cx22,cy2,cz2,c2)


#corrected clody conditions
cx3 = 'ccld'
cx33 = 'ccld2'
cy3 = GHIccld
cz3 = cloudy['GHIobs']


graph(cz3,cy3,r3,c3)
MBE(cx3, cy3, cz3, c3)
RMSE(cx33,cy3,cz3,c3)


data.head(100)



#μέθοδος 2η: EQM (empirical quantile mapping)
num = list(np.arange(0,100,0.5))

#calculating the percentiles for the module and the observed values for all weather conditions

clobper=np.percentile(clear['GHIobs'],num)
clrawper=np.percentile(clear['GHIraw'],num)

inobper=np.percentile(intermediate['GHIobs'],num)
inrawper=np.percentile(intermediate['GHIraw'],num)

cdobper=np.percentile(cloudy['GHIobs'],num)
cdrawper=np.percentile(cloudy['GHIraw'],num)



#visualization of data

#plot2 function for the percentiles of mod and obs values
def graph2(obper,rawper):
    plt.figure(figsize=(10,10))
    plt.plot(obper, num, color='red', label = "obs")
    plt.plot(rawper, num,color='pink',label = "raw")
    #plt.legend()
    plt.show()
    
    
    
print("for the clear weather conditions")
graph2(clobper,clrawper)

print("for the intermediate weather conditions")
graph2(inobper,inrawper)

print("for the cloudy weather conditions")
graph2(cdobper,cdrawper)



#plot3 function for comparing mod and obs values
def graph3(perobs,permod,col):
    plt.figure(figsize=(10,10))
    plt.plot(perobs, permod, color=col)
    plt.legend()
    plt.show()
    
    
    
graph3(clobper,clrawper,'pink')

graph3(inobper,inrawper,'purple')

graph3(cdobper,cdrawper,'gray')

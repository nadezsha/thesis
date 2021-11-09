import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
%matplotlib inline


#import data to a pandas dataframe
data = pd.read_csv(r'C:\Users\nadia\OneDrive\Desktop/thesis/solar_data.csv')


#add colums to our dataframe
data['kt'] = data['GHIobs']/ data['TOA']
data['h']= np.radians(90 - data['sza'])
data['m']= (np.sin(data['h']) + 0.15 * (np.degrees(data['h']) + 3.885)**(-1.253))**(-1)
data['kt_1']= data['kt']/(1.031 * np.exp(-1.4/(0.9+ 9.4/data['m'])) + 0.1)


#divide dataframe according to weather conditions
clear = data[data['kt_1'].between(0.65,100)]
intermediate = data[data['kt_1'].between(0.35,0.65)]
cloudy = data[data['kt_1'].between(0.00,0.35)]

print(data.head(50))
#print(clear.head(15))


#plotting clear weather conditions and calculating slope and intercept
x1=clear['GHIobs']
y1=clear['GHIraw']
plt.figure(figsize=(10,10))
plt.plot(x1,y1,'ro',color='pink',markersize=1) 

model1 = LinearRegression().fit(x1.values.reshape(-1,1), y1.values.reshape((-1, 1)))

print('for clear weather conditions')
print('*slope (a) is:', model1.coef_)
print("*intercept(b) is",model1.intercept_)


#calculating MBE and RMSE
data['clr']= clear['GHIraw'] - clear['GHIobs']
sum_clr = data['clr'].sum()
n = len(data['clr'].index)
MBEclr= sum_clr / n
print("the MBE for the clear conditions is:",MBEclr)
data['clr2'] = (clear['GHIraw'] - clear['GHIobs'])**2
sum_clr2 = data['clr2'].sum()
RMSEclr = (sum_clr2/n)**0.5
print("the RMSE for the clear conditions is:",RMSEclr)


#plotting intermediate weather conditions and calculating slope and intercept
x2=intermediate['GHIobs']
y2=intermediate['GHIraw']
plt.figure(figsize=(10,10))
plt.plot(x2,y2,'ro',color='purple',markersize=1) 

model2 = LinearRegression().fit(x2.values.reshape(-1,1), y2.values.reshape((-1, 1)))

print('for intermediate weather conditions')
print('*slope (a) is:', model2.coef_)
print("*intercept(b) is",model2.intercept_)


#calculating MBE and RMSE
data['int']= intermediate['GHIobs'] - intermediate['GHIraw']
sum_int = data['int'].sum()
n = len(data['int'].index)
MBEi= sum_int / n
print("the MBE for the intermediate conditions is:",MBEi)
data['int2'] = (intermediate['GHIraw'] - intermediate['GHIobs'])**2
sum_int2 = data['int2'].sum()
RMSEi = (sum_int2/n)**0.5
print("the RMSE for the intermediate conditions is:",RMSEi)


#plotting cloudy weather conditions and calculating slope and intercept
x3=cloudy['GHIobs']
y3=cloudy['GHIraw']

plt.figure(figsize=(10,10))
plt.plot(x3,y3,'ro',color='grey',markersize=1) 

model3 = LinearRegression().fit(x3.values.reshape(-1,1), y3.values.reshape((-1, 1)))

print('for cloudy weather conditions')
print('*slope (a) is:', model3.coef_)
print("*intercept(b) is",model3.intercept_)


#calculating MBE and RMSE
data['cld']= cloudy['GHIobs'] - cloudy['GHIraw']
sum_cld = data['cld'].sum()
n = len(data['cld'].index)
MBEcld= sum_cld / n
print("the MBE for the cloudy conditions is:",MBEcld)
data['cld2'] = (cloudy['GHIraw'] - cloudy['GHIobs'])**2
sum_cld2 = data['cld2'].sum()
RMSEcld = (sum_cld2/n)**0.5
print("the RMSE for the cloudy conditions is:",RMSEcld)

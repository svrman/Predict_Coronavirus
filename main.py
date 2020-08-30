import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
import sklearn
import matplotlib.pyplot as plt

#### LOAD DATA ####
data = pd.read_csv('day_and_cases.csv', sep=';')
data = data[['day','cases']]
print('-'*30);print('HEAD');print('-'*30)
print(data.head())

#### PREPARE DATA ####
print('-'*30);print('PREPARE DATA');print('-'*30)
x = np.array(data['day']).reshape(-1, 1)
y = np.array(data['cases']).reshape(-1, 1)
plt.plot(y,'-m*')
#plt.show()
polyFeat = PolynomialFeatures(degree=3)
x = polyFeat.fit_transform(x)
print(x)
#### TRAINING DATA ####
print('-'*30);print('TRAINING DATA');print('-'*30)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f'Accuracy:{round(accuracy*100,3)} %')
y0 = model.predict(x)
plt.plot(y0,'--b')
plt.show()
#### PREDICTION ####
days = 2
print('-'*30);print('PREDICTION');print('-'*30)
print(f'Prediction - Cases after {days} days:',end='')
print(round(int(model.predict(polyFeat.fit_transform([[242+days]])))/1000000,2),'Million')

x1 = np.array(list(range(1,244+days))).reshape(-1,1)
y1 = model.predict(polyFeat.fit_transform(x1))
plt.plot(y1,'--r')
plt.plot(y0,'--b')
#plt.show()

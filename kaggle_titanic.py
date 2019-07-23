
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import pandas
import sys
#전처리 함수
def Basic_trans(setting):
    for i in data: 
        setting['Embarked'].fillna('S',inplace=True)
    setting['Sex'].replace(['male', 'female'], [0,1],inplace=True)
    setting['Embarked'].replace(['C', 'Q', 'S'], [0, 1, 2], inplace = True)
    
def null_ageset(setting):
    setting['Initial'] = 0
    for i in data:
        setting['Initial']=setting.Name.str.extract('([A-Za-z]+)\.')
    setting['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
    setting['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'], [0, 1, 2, 3, 4], inplace=True)
    setting.loc[(setting.Age.isnull())&(setting.Initial=='Mr'),'Age']=33
    setting.loc[(setting.Age.isnull())&(setting.Initial=='Mrs'),'Age']=36
    setting.loc[(setting.Age.isnull())&(setting.Initial=='Master'),'Age']=5
    setting.loc[(setting.Age.isnull())&(setting.Initial=='Miss'),'Age']=22
    setting.loc[(setting.Age.isnull())&(setting.Initial=='Other'),'Age']=46
    
def Special_trans(setting):
    setting['Age_band'] = 0
    setting.loc[setting['Age']<=16, 'Age_band'] = 0
    setting.loc[(setting['Age']>16) & (setting['Age']<=32), 'Age_band'] = 1
    setting.loc[(setting['Age']>32) & (setting['Age']<=48), 'Age_band'] = 2
    setting.loc[(setting['Age']>48) & (setting['Age']<=64), 'Age_band'] = 3
    setting.loc[setting['Age']> 64, 'Age_band'] = 4

    setting.loc[setting.Fare.isnull(), 'Fare'] = setting['Fare'].mean()
    setting['Fare'] = setting['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
    setting['Fare_cat'] = 0
    setting.loc[setting['Fare']<=7.91, 'Fare_cat'] = 0
    setting.loc[(setting['Fare']>7.91) & (setting['Fare']<=14.454), 'Fare_cat']= 1
    setting.loc[(setting['Fare']>14.454) & (setting['Fare']<=31), 'Fare_cat']= 2
    setting.loc[(setting['Fare']>31) & (setting['Fare']<=513), 'Fare_cat']= 3
    
    setting['Special'] = 0
    setting['Special'] = setting['Age_band'] * setting['Fare_cat']

#h(x)를 만드는 sigmoid 함수
def sigmoid(X, W):
    return 1 / (1+np.exp(-X.dot(W))) # 행렬연산
#J(W)를 만드는 cost_function함수
def cost_function(X, Y, W):
    m = len(Y)
    J = -(1/m) * ( Y * np.log(sigmoid(X, W)) + (1-Y) * np.log(1 - sigmoid(X, W)))
    return np.sum(J)

def gradient_descent(X, Y, W, learning_Rate, iterations):
    cost_history = [0] * iterations #cost값 기록
    grad_history = [0] * iterations #수렴되는지 판단을 위한 gradient 값 기록
    m = len(Y)
    
    for iteration in range(iterations):
        # compute the partial derivative w.r.t wi
        h = sigmoid(X,W) # W = weight, X = attribute
        loss = h - Y #실제값과 나온 결과 차이
        gradient = loss.dot(X) * (1/m) #행렬 연산
        # update wi
        W = W - learning_Rate * gradient
        
        cost = cost_function(X, Y, W)
        if( iteration % 10000 == 0): #10000번마다 cost값 출력
            print ("[", iteration , "] = ", cost)

        cost_history[iteration] = cost
        grad_history[iteration] = gradient
        
    return W, cost_history, grad_history


data = pandas.read_csv(sys.argv[1])
test = pandas.read_csv(sys.argv[2])
#데이터 전처리
Basic_trans(data)
null_ageset(data)
Special_trans(data)

    

train_d = data[['Sex', 'Special', 'Pclass', 'Embarked']] #사용한 attribute , special = fare * age
Sur = data['Survived']
    
#행렬 구성하기
m = len(train_d)
x0 = np.ones(m).reshape(-1,1)
X = np.concatenate((x0,train_d.values),axis=1)
y = Sur.values
    
#weight 초기 설정    
Weight = np.array([ 1.63148820e-01 , 9.36024581e-01, -3.43907308e-04 ,-5.52796430e-01, -9.18888469e-01]) 
new_weight, cost_history, grad_history = gradient_descent(X,y, Weight, 0.0001, 100000) #learning_rate = 0.001, iterations = 100000번
#gradient 수렴 확인용
#plt.plot(grad_history)
#plt.show()
#------------------------TestData---------------------------------
#test 전처리
Basic_trans(test)
Special_trans(test)
sur_V = [] # 결과 저장을 위한 sur_V
test_d = test[['Sex','Special', 'Pclass','Embarked']]
test['Survived'] = 0
#행렬 구성
length = len(test.Survived)
t0 = np.ones(length).reshape(-1,1)
X_test =np.concatenate((t0,test_d.values),axis=1)

print(new_weight)
for i in X_test:
    value = sigmoid(i,new_weight) #test 데이터에 대한 예측
    if value > 0.5:
        sur_V.append(1)
    else:
        sur_V.append(0)
   
test.Survived = sur_V
kaggle = pandas.DataFrame({"PassengerId": test["PassengerId"], "Survived": test["Survived"]})
kaggle.to_csv("1610254.csv", index = False)


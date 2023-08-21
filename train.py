import pandas as pd
import pickle
df = pd.read_csv('/content/500_Person_Gender_Height_Weight_Index.csv')
#https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex

data = {
0:'Extremely Weak',
1:'Weak',
2:'Normal',
3:'Overweight',
4:'Obesity',
5:'Extreme Obesity'
}

df.head()

df.isnull().sum()

x = df.drop(['Index'],axis = 1)

y = df['Index']

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
x['Gender'] = lb.fit_transform(x['Gender'])

import matplotlib.pyplot as plt
plt.hist(x['Height'])
plt.show()

plt.hist(x['Weight'])
plt.show()

plt.hist(x['Gender'])
plt.show()

from sklearn.preprocessing import StandardScaler
sd = StandardScaler()
x.iloc[:,1:] = sd.fit_transform(x.iloc[:,1:])

x.head()

import matplotlib.pyplot as plt
plt.hist(x['Height'])
plt.show()

plt.hist(x['Weight'])
plt.show()

lb.inverse_transform([0])





from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

x_train.shape

y_train.shape

from sklearn.linear_model import LogisticRegression
mn = LogisticRegression()
mn.fit(x_train,y_train)
mn.score(x_test,y_test)
data = {
    'sd':sd,
    'lb':lb,
    'mn':mn
}

f = open('bmimodel','wb')
pickle.dump(data,f)
f.close()
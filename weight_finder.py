import pandas as pd 
import matplotlib.pyplot as pt 
import sklearn.linear_model as lm 
mydata=pd.read_csv("data_height.csv") 
x=mydata[["height"]]
y=mydata[["weight"]]
pt.scatter(x,y) 
pt.show() 
model =lm.LinearRegression()
model.fit(x,y)
print("coefficient:", model.coef_[0])
print("intercept:", model.intercept_)
print(model.predict([[160]])) 

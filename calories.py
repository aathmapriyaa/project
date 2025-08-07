import pandas as pd 
import matplotlib.pyplot as pt 
import sklearn.linear_model as lm 
mydata=pd.read_csv("calories.csv") 
x=mydata[["steps_walked"]]
y=mydata[["calories_burned"]]
pt.scatter(x,y) 
pt.show() 
model =lm.LinearRegression()
model.fit(x,y)
print(model.predict([[4500]])) 

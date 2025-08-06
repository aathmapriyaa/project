import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
mydata = pd.read_csv("charge_percent.csv")
x=mydata[["time"]]
y=mydata[["percentage_charge"]]
pt.scatter(x, y)
pt.show()
model = lm.LinearRegression()
model.fit(x, y)
print(model.predict([[22]]))  
import pandas as pd
import matplotlib.pyplot as pt
import sklearn.linear_model as lm
mydata = pd.read_csv("house_price_prediction.csv")
x=mydata[["sqft"]]
y=mydata[["price"]]
pt.scatter(x, y)
pt.show()
model = lm.LinearRegression()
model.fit(x, y)
print(model.predict([[1700]]))  
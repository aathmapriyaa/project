import pandas as pd
import matplotlib.pyplot as pt
import sklearn.neighbors as knn
mydata=pd.read_csv("new_weight_predicter.csv")
x= mydata[["height"]]
y= mydata[["weight"]] 
model=knn.KNeighborsRegressor(n_neighbors=2)
model.fit(x,y)
print(model.predict([[160]]))

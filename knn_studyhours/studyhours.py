import pandas as pd 
import matplotlib.pyplot as pt 
import sklearn.neighbors as knn
mydata=pd.read_csv("studyhours.csv")
x=mydata[["studyhours"]]
y=mydata[["score"]] 
model=knn.KNeighborsRegressor(n_neighbors=3)

model.fit(x,y)
print(model.predict([[8]]))
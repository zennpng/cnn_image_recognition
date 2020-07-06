from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()

x = iris.data[:,(2,3)]    # petal width and petal length are the independent variables
y = (iris.target == 0).astype(np.int)    # converting the outputs to binary (True/1 -> Setosa, False/0 -> Virginica/Versicolor)

pct = Perceptron(random_state=42)
pct.fit(x,y)    # fit the model 

y_pred = pct.predict(x)    # generate predictions 

acc1 = accuracy_score(y, y_pred)
print(acc1)    # 1.0 accuracy means 100% accurate 
print(pct.intercept_)
print(pct.coef_)
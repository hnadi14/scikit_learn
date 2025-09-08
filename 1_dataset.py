from sklearn.datasets import fetch_california_housing
import pandas
from sklearn.datasets import make_classification

cal = fetch_california_housing(as_frame=True)
print(cal.frame)

x, y = make_classification(n_samples=100, n_features=6)

print(f'x: {x}')
print(f'y: {y}')
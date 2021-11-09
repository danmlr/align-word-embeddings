from tomaster import tomato
from sklearn import datasets

X, y = datasets.make_moons(n_samples=1000, noise=0.05, random_state=1337)
res = tomato(points=X, k=5, raw=True)

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize

weights = np.array([-5, -3, 1])
weights = weights - min(weights)
weights = weights *  [1, 0, 1]
weights = normalize(weights[:,np.newaxis], axis=0).ravel()
multinomial = np.random.multinomial(100, weights, size=1)


print(multinomial)

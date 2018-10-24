from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
import numpy as np

X = 30 * np.random.randn(1000000, 2) + 5000
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
#X = np.r_[X + 2, X - 2, X_outliers]

outliers_fraction = 0.0001

model = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.000999)

model.fit(X)
y_pred = model.predict(X)
y_pred_outlier = model.predict(X_outliers)

print(y_pred[-100:])
print(np.average(y_pred))
print(y_pred_outlier)
print(np.average(y_pred_outlier))
# import dataset - csv using pandas
import pandas as pd
train_data = pd.read_csv('dataset/train.csv')
label = train_data['label']
features = train_data.drop(['label'], axis=1)
# convert data into numpy array

import numpy as np
features = np.array(features)
label = np.array(label)

# split features and labels
from sklearn.model_selection import train_test_split
# split training and test set
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

# import model - from sklearn - knn
knn = KNeighborsClassifier(n_neighbors=7)
# fit model
knn.fit(x_train, y_train)

y_predict = knn.predict(x_test)
print(knn.score(x_test, y_test))
# test
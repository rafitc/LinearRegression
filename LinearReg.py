import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit

uh = pd.read_csv('USA_Housing.csv')
#uh.head()
#uh.info()
#uh.describe()
#uh.coloumns()
sns.distplot(uh['Price'])
#plt.matshow(uh.corr())
#plt.xticks(range(len(uh.columns)), uh.columns)
#plt.yticks(range(len(uh.columns)), uh.columns)
#plt.colorbar()
#plt.show()

X = uh[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = uh['Price']
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 101)
lm = LinearRegression()
lm.fit(X_train,y_train)
print("Trained dataset")

prediction = lm.predict(X_test)
plt.scatter(y_test,prediction)

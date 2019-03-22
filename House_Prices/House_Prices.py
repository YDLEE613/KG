import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # statistical data visual
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew

import warnings
pd.set_option('display.max_rows', 90)
pd.options.mode.chained_assignment = None

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn


# Read train and test set
dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')

dataset.shape
testset.shape

'''
            TRAIN SET

            Target variable: SalePrice
'''


# plot distribution of train set before normalizing
sns.distplot(dataset['SalePrice'], fit=norm);
(mu, sigma) = norm.fit(dataset['SalePrice'])
print('\n mu={:.2f} and sigma {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)])
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(dataset['SalePrice'], plot=plt)
plt.show()

# As the plot is right-skewed, use Log transformation to normalize
#dataset['SalePrice'] = np.log1p(dataset['SalePrice'])

# plot distribution train set after normalizing
sns.distplot(dataset['SalePrice'], fit=norm);
(mu, sigma) = norm.fit(dataset['SalePrice'])
print('\n mu={:.2f} and sigma {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)])
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(dataset['SalePrice'], plot=plt)
plt.show()


# Check correlation of train set 
corrmat = dataset.corr()
corrList = list(corrmat.sort_values(by = 'SalePrice',ascending=False))
#plt.subplots(figsize=(10,9))
#sns.heatmap(corrmat, vmax=0.9, square=True)

'''
Feature Engineering
'''
dataMissing = dataset.isnull().sum().sort_values(ascending=False).head(19)
missingType = pd.DataFrame( dataset[dataMissing.index].dtypes, dataMissing.index)
missingType

# Fill with none
dataset['PoolQC'].fillna('none', inplace=True)
dataset['MiscFeature'].fillna('none', inplace=True)
dataset['Alley'].fillna('none', inplace=True)
dataset['Fence'].fillna('none', inplace=True)
dataset['FireplaceQu'].fillna('none', inplace=True)
dataset['GarageCond'].fillna('none', inplace=True)
dataset['GarageType'].fillna('none', inplace=True)
dataset['GarageFinish'].fillna('none', inplace=True)
dataset['GarageQual'].fillna('none', inplace=True)
dataset['BsmtExposure'].fillna('none', inplace=True)
dataset['BsmtFinType2'].fillna('none', inplace=True)
dataset['BsmtFinType1'].fillna('none', inplace=True)
dataset['BsmtCond'].fillna('none', inplace=True)
dataset['BsmtQual'].fillna('none', inplace=True)
dataset['MasVnrType'].fillna('none', inplace=True)
dataset['Electrical'].fillna('none', inplace=True)

# Fill with 0
dataset['GarageYrBlt'].fillna(0, inplace=True)
dataset['MasVnrArea'].fillna(0, inplace=True)

# Fill with median
dataset['LotFrontage'] = dataset.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

cols = ['MSSubClass', 'MoSold', 'YrSold']

train_set = dataset[corrList].drop(cols, axis=1)
train_set.isnull().sum()
train_set.dtypes

'''
                TEST SET
'''
corrList.remove('SalePrice')
test_set = testset[corrList].drop(cols, axis=1)
test_set.isnull().sum()

''' Feature Engineering '''
test_set['LotFrontage'] = testset.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
test_set['GarageYrBlt'].fillna(0, inplace=True)
test_set['MasVnrArea'].fillna(0, inplace=True)
test_set['BsmtFinSF1'].fillna(0, inplace=True)
test_set['BsmtFinSF2'].fillna(0, inplace=True)
test_set['BsmtUnfSF'].fillna(0, inplace=True)
test_set['TotalBsmtSF'].fillna(0, inplace=True)
test_set['BsmtFullBath'].fillna(0, inplace=True)
test_set['BsmtHalfBath'].fillna(0, inplace=True)
test_set['GarageCars'].fillna(0, inplace=True)
test_set['GarageArea'].fillna(0, inplace=True)

    
y_train = train_set.SalePrice
test_pass = test_set['Id']
x_train = train_set.drop(['Id','SalePrice'], axis=1).copy()
x_test = test_set.drop('Id', axis=1).copy()

x_train.isnull().sum()
x_test.isnull().sum()

x_train.dtypes
x_test.dtypes
y_train = y_train.astype('int')

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

### Logistic Regression ###
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state=0)
logistic.fit(x_train, y_train)
y_pred = logistic.predict(x_test)
acc_log = round(logistic.score(x_train, y_train) * 100,2)

'''
train_df = x_train.copy()
coeff_df = pd.DataFrame(train_df.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logistic.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
'''
### Support Vector Machine ###
from sklearn.svm import SVR
svr = SVR()
svr.fit(x_train, y_train)
y_pred = svr.predict(x_test)
acc_svc = round(svr.score(x_train, y_train)*100, 2)

### K-NN ###
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) *100, 2)

### Decision Tree ###
from sklearn.tree import DecisionTreeRegressor
classifier = DecisionTreeRegressor(random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc_dec = round(classifier.score(x_train, y_train) *100, 2)

### Random Forest ###
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
acc_random_forest = round(random_forest.score(x_train, y_train) *100, 2)



submission = pd.DataFrame({
                           'Id': test_pass,
                           'SalePrice': y_pred
                           })
submission.to_csv('submission.csv', index=False)






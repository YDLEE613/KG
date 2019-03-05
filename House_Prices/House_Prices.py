import pandas as pd

pd.set_option('display.max_rows', 90)

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

train_set.shape
train_set.isnull().sum()

test_set.shape
test_set.isnull().sum()

types = train_set.dtypes.copy()
types_null = pd.DataFrame(train_set.isnull().sum(), index = types.index)

'''
### drop rows that have relatively small number of rows with missing values ###
train_set.dropna(axis=0, subset=['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                               'BsmtFinType2', 'GarageQual', 'GarageCond'], inplace=True)

### drop rows that have relatively small number of rows with missing values ###
test_set.dropna(axis=0, subset=['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                               'BsmtFinType2', 'GarageQual', 'GarageCond'], inplace=True)
'''

'''
Only consider columns with quality rating 
# OverallQual
# OverallCond
# ExterQual
# ExterCond
# HeatinQC
# KitchenQual
# MiscVal --> value of miscellaneous feature

# BsmtQual     -> 37 null values
# BsmtCond     -> 37 null values
# BsmtExposure -> 38 null values
# BsmtFinType1 -> 37 null values
# BsmtFinType2 -> 38 null values
# FireplaceQu -> 690 null values
# GarageQual -> 81 null values
# GarageCond -> 81 null values
# PoolQC  -> 1453 null values
# Fence   -> 1179 null values 

# GrLivArea    -> total area
# TotRmsAbvGrd

'''

### Extract columns that are needed ###
dataset = train_set[['OverallQual', 'OverallCond','ExterQual','ExterCond','HeatingQC','KitchenQual',
                     'MiscVal','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                     'FireplaceQu','GarageQual','GarageCond','PoolQC','Fence', 'GrLivArea', 'TotRmsAbvGrd']]
                     
                     
### Extract columns that are needed ###
testset = test_set[['OverallQual', 'OverallCond','ExterQual','ExterCond','HeatingQC','KitchenQual',
                     'MiscVal','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                     'FireplaceQu','GarageQual','GarageCond','PoolQC','Fence', 'GrLivArea', 'TotRmsAbvGrd']]

dataset.isnull().sum()
testset.isnull().sum()

### Fill null values with 'none' and change to ordinal data ###
# Train set
dataset.BsmtQual.fillna('none', inplace=True) 
dataset.BsmtCond.fillna('none', inplace=True)
dataset.BsmtExposure.fillna('none', inplace=True)
dataset.BsmtFinType1.fillna('none', inplace=True)
dataset.BsmtFinType2.fillna('none', inplace=True)
dataset.GarageQual.fillna('none', inplace=True)
dataset.GarageCond.fillna('none', inplace=True)
# Test set
testset.KitchenQual.fillna('none', inplace=True)
testset.BsmtQual.fillna('none', inplace=True)
testset.BsmtCond.fillna('none', inplace=True)
testset.BsmtExposure.fillna('none', inplace=True)
testset.BsmtFinType1.fillna('none', inplace=True)
testset.BsmtFinType2.fillna('none', inplace=True)
testset.GarageQual.fillna('none', inplace=True)
testset.GarageCond.fillna('none', inplace=True)

### Map quality to ordinal data ###
quality = {'Ex':5, 'Gd':4, 'TA':3, 'Av':3, 'Fa':2, 'Po':1, 'Mn':1, 'No':1, 'none':0}
quality2 = {'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':3, 'LwQ':2, 'Unf':1, 'none':0}

# Train set
dataset.ExterQual = [quality[item] for item in dataset.ExterQual]
dataset.ExterCond = [quality[item] for item in dataset.ExterCond]
dataset.HeatingQC = [quality[item] for item in dataset.HeatingQC]
dataset.KitchenQual = [quality[item] for item in dataset.KitchenQual]
dataset.BsmtQual = [quality[item] for item in dataset.BsmtQual]
dataset.BsmtCond = [quality[item] for item in dataset.BsmtCond]
dataset.BsmtExposure = [quality[item] for item in dataset.BsmtExposure]
dataset.GarageQual = [quality[item] for item in dataset.GarageQual]
dataset.GarageCond = [quality[item] for item in dataset.GarageCond]
dataset.BsmtFinType1 = [quality2[item] for item in dataset.BsmtFinType1]
dataset.BsmtFinType2 = [quality2[item] for item in dataset.BsmtFinType2]
# Test set
testset.ExterQual = [quality[item] for item in testset.ExterQual]
testset.ExterCond = [quality[item] for item in testset.ExterCond]
testset.HeatingQC = [quality[item] for item in testset.HeatingQC]
testset.KitchenQual = [quality[item] for item in testset.KitchenQual]
testset.BsmtQual = [quality[item] for item in testset.BsmtQual]
testset.BsmtCond = [quality[item] for item in testset.BsmtCond]
testset.BsmtExposure = [quality[item] for item in testset.BsmtExposure]
testset.GarageQual = [quality[item] for item in testset.GarageQual]
testset.GarageCond = [quality[item] for item in testset.GarageCond]
testset.BsmtFinType1 = [quality2[item] for item in testset.BsmtFinType1]
testset.BsmtFinType2 = [quality2[item] for item in testset.BsmtFinType2]


### Drop PoolQC, Fence and FireplaceQu due to too many houses without these. ###
dataset.drop(['PoolQC', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
testset.drop(['PoolQC', 'Fence', 'FireplaceQu'], axis=1, inplace=True)

### Group MiscVal###
# Train set
dataset['MiscValBand'] = pd.cut(dataset['MiscVal'],5)
miscval = dataset.MiscVal.copy()
miscval.value_counts

for i, val in miscval.iteritems():
    if val <= 3100:
        dataset.loc[i, 'MiscVal'] = 0
    elif val > 3100 and val <= 6200:
        dataset.loc[i, 'MiscVal'] = 1
    elif val > 6200 and val <= 9300:
        dataset.loc[i, 'MiscVal'] = 2
    elif val > 9300 and val <= 12400:
        dataset.loc[i, 'MiscVal'] = 3
    else:
        dataset.loc[i, 'MiscVal'] = 4

dataset.drop('MiscValBand', axis=1, inplace=True)

# Test set
miscval = testset.MiscVal.copy()

for i, val in miscval.iteritems():
    if val <= 3100:
        testset.loc[i, 'MiscVal'] = 0
    elif val > 3100 and val <= 6200:
        testset.loc[i, 'MiscVal'] = 1
    elif val > 6200 and val <= 9300:
        testset.loc[i, 'MiscVal'] = 2
    elif val > 9300 and val <= 12400:
        testset.loc[i, 'MiscVal'] = 3
    else:
        testset.loc[i, 'MiscVal'] = 4

### Group living area ###
# Train set
dataset['LivAreaBand'] = pd.cut(dataset['GrLivArea'], 5)
dataset['LivAreaBand']
livBand = dataset.GrLivArea.copy()

for i, val in livBand.iteritems():
    if val <= 1395.6:
        dataset.loc[i, 'GrLivArea'] = 0
    elif val > 1395.6 and val <= 2457.2:
        dataset.loc[i, 'GrLivArea'] = 1
    elif val > 2457.2 and val <= 3518.8:
        dataset.loc[i, 'GrLivArea'] = 2
    elif val > 3518.8 and val <= 4580.4:
        dataset.loc[i, 'GrLivArea'] = 3
    else:
        dataset.loc[i, 'GrLivArea'] = 4

dataset.drop('LivAreaBand', axis=1, inplace=True)

# Test set
livBand = testset.GrLivArea.copy()
for i, val in livBand.iteritems():
    if val <= 1395.6:
        testset.loc[i, 'GrLivArea'] = 0
    elif val > 1395.6 and val <= 2457.2:
        testset.loc[i, 'GrLivArea'] = 1
    elif val > 2457.2 and val <= 3518.8:
        testset.loc[i, 'GrLivArea'] = 2
    elif val > 3518.8 and val <= 4580.4:
        testset.loc[i, 'GrLivArea'] = 3
    else:
        testset.loc[i, 'GrLivArea'] = 4


'''

Machine Learning models

'''
y_train = train_set.SalePrice
test_pass = test_set.Id
x_train = dataset.copy()
x_test = testset.copy()


### Feature Scaling (due to MiscVal) ###
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

x_train.dtypes
x_test.dtypes
y_train.dtypes

### Logistic Regression ###
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state=0)
logistic.fit(x_train, y_train)
y_pred = logistic.predict(x_test)
acc_log = round(logistic.score(x_train, y_train) * 100,2)


### Support Vector Machine ###
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train)*100, 2)

### K-NN ###
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) *100, 2)


### Decision Tree ###
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc_dec = round(classifier.score(x_train, y_train) *100, 2)

### Random Forest ###
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
acc_random_forest = round(random_forest.score(x_train, y_train) *100, 2)

### XGBoost ###
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
acc_xgb = round(xgb.score(x_train, y_train) *100, 2)



submission = pd.DataFrame({
                           'Id': test_pass,
                           'SalePrice': y_pred
                           })
submission.to_csv('submission.csv', index=False)

submission.shape












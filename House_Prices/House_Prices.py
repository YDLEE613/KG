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
'''

### Extract columns that are needed ###
dataset = train_set[['OverallQual', 'OverallCond','ExterQual','ExterCond','HeatingQC','KitchenQual',
                     'MiscVal','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                     'FireplaceQu','GarageQual','GarageCond','PoolQC','Fence']]
                     
                     
### Extract columns that are needed ###
testset = test_set[['OverallQual', 'OverallCond','ExterQual','ExterCond','HeatingQC','KitchenQual',
                     'MiscVal','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                     'FireplaceQu','GarageQual','GarageCond','PoolQC','Fence']]


### Fill null values with 'none' and change to ordinal data ###
dataset.BsmtQual = dataset.BsmtQual.fillna(value='none')
dataset.BsmtCond = dataset.BsmtCond.fillna(value='none')
dataset.BsmtExposure = dataset.BsmtExposure.fillna(value='none')
dataset.BsmtFinType1 = dataset.BsmtFinType1.fillna(value='none')
dataset.BsmtFinType2 = dataset.BsmtFinType2.fillna(value='none')
dataset.GarageQual = dataset.GarageQual.fillna(value='none')
dataset.GarageCond = dataset.GarageCond.fillna(value='none')
dataset.FireplaceQu = dataset.FireplaceQu.fillna(value='none')
dataset.PoolQC = dataset.PoolQC.fillna(value='none')
dataset.Fence = dataset.Fence.fillna(value='none')

### Map quality to ordinal data ###
dataset.ExterQual = dataset.ExterQual.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}).astype('int')
dataset.ExterCond = dataset.ExterCond.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}).astype('int')
dataset.HeatingQC = dataset.HeatingQC.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}).astype('int')
dataset.KitchenQual = dataset.KitchenQual.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}).astype('int')

### Map quality to ordinal data ###
dataset.BsmtQual = dataset.BsmtQual.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'none':0}).astype('int')
dataset.BsmtCond = dataset.BsmtCond.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'none':0}).astype('int')
dataset.BsmtExposure = dataset.BsmtExposure.map({'Gd':3, 'Av':2, 'Mn':1, 'No':0, 'none':0}).astype('int')
dataset.BsmtFinType1 = dataset.BsmtFinType1.map({'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':2, 'LwQ':1, 'Unf':0, 'none':0}).astype('int')
dataset.BsmtFinType2 = dataset.BsmtFinType2.map({'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':2, 'LwQ':1, 'Unf':0, 'none':0}).astype('int')
dataset.GarageQual = dataset.GarageQual.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'none':0}).astype('int')
dataset.GarageCond = dataset.GarageCond.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'none':0}).astype('int')



dataset.FireplaceQu = dataset.FireplaceQu.map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'none':0}).astype('int')
dataset.PoolQC = dataset.PoolQC.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'none':0}).astype('int')
dataset.Fence = dataset.Fence.map({'GdPrv':2, 'MnPrv':1, 'GdWo':2, 'MnWw':1, 'none':0}).astype('int')

dataset.isnull().sum()


### Comine Overal Qual and Cond ? ###




'''

TEST SET

'''
                     
testset.isnull().sum()

### Fill null values with 'none' and change to ordinal data ###
testset.KitchenQual = testset.KitchenQual.fillna(value='none')
testset.BsmtQual = testset.BsmtQual.fillna(value='none')
testset.BsmtCond = testset.BsmtCond.fillna(value='none')
testset.BsmtExposure = testset.BsmtExposure.fillna(value='none')
testset.BsmtFinType1 = testset.BsmtFinType1.fillna(value='none')
testset.BsmtFinType2 = testset.BsmtFinType2.fillna(value='none')
testset.GarageQual = testset.GarageQual.fillna(value='none')
testset.GarageCond = testset.GarageCond.fillna(value='none')
testset.FireplaceQu = testset.FireplaceQu.fillna(value='none')
testset.PoolQC = testset.PoolQC.fillna(value='none')
testset.Fence = testset.Fence.fillna(value='none')

### Map quality to ordinal data ###
testset.ExterQual = testset.ExterQual.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}).astype('int')
testset.ExterCond = testset.ExterCond.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}).astype('int')
testset.HeatingQC = testset.HeatingQC.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}).astype('int')
testset.KitchenQual = testset.KitchenQual.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'none':0}).astype('int')



### Map quality to ordinal data ###
testset.BsmtQual = testset.BsmtQual.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'none':0}).astype('int')
testset.BsmtCond = testset.BsmtCond.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'none':0}).astype('int')
testset.BsmtExposure = testset.BsmtExposure.map({'Gd':3, 'Av':2, 'Mn':1, 'No':0, 'none':0}).astype('int')
testset.BsmtFinType1 = testset.BsmtFinType1.map({'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':2, 'LwQ':1, 'Unf':0, 'none':0}).astype('int')
testset.BsmtFinType2 = testset.BsmtFinType2.map({'GLQ':5, 'ALQ':4, 'BLQ':3, 'Rec':2, 'LwQ':1, 'Unf':0, 'none':0}).astype('int')
testset.GarageQual = testset.GarageQual.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'none':0}).astype('int')
testset.GarageCond = testset.GarageCond.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0, 'none':0}).astype('int')


testset.FireplaceQu = testset.FireplaceQu.map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'none':0}).astype('int')
testset.PoolQC = testset.PoolQC.map({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'none':0}).astype('int')
testset.Fence = testset.Fence.map({'GdPrv':2, 'MnPrv':1, 'GdWo':2, 'MnWw':1, 'none':0}).astype('int')

testset.isnull().sum()

### Check dataset and testset have same dimension ###
dataset.shape
testset.shape


'''

Machine Learning models

'''
y_train = train_set.SalePrice
test_pass = test_set.Id
x_train = dataset.copy()
x_test = testset.copy()


'''
### Feature Scaling (due to MiscVal) ###
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
'''

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

'''
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
'''


submission = pd.DataFrame({
                           'Id': test_pass,
                           'SalePrice': y_pred
                           })
submission.to_csv('submission.csv', index=False)

submission.shape












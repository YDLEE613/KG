### import libraries
import pandas as pd # manage dataset
import numpy as np

### import dataset
dataset = pd.read_csv('train.csv')

### Drop Cabin column
dataset.drop('Cabin', axis = 1, inplace=True)

### fill missing values for Embarked with most frequent ###
dataset['Embarked'].value_counts() # S
dataset.Embarked.fillna('S', inplace=True)
assert not dataset.Embarked.isnull().any()

### Construct Title column out of Name column ###
tmp = dataset['Name'].str.split(".", n=1, expand = True)
titles = tmp[0].str.split(",", expand = True)
dataset['Title'] = titles[1].str.strip()
true_names = dataset['Name']
dataset.drop('Name', axis=1, inplace = True)

### Change Title to Mr, Mrs, Master or Miss
#dataset.Title.value_counts()

dataset['Title'].value_counts()

conditions = [(dataset['Title'] == 'Dr') | (dataset['Title'] == 'Rev') | (dataset['Title'] == 'Mlle') 
                | (dataset['Title'] == 'Major') | (dataset['Title'] == 'Col') | (dataset['Title'] == 'Ms')
                | (dataset['Title'] == 'the Countess') | (dataset['Title'] == 'Sir') | (dataset['Title'] == 'Lady')
                | (dataset['Title'] == 'Jonkheer') | (dataset['Title'] == 'Don') | (dataset['Title'] == 'Capt')
                | (dataset['Title'] == 'Mme'),
                (dataset['Title'] == 'Mr'),
                (dataset['Title'] == 'Miss'),
                (dataset['Title'] == 'Master')] 
               
choices = ['Misc', 'Mr', 'Miss', 'Master']
dataset['Title'] = np.select(conditions, choices, default='Mrs')

dataset['Title'].value_counts()



### Change Sex to number (female:0, male:1) ###
dataset.Sex = dataset.Sex.map({'female':0, 'male':1}).astype('int')
dataset.Sex.unique()



### Change Title to Categorical Data ###
# Title vs Survived ratio

title_surv = dataset[dataset['Survived'] == 1].groupby('Title')['Survived'].count()
total_title = dataset.Title.value_counts()
print('Survival rate for Mr: ' , round(title_surv['Mr']/total_title['Mr'] *100,1),'%')
print('Survival rate for Miss: ' , round(title_surv['Miss']/total_title['Miss'] *100,1),'%')
print('Survival rate for Mrs: ' , round(title_surv['Mrs']/total_title['Mrs'] *100,1),'%')
print('Survival rate for Master: ' , round(title_surv['Master']/total_title['Master'] *100,1),'%')

# Mrs: 79.7     --> 3
# Miss: 70.3      --> 2
# Master: 57.5  --> 1
# Mr: 16.2      --> 0

# Ordinal Data based on Survival ratio
dataset.Title = dataset.Title.map({'Misc':4,'Mrs':3, 'Miss':2, 'Master':1, 'Mr':0}).astype('int')


### Calculate missing values for Age (use Standard deviation) ###
import random as rd
each_mean = np.arange(4)
each_std = np.arange(4)

# mean calculations
for i in range(4):
    age_each = dataset[dataset['Title'] == i].Age.copy()
    age_each_mean = age_each[age_each.notnull()].describe()['mean']
    age_each_std = age_each[age_each.notnull()].describe()['std']
                        
    for j, val in age_each.isnull().iteritems():
        if val == True:
            dataset.loc[j, 'Age'] = rd.uniform(age_each_mean-age_each_std, age_each_mean+age_each_std+1)
    
dataset['AgeBand'] = pd.cut(dataset['Age'],5)        
dataset[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean().sort_values(by='AgeBand', ascending=True)

age = dataset.Age.copy()
for i, val in age.iteritems():
    if val <= 16:
        dataset.loc[i, 'Age'] = 0
    elif val > 16 and val <= 32:
        dataset.loc[i, 'Age'] = 1
    elif val > 32 and val <= 48:
        dataset.loc[i, 'Age'] = 2
    elif val > 48 and val <= 64:
        dataset.loc[i, 'Age'] = 3
    else:
        dataset.loc[i, 'Age'] = 4

dataset.drop(['AgeBand', 'Fare', 'Ticket'], axis=1, inplace=True)
    




### Change Embarked to Categorical Data ###
# Embarked vs Survived ratio
# S: 33.9 --> 0
# Q: 39.0 --> 1
# C: 55.4 --> 2

# Ordinal data
dataset.Embarked = dataset.Embarked.map({'C':2, 'Q':1, 'S':0}).astype('int')



### Add SibSp + Parch to check if the passenger was alone ###
dataset['Alone'] = dataset['Parch'] + dataset['SibSp']

conditions = [dataset['Alone'] == 0]
choices = [1]
dataset['Alone'] = np.select(conditions, choices, default = 0)
dataset.drop(['SibSp','Parch'], axis=1, inplace=True)



'''

TEST SET 


'''


### import testset
testset = pd.read_csv('test.csv')

### Drop Cabin column
testset.drop('Cabin', axis = 1, inplace=True)

### fill missing values for Embarked with most frequent ###
testset['Embarked'].value_counts() # S
testset.Embarked.fillna('S', inplace=True)
assert not testset.Embarked.isnull().any()

### Construct Title column out of Name column ###
tmp = testset['Name'].str.split(".", n=1, expand = True)
titles = tmp[0].str.split(",", expand = True)
testset['Title'] = titles[1].str.strip()
true_names = testset['Name']
testset.drop('Name', axis=1, inplace = True)

### Change Title to Mr, Mrs, Master or Miss

conditions = [(testset['Title'] == 'Dr') | (testset['Title'] == 'Rev') | (testset['Title'] == 'Mlle') 
                | (testset['Title'] == 'Major') | (testset['Title'] == 'Col') | (testset['Title'] == 'Ms')
                | (testset['Title'] == 'the Countess') | (testset['Title'] == 'Sir') | (testset['Title'] == 'Lady')
                | (testset['Title'] == 'Jonkheer') | (testset['Title'] == 'Don') | (testset['Title'] == 'Capt')
                | (testset['Title'] == 'Mme'),
                (testset['Title'] == 'Mr'),
                (testset['Title'] == 'Miss'),
                (testset['Title'] == 'Master')] 
               
choices = ['Misc', 'Mr', 'Miss', 'Master']
testset['Title'] = np.select(conditions, choices, default='Mrs')

### Change Sex to number (female:0, male:1) ###
testset.Sex = testset.Sex.map({'female':0, 'male':1}).astype('int')
testset.Sex.unique()



### Change Title to Categorical Data ###
# Title vs Survived ratio
# Mrs: 79.7     --> 3
# Miss: 70.3      --> 2
# Master: 57.5  --> 1
# Mr: 16.2      --> 0

# Ordinal Data based on Survival ratio
testset.Title = testset.Title.map({'Misc':4,'Mrs':3, 'Miss':2, 'Master':1, 'Mr':0}).astype('int')

### Calculate missing values for Age (use Standard deviation) ###
import random as rd
each_mean = np.arange(4)
each_std = np.arange(4)

# mean calculations
for i in range(4):
    age_each = testset[testset['Title'] == i].Age.copy()
    age_each_mean = age_each[age_each.notnull()].describe()['mean']
    age_each_std = age_each[age_each.notnull()].describe()['std']
                        
    for j, val in age_each.isnull().iteritems():
        if val == True:
            testset.loc[j, 'Age'] = rd.uniform(age_each_mean-age_each_std, age_each_mean+age_each_std+1)
    

age = testset.Age.copy()
for i, val in age.iteritems():
    if val <= 16:
        testset.loc[i, 'Age'] = 0
    elif val > 16 and val <= 32:
        testset.loc[i, 'Age'] = 1
    elif val > 32 and val <= 48:
        testset.loc[i, 'Age'] = 2
    elif val > 48 and val <= 64:
        testset.loc[i, 'Age'] = 3
    else:
        testset.loc[i, 'Age'] = 4



### Change Embarked to Categorical Data ###
# Embarked vs Survived ratio

# Ordinal data
testset.Embarked = testset.Embarked.map({'C':2, 'Q':1, 'S':0}).astype('int')

testset.drop(['Fare','Ticket'], axis=1, inplace=True)



### Add SibSp + Parch to check if the passenger was alone ###
testset['Alone'] = testset['Parch'] + testset['SibSp']

conditions = [testset['Alone'] == 0]
choices = [1]
testset['Alone'] = np.select(conditions, choices, default = 0)
testset.drop(['SibSp','Parch'], axis=1, inplace=True)

'''
Machine learning models
'''
y_train = dataset.Survived
test_pass = testset.PassengerId
x_train = dataset.drop(['Survived', 'PassengerId'], axis=1)
x_test = testset.drop('PassengerId', axis=1)


# FEATURE SCALING
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


### Decision Tree ###
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc_dec = round(classifier.score(x_train, y_train) *100, 2)

### K-NN ###
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) *100, 2)

### Support Vector Machine ###
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train)*100, 2)

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
                           'PassengerId': test_pass,
                           'Survived': y_pred
                           })
submission.to_csv('submission.csv', index=False)





















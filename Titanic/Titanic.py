
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

### Change Title to Mr, Mrs, Master or Ms
#dataset.Title.value_counts()
conditions = [(dataset['Title'] == 'the Countess') | (dataset['Title'] == 'Mme') | (dataset['Title'] == 'Mrs') 
                | (dataset['Title'] == 'Lday') | (dataset['Title'] == 'Dr') & (dataset['Sex'] == 'female'),
               (dataset['Title'] == 'Mlle') | (dataset['Title'] == 'Ms') | (dataset['Title'] == 'Miss'),
               (dataset['Title'] == 'Master')]
choices = ['Mrs', 'Ms', 'Master']
dataset['Title'] = np.select(conditions, choices, default = 'Mr')


### Change Sex to number (female:0, male:1) ###
dataset.Sex = dataset.Sex.map({'female':0, 'male':1}).astype('int')
dataset.Sex.unique()



### Calculate missing values for Age (use Standard deviation) ###
import random as rd

# fill missing value in Age for Master
age_Master = dataset[dataset['Title'] =='Master'].Age.copy()
age_Master_missing = dataset[dataset['Age'].isnull()].copy()
age_Master_missing = age_Master_missing[age_Master_missing['Title']=='Master'].Age
age_Master = age_Master.dropna(how='all')
assert not age_Master.isnull().any()

age_Master_mean = age_Master.describe()['mean'] 
age_Master_std = age_Master.describe()['std']   
age_min = age_Master_mean - age_Master_std
age_max = age_Master_mean + age_Master_std

for i,val in age_Master_missing.iteritems():
    age_Master_missing[i] = int(rd.uniform(age_min, age_max+1))

missing_index = age_Master_missing.keys().tolist()

for i, val in age_Master_missing.iteritems():
    dataset.loc[i, 'Age'] = val


# fill missing value in Age for Mr
age_Mr = dataset[dataset['Title'] == 'Mr'].Age.copy()
age_Mr_missing = dataset[dataset['Age'].isnull()].copy()
age_Mr_missing = age_Mr_missing[age_Mr_missing['Title'] == 'Mr'].Age
age_Mr = age_Mr.dropna(how='all')
assert not age_Mr.isnull().any()

age_Mr_mean = age_Mr.describe()['mean'] 
age_Mr_std = age_Mr.describe()['std']  
age_min = age_Mr_mean - age_Mr_std
age_max = age_Mr_mean + age_Mr_std

for i,val in age_Mr_missing.iteritems():
    age_Mr_missing[i] = int(rd.uniform(age_min, age_max+1))

missing_index = age_Mr_missing.keys().tolist()

for i, val in age_Mr_missing.iteritems():
    dataset.loc[i, 'Age'] = val

# fill missing value in Age for Mrs
age_Mrs = dataset[dataset['Title'] == 'Mrs'].Age.copy()
age_Mrs_missing = dataset[dataset['Age'].isnull()].copy()
age_Mrs_missing = age_Mrs_missing[age_Mrs_missing['Title'] == 'Mrs'].Age
age_Mrs = age_Mrs.dropna(how='all')
assert not age_Mrs.isnull().any()

age_Mrs_mean = age_Mrs.describe()['mean']
age_Mrs_std = age_Mrs.describe()['std']  
age_min = age_Mrs_mean - age_Mrs_std
age_max = age_Mrs_mean + age_Mrs_std

for i,val in age_Mrs_missing.iteritems():
    age_Mrs_missing[i] = int(rd.uniform(age_min, age_max+1))

missing_index = age_Mrs_missing.keys().tolist()

for i, val in age_Mrs_missing.iteritems():
    dataset.loc[i, 'Age'] = val

# fill missing value in Age for Ms
age_Ms = dataset[dataset['Title'] == 'Ms'].Age.copy()
age_Ms_missing = dataset[dataset['Age'].isnull()].copy()
age_Ms_missing = age_Ms_missing[age_Ms_missing['Title'] == 'Ms'].Age
age_Ms = age_Ms.dropna(how='all')
assert not age_Ms.isnull().any()

age_Ms_mean = age_Ms.describe()['mean'] 
age_Ms_std = age_Ms.describe()['std']   
age_min = age_Ms_mean - age_Ms_std
age_max = age_Ms_mean + age_Ms_std

for i,val in age_Ms_missing.iteritems():
    age_Ms_missing[i] = int(rd.uniform(age_min, age_max+1))

missing_index = age_Ms_missing.keys().tolist()

for i, val in age_Ms_missing.iteritems():
    dataset.loc[i, 'Age'] = val

### Change Title to Categorical Data ###
# Title vs Survived ratio
'''
import matplotlib.pyplot as plt

color_survived = '#57e8fc'
color_dead = '#fc5e57'
counts = dataset.Survived.value_counts()
sizes = [counts[0],counts[1]] # dead, survived
colors = [color_dead, color_survived]

ct = pd.crosstab(dataset.Title, dataset.Survived)
survived_vals = [ct[1][0], ct[1][1], ct[1][2], ct[1][3]]
dead_vals = [ct[0][0], ct[0][1], ct[0][2], ct[0][3]]

width = 0.3
ind = np.arange(4)

plt.bar(ind, survived_vals, width, label = 'Survived', color = color_survived)
plt.bar(ind+width, dead_vals, width, label = 'Dead', color = color_dead)

plt.xticks(ind+width/2, ('Master', 'Mr', 'Mrs', 'Ms'))
plt.yticks(np.arange(0, 600, 50))
plt.legend(loc='upper right')
plt.show()
'''
'''
title_surv = dataset[dataset['Survived'] == 1].groupby('Title')['Survived'].count()
total_title = dataset.Title.value_counts()
print('Survival rate for Mr: ' , round(title_surv['Mr']/total_title['Mr'] *100,1),'%')
print('Survival rate for Ms: ' , round(title_surv['Ms']/total_title['Ms'] *100,1),'%')
print('Survival rate for Mrs: ' , round(title_surv['Mrs']/total_title['Mrs'] *100,1),'%')
print('Survival rate for Master: ' , round(title_surv['Master']/total_title['Master'] *100,1),'%')
'''
# Mrs: 79.7     --> 3
# Ms: 70.3      --> 2
# Master: 57.5  --> 1
# Mr: 16.2      --> 0

# Ordinal Data based on Survival ratio
dataset.Title = dataset.Title.map({'Mrs':3, 'Ms':2, 'Master':1, 'Mr':0}).astype('int')

'''
# OHE data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(dataset.Title)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
onehotEncoder = OneHotEncoder()
onehot_encoded = onehotEncoder.fit_transform(integer_encoded).toarray()
dataset['Mr'] = onehot_encoded[:,1]
dataset['Mrs'] = onehot_encoded[:,2]
dataset['Ms'] = onehot_encoded[:,3]
dataset['Master'] = onehot_encoded[:,0]
dataset.drop('Title', axis = 1, inplace = True)
'''


### Change Embarked to Categorical Data ###
# Embarked vs Survived ratio
'''
ct = pd.crosstab(dataset.Embarked, dataset.Survived)
survival_vals = ct[1]
dead_vals = ct[0]
ind = np.arange(3)

plt.bar(ind, survival_vals, width, label='Survived', color = color_survived)
plt.bar(ind+width, dead_vals, width, label='dead', color = color_dead)

plt.xticks(ind+width/2, ('C', 'Q,', 'S'))
plt.yticks(np.arange(0, 600, 50))
plt.legend(loc='upper right')
plt.show()
'''
'''
embark_surv = dataset[dataset['Survived'] == 1].groupby('Embarked')['Survived'].count()
total_embark = dataset.Embarked.value_counts()
print('Survival rate for S: ', round(embark_surv['S']/total_embark['S']*100, 1), '%')
print('Survival rate for Q: ', round(embark_surv['Q']/total_embark['Q']*100, 1), '%')
print('Survival rate for C: ', round(embark_surv['C']/total_embark['C']*100, 1), '%')
'''
# S: 33.9 --> 0
# Q: 39.0 --> 1
# C: 55.4 --> 2

# Ordinal data
dataset.Embarked = dataset.Embarked.map({'C':2, 'Q':1, 'S':0}).astype('int')

'''
# OHE data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(dataset.Embarked)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehotEncoder = OneHotEncoder()
onehot_encoded = onehotEncoder.fit_transform(integer_encoded).toarray()
dataset['Embark_C'] = onehot_encoded[:,0]
dataset['Embark_Q'] = onehot_encoded[:,1]
dataset['Embark_S'] = onehot_encoded[:,2]
dataset.drop('Embarked', axis=1, inplace=True)
'''

### Calculate the fare per person ###
ticket_counts = dataset.Ticket.value_counts()
num_ticket = pd.Series([-1] *len(dataset.Ticket))

for i, val in dataset.Ticket.iteritems():
    num_ticket[i] = ticket_counts[val]

dataset['Num_Ticket'] = num_ticket # number of same tickets
dataset['Fare_pp'] = dataset['Fare']/dataset['Num_Ticket']
dataset.drop('Ticket', axis=1,inplace=True)
dataset.drop('Num_Ticket', axis=1,inplace=True)

#dataset.to_csv('final_data.csv')


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

### Change Title to Mr, Mrs, Master or Ms
#testset.Title.value_counts()
conditions = [(testset['Title'] == 'the Countess') | (testset['Title'] == 'Mme') | (testset['Title'] == 'Mrs') 
                | (testset['Title'] == 'Lday') | (testset['Title'] == 'Dr') & (testset['Sex'] == 'female'),
               (testset['Title'] == 'Mlle') | (testset['Title'] == 'Ms') | (testset['Title'] == 'Miss'),
               (testset['Title'] == 'Master')]
choices = ['Mrs', 'Ms', 'Master']
testset['Title'] = np.select(conditions, choices, default = 'Mr')


### Change Sex to number (female:0, male:1) ###
testset.Sex = testset.Sex.map({'female':0, 'male':1}).astype('int')
testset.Sex.unique()


### Calculate missing values for Age (use Standard deviation) ###
import random as rd

# fill missing value in Age for Master
age_Master = testset[testset['Title'] =='Master'].Age.copy()
age_Master_missing = testset[testset['Age'].isnull()].copy()
age_Master_missing = age_Master_missing[age_Master_missing['Title']=='Master'].Age
age_Master = age_Master.dropna(how='all')
assert not age_Master.isnull().any()

age_Master_mean = age_Master.describe()['mean'] 
age_Master_std = age_Master.describe()['std']   
age_min = age_Master_mean - age_Master_std
age_max = age_Master_mean + age_Master_std

for i,val in age_Master_missing.iteritems():
    age_Master_missing[i] = int(rd.uniform(age_min, age_max+1))

missing_index = age_Master_missing.keys().tolist()

for i, val in age_Master_missing.iteritems():
    testset.loc[i, 'Age'] = val


# fill missing value in Age for Mr
age_Mr = testset[testset['Title'] == 'Mr'].Age.copy()
age_Mr_missing = testset[testset['Age'].isnull()].copy()
age_Mr_missing = age_Mr_missing[age_Mr_missing['Title'] == 'Mr'].Age
age_Mr = age_Mr.dropna(how='all')
assert not age_Mr.isnull().any()

age_Mr_mean = age_Mr.describe()['mean'] 
age_Mr_std = age_Mr.describe()['std']   
age_min = age_Mr_mean - age_Mr_std
age_max = age_Mr_mean + age_Mr_std

for i,val in age_Mr_missing.iteritems():
    age_Mr_missing[i] = int(rd.uniform(age_min, age_max+1))

missing_index = age_Mr_missing.keys().tolist()

for i, val in age_Mr_missing.iteritems():
    testset.loc[i, 'Age'] = val

# fill missing value in Age for Mrs
age_Mrs = testset[testset['Title'] == 'Mrs'].Age.copy()
age_Mrs_missing = testset[testset['Age'].isnull()].copy()
age_Mrs_missing = age_Mrs_missing[age_Mrs_missing['Title'] == 'Mrs'].Age
age_Mrs = age_Mrs.dropna(how='all')
assert not age_Mrs.isnull().any()

age_Mrs_mean = age_Mrs.describe()['mean'] 
age_Mrs_std = age_Mrs.describe()['std']   
age_min = age_Mrs_mean - age_Mrs_std
age_max = age_Mrs_mean + age_Mrs_std

for i,val in age_Mrs_missing.iteritems():
    age_Mrs_missing[i] = int(rd.uniform(age_min, age_max+1))

missing_index = age_Mrs_missing.keys().tolist()

for i, val in age_Mrs_missing.iteritems():
    testset.loc[i, 'Age'] = val

# fill missing value in Age for Ms
age_Ms = testset[testset['Title'] == 'Ms'].Age.copy()
age_Ms_missing = testset[testset['Age'].isnull()].copy()
age_Ms_missing = age_Ms_missing[age_Ms_missing['Title'] == 'Ms'].Age
age_Ms = age_Ms.dropna(how='all')
assert not age_Ms.isnull().any()

age_Ms_mean = age_Ms.describe()['mean'] 
age_Ms_std = age_Ms.describe()['std']   
age_min = age_Ms_mean - age_Ms_std
age_max = age_Ms_mean + age_Ms_std

for i,val in age_Ms_missing.iteritems():
    age_Ms_missing[i] = int(rd.uniform(age_min, age_max+1))

missing_index = age_Ms_missing.keys().tolist()

for i, val in age_Ms_missing.iteritems():
    testset.loc[i, 'Age'] = val

### Change Title to Categorical Data ###
# Title vs Survived ratio
# Mrs: 79.7     --> 3
# Ms: 70.3      --> 2
# Master: 57.5  --> 1
# Mr: 16.2      --> 0

# Ordinal Data based on Survival ratio
testset.Title = testset.Title.map({'Mrs':3, 'Ms':2, 'Master':1, 'Mr':0}).astype('int')

'''
# OHE data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(testset.Title)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
onehotEncoder = OneHotEncoder()
onehot_encoded = onehotEncoder.fit_transform(integer_encoded).toarray()
testset['Mr'] = onehot_encoded[:,1]
testset['Mrs'] = onehot_encoded[:,2]
testset['Ms'] = onehot_encoded[:,3]
testset['Master'] = onehot_encoded[:,0]
testset.drop('Title', axis = 1, inplace = True)
'''

### Change Embarked to Categorical Data ###
# Embarked vs Survived ratio

# Ordinal data
testset.Embarked = testset.Embarked.map({'C':2, 'Q':1, 'S':0}).astype('int')

'''
# OHE data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(testset.Embarked)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehotEncoder = OneHotEncoder()
onehot_encoded = onehotEncoder.fit_transform(integer_encoded).toarray()
testset['Embark_C'] = onehot_encoded[:,0]
testset['Embark_Q'] = onehot_encoded[:,1]
testset['Embark_S'] = onehot_encoded[:,2]
testset.drop('Embarked', axis=1, inplace=True)
'''

### Calculate the fare per person ###
ticket_counts = testset.Ticket.value_counts()
num_ticket = pd.Series([-1] *len(testset.Ticket))

for i, val in testset.Ticket.iteritems():
    num_ticket[i] = ticket_counts[val]
    if testset.Fare[i] == None:
        num_ticket[i] = 0

testset['Num_Ticket'] = num_ticket # number of same tickets
testset['Fare_pp'] = testset['Fare']/testset['Num_Ticket']
testset.drop('Ticket', axis=1,inplace=True)
testset.drop('Num_Ticket', axis=1,inplace=True)
test_pass = testset.PassengerId
testset.to_csv('final_test.csv')

'''
Train set and Test set
'''
testset.drop('Fare', axis=1, inplace=True)
dataset.drop('Fare', axis=1, inplace=True)

'''
Machine learning models
'''
y_train = dataset.Survived
x_train = dataset.drop(['Survived', 'PassengerId'], axis=1)
x_test = testset.drop('PassengerId', axis=1)

x_test.fillna(0, inplace=True)

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

'''
corr = pd.DataFrame(x_train.columns.delete(0))
corr.columns = ['Feature']
corr['Correlations'] = pd.Series(logistic.coef_[0])
corr.sort_values(by='Correlations', ascending=False)
'''
### Decision Tree ###
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc_dec = round(classifier.score(x_train, y_train) *100, 2)

acc_dec
acc_log

### K-NN ###

### Support Vector Machine ###

### Kernel SVM ###

### Naive Bayes ###

### Random Forest ###


submission = pd.DataFrame({
                           'PassengerId': test_pass,
                           'Survived': y_pred
                           })
submission.to_csv('submission.csv', index=False)























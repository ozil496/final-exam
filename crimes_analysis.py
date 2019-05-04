import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree

dataset = pd.read_csv("~/Documents/crimes.csv",usecols=[2,5,7,10,13,17],nrows=607647)
dataset = dataset.dropna()

print(dataset.head())

month_array = []
hour_array = []

for i in range(0,len(dataset)):
	month_array.append(int(dataset.iloc[:,0].values[i].split("/",1)[0],10))
	hour_array.append(int(dataset.iloc[:,0].values[i].split(" ")[1].split(":")[0],10))

month_df = pd.DataFrame({"Month": month_array})
hour_df = pd.DataFrame({"Hour": hour_array})

dataset = dataset.reset_index(drop=True)
month_df = month_df.reset_index(drop=True)
dataset = pd.concat([dataset,month_df],axis=1)

dataset = dataset.reset_index(drop=True)
hour_df = hour_df.reset_index(drop=True)
dataset = pd.concat([dataset,hour_df],axis=1)

print(dataset.head())

location_array = dataset.iloc[:,2].values

labelencoder_Y = LabelEncoder()
location_array = labelencoder_Y.fit_transform(location_array)

location_df = pd.DataFrame({"Location":location_array})
location_df = location_df.reset_index(drop=True)
dataset = dataset.reset_index(drop=True)
dataset = pd.concat([dataset,location_df],axis=1)

crimes_array = dataset.iloc[:,1].values

labelencoder_X = LabelEncoder()
crimes_array = labelencoder_X.fit_transform(crimes_array)

crimes_df = pd.DataFrame({"Crime":crimes_array})
crimes_df = crimes_df.reset_index(drop=True)
dataset = dataset.reset_index(drop=True)
dataset = pd.concat([dataset,crimes_df],axis=1)

print(dataset.head())

print("#######################################################################")
data = dataset.iloc[:,3:8]
target = dataset.iloc[:,9].values

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.25, random_state = 0)

linear_machine = linear_model.LinearRegression()
linear_machine.fit(data_training, target_training)
prediction = linear_machine.predict(data_test)

calculate_score = round(metrics.r2_score(target_test, prediction), 4)
print("Predicted vs Actuals for Crime: ", calculate_score)

print("#######################################################################")

data = data.values

kfold_machine = KFold(n_splits = 5)
kfold_machine.get_n_splits(data)
print(kfold_machine)

for training_index, test_index in kfold_machine.split(data):
	print("Training: ", training_index)
	print("Test: ", test_index)
	data_training, data_test = data[training_index], data[test_index]
	target_training, target_test = target[training_index], target[test_index]
	linear_machine = linear_model.LinearRegression()
	linear_machine.fit(data_training,target_training)
	prediction = linear_machine.predict(data_test)
	print(metrics.r2_score(target_test,prediction))

print("#######################################################################")
target = dataset.iloc[:,9].values

data = dataset.iloc[:,3:9]

#print(data.head())

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.2, random_state=1)

# n_estimators of 10 or 11 result in an error saying my computer does not have
# enough memory to run this decision tree, and lower numbers get killed before
# I get any results. 
random_forest_machine = RandomForestClassifier(n_estimators=11)

random_forest_machine.fit(data_training, target_training)

predictions = random_forest_machine.predict(data_test)

#Process is killed before reaching any of these lines
print(accuracy_score(target_test, predictions))

#There are 31 columsn below because my computer claimed that the data I was 
#passing into the random forest machine had a 31x31 shape. Unfortunately, my
#computer killed the program before I could see the results
confusion_matrix = pd.DataFrame(
	confusion_matrix(target_test,predictions),
	columns = ['Predict 0', 'Predict 1', 'Predict 2', 'Predict 3','Predict 4','Predict 5','Predict 6','Predict 7','Predict 8','Predict 9','Predict 10', 'Predict 11','Predict 12','Predict 13','Predict 14','Predict 15','Predict 16','Predict 17','Predict 18','Predict 19','Predict 20','Predict 21','Predict 22','Predict 23','Predict 24','Predict 25','Predict 26','Predict 27','Predict 28','Predict 29','Predict 30'],
	index = ['True 0', 'True 1', 'True 2', 'True 3','True 4','True 5','True 6','True 7','True 8','True 9','True 10','True 11','True 12','True 13','True 14','True 15','True 16','True 17','True 18','True 19','True 20','True 21','True 22','True 23','True 24','True 25','True 26','True 27','True 28','True 29','True 30']
)

print(confusion_matrix)

print(dict(zip(data.columns, random_forest_machine.feature_importances_)))







# ANN
CHURN_MODEL
We would be dealing with Churn Modeling i.e. we would be writing a Artificial Neural Network to find out reasons as to why and which customers are actually leaving the bank and their dependencies on one another.
This is a classification problem 0-1 classification(1 if Leaves 0 if customer stays).
We can use theano or tensorflow for this, but using these libraries require to write most of the code of ML from scratch, so I am gonna use "KERAS" which will enable me to write powerful Neural Networks with a few lines of code.
Keras runs on Theano and Tensorflow and you can think it of as a Sklearn for Deep Learning.

import os
os.getcwd()

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

churn_data = pd.read_csv("Churn_Modelling.csv")
churn_data.head(10)

print(churn_data.isnull().values.any())         ##  checking for missing values and capturing the count(NO MISSING VALUES)
churn_data.isnull().sum()

plt.scatter(x = churn_data["Balance"], y = churn_data["Exited"])
plt.show()

# Now lets check the class distributions

sns.countplot("Exited", data = churn_data)

# now let us check in the number of Percentage

Count_Exited_0 = len(churn_data[churn_data["Exited"]==0])      # customer staying are repersented by 0
Count_Exited_1 = len(churn_data[churn_data["Exited"]==1])                # customer leaving the bank represented by 1

Percentage_of_Count_Exited_0 = Count_Exited_0/(Count_Exited_0+Count_Exited_1)
print("percentage of Count_Exited_0 ",Count_Exited_0/100)

Percentage_of_Count_Exited_1= Count_Exited_1/(Count_Exited_0+Count_Exited_1)
print("percentage of Count_Exited_1 ",Count_Exited_1/100)

percentage of Count_Exited_0  79.63
percentage of Count_Exited_1  20.37


# Looking at the features we can see that row no.,surname will have no relation with a customer with leaving the bank
# so we drop them from X which contains the features Indexes from 3 to 12

X = churn_data.iloc[:, 3:13].values

#We store the Dependent value/predicted value in y by storing the 13th index in the variable y

y = churn_data.iloc[:, 13].values

#############################################Encoding categorical data########################################################

# Now we encode the string values in the features to numerical values.
# The only 2 values are "Gender" and "Region" which need to converted into numerical data.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

#creating label encoder object no. 1 to encode Geography name(index 1 in features)

X[:, 1] = labelencoder_X.fit_transform(X[:, 1])          #encoding Geography from string to just 3 no.s 0,1,2 respectively
print(X)

#creating label encoder object no. 2 to encode Gender name(index 2 in features)

labelencoder_X_1 = LabelEncoder()

 #encoding Gender from string to just 2 no.s 0,1(male,female) respectively

X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
print(X)


Now creating Dummy variables using :- "OneHotEncoder"

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
print(X)

onehotencoder_2 = OneHotEncoder(categorical_features = [2])
X = onehotencoder_2.fit_transform(X).toarray()
X = X[:, 2:]
print(X)

 ############################################### Splitting the dataset to train and test ##########################################
 
 from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

 ###################################### As class imbalance is there, we will upsample the minority class ############################
 
 def makeOverSamplesSMOTE(X,y):
 #input DataFrame
 #X →Independent Variable in DataFrame\
 #y →dependent Variable in Pandas DataFrame format
 from imblearn.over_sampling import SMOTE
 sm = SMOTE()
 X, y = sm.fit_sample(X, y)
 return X,y

################################################## feature scaling ###################################################

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

print(X_train)
print(X_test)
################################################# HERE OBSERVATION SIZE SHOULD CHANGE ###############################################

print(X_train.shape)
print(y_train.shape)      
print(X_test.shape)
print(y_test.shape)

# 'to_categorical' converts the class lebels to one-hot vectors. One-hot vector is nothing but dummifying in R.

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

############################################ NOW CHECK THE CHANGED OBSERVATION ######################################################

print(X_train.shape)
print(y_train.shape)      
print(X_test.shape)
print(y_test.shape)

############################################ PLOTTING #########################################################

plt.plot(X_train)
plt.plot(y_train)
plt.plot(X_test)
plt.plot(y_train)

####################################################### Now let's make the ANN ########################################################

# Importing the Keras libraries and packages

import keras

from keras.models import Sequential                #For building the Neural Network layer by layer
from keras.layers import Dense          #To randomly initialize the weights to small numbers close to 0(But not 0)


################################################### Initialising the ANN #################################################
#Defining each layer one by one

classifier_model = Sequential()

classifier_model.add(Dense(200, kernel_initializer='uniform', input_dim = 468, activation='relu'))

############################################### Adding the second hidden layer #################################################
classifier_model.add(Dense(20, kernel_initializer='uniform', activation='relu'))

######Sigmoid activation function is used whenever we need Probabilities of 2 categories or less(Similar to Logistic Regression)########

classifier_model.add(Dense(2, kernel_initializer = 'uniform', activation = 'sigmoid'))

################################################# compilation of model ###########################################################

from keras.optimizers import Adam
ada= Adam(lr=0.001)

classifier_model.compile(loss='binary_crossentropy',        # binary_CrossEntropy is the loss function. 
              optimizer=ada,                                    # Mention the optimizer
              metrics=['accuracy'])                               # Mention the metric to be printed while training
              
              
# Fitting the ANN to the Training set

classifier = classifier_model.fit(X_train, y_train, batch_size = 60, epochs = 15, validation_split = 0.3)


############################################## Part 3 - Making the predictions and evaluating the model #############################
################### Predicting the Test set results

y_pred = classifier_model.predict(X_test)
y_pred = (y_pred > 0.5)                  #if y_pred is larger than 0.5 it returns true(1) else false(2)
print(y_pred)
print(y_test.shape)
print(y_pred.shape)

################################################# ACCURACY ACCORDING TO NO OF EPOCHS ##################################################
train_acc = classifier.history['acc']
print(train_acc)

print("\n")

val_acc = classifier.history['val_acc']

################################################### CONFUSION MATRIX ########################################################
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(cm)
print(val_acc)

################################################### ACCURACY PERCENTAGE ###########################################################

accuracy=(1479+124)/2000           
print(accuracy)

THE ACCURACY RATE IS 80.15%, THIS WAS WITH UPSAMPLING TECHNIQUE.....FOR UNDERSAMPLE I GOT AROUND 85.7% ACCURACY....BUT THIS CASE DOES NOT IMPLIES THAT EVERYONE WILL GET THE SAME RESULTS, IT MIGHT DIFFER BY HOW YOU CLEAN YOUR DATA AND BALANCING THE CLASS...





# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY:
A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along. A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data.

## Neural Network Model:

![image](https://github.com/Bhuvaneshwari-2003/basic-nn-model/assets/94828604/99022ac8-bae7-417d-8294-032b954c6cc1)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:S.bhuvaneshwari
### Register Number:212221240010
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('sampledl').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})
dataset1.head()
X = dataset1[['Input']].values
y = dataset1[['Output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([Dense(units=3,input_shape=[1]), Dense(units=3), Dense(units=1)])
ai_brain.compile(optimizer="rmsprop", loss="mae")
ai_brain.fit(X_train1, y_train, epochs=2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information

![image](https://github.com/Bhuvaneshwari-2003/basic-nn-model/assets/94828604/ef6ec2b8-c11d-4fe4-b38b-6b84525456f9)


## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/Bhuvaneshwari-2003/basic-nn-model/assets/94828604/10e30976-6d04-4e5e-9f41-b077e35c94df)


### Test Data Root Mean Squared Error

![image](https://github.com/Bhuvaneshwari-2003/basic-nn-model/assets/94828604/3471e23c-f5c4-4224-b72e-fb50f1a6db71)


### New Sample Data Prediction

![image](https://github.com/Bhuvaneshwari-2003/basic-nn-model/assets/94828604/cc07289d-cec0-4bfa-ba0f-04d2573bbaf6)


## RESULT

Thus, a neural network regression model for the dataset is created and successfully executed.

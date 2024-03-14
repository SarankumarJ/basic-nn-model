# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

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
### Name: Sarankumar J
### Register Number: 212221230087
```python
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
worksheet = gc.open('Ex-01').sheet1
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
ai_brain = Sequential([
    Dense(6,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train,epochs = 2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information

![image](https://github.com/SarankumarJ/basic-nn-model/assets/94778101/02c97b46-6dc4-4631-8db4-02e26524502b)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/SarankumarJ/basic-nn-model/assets/94778101/1f251871-e99b-45aa-afb1-214cdfc4657b)



### Test Data Root Mean Squared Error

![image](https://github.com/SarankumarJ/basic-nn-model/assets/94778101/6e8ca00a-16fe-42db-8224-7e68f1630c0e)



### New Sample Data Prediction

![image](https://github.com/SarankumarJ/basic-nn-model/assets/94778101/96889a72-d786-49e6-a953-41506a93f6a0)


## RESULT

A Neural Network regression model for the given dataset has been developed Sucessfully.

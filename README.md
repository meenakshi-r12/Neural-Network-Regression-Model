# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

<img width="1134" height="647" alt="418446260-84093ee0-48a5-4bd2-b78d-5d8ee258d189" src="https://github.com/user-attachments/assets/3481f479-575c-4baf-b521-e6840d9f08d4" />


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
### Name:MEENAKSHI.R
### Register Number:212224220062
```
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history={'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x)) 
    x=self.relu(self.fc2(x))
    x=self.fc3(x)  
    return x


# Initialize the Model, Loss Function, and Optimizer

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
## Dataset Information

<img width="535" height="544" alt="418437786-6d0855b9-04e0-4547-9d3a-e9552e1e85bd" src="https://github.com/user-attachments/assets/d5323302-7192-4767-a05a-d9afbebea47b" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="815" height="616" alt="418442087-95b595fe-b9a6-4132-a8e3-1552e1d69fb2" src="https://github.com/user-attachments/assets/fd9660bd-0601-456e-b39d-b51e7bc43e20" />


### New Sample Data Prediction

<img width="847" height="159" alt="418442122-4491346b-8e48-4969-af1e-197c706fa4d7" src="https://github.com/user-attachments/assets/67071662-facd-4e37-825c-4600cac7c24f" />


## RESULT

Thus a neural network regression model for the dataset was developed successfully.

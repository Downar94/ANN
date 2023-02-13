import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

class NeuralNetwork:
    
    def __init__(self, epochs = 3000, eta = 0.5, learning_method = 'batch', batch_size = 15, is_bias = False, verbose = False):
        
        # the number of epochs
        self.epochs = epochs
        # learning rate
        self.eta = eta
        # learning method
        self.learning_method = learning_method
        # batch size if learning method is based on batch gradient descent
        self.batch_size = batch_size
        # adding biases
        self.is_bias = is_bias
        # is nn verbose
        self.verbose = verbose
     
    #function for data standarization
    def standarize_data(self, data):
        
        for column in range(data.shape[1]):
            #standard deviation
            std = data[:,column].std()
            mean = data[:,column].mean()
            standarized_data = (data[:,column] - mean) / std
        
        return standarized_data
    
    # sigmoid activation function
    def sigmoid(self, sum):
        
        return 1 / (1 + np.exp(-sum))
    
    # sigmoid activation function derivative
    def sigmoid_derivative(self, sigmoid):
        
        return sigmoid * (1 - sigmoid)
    
    # the process of training neural network
    def fit(self, inputs, outputs):
         
        # randomly generated weights between input and hidden layer
        self.weights0 = np.random.rand(inputs.shape[1], round((inputs.shape[1] + outputs.shape[1]) / 2))
        # randomly generated weights between hidden and output layer      
        self.weights1 = np.random.rand(round((inputs.shape[1] + outputs.shape[1]) / 2), outputs.shape[1])

        if self.is_bias == True:
            # randomly generated biases
            self.biases = np.random.rand(2)
        else:
            self.biases = np.array([0,0])
        
        if self.learning_method == 'stochastic':
            error = []
            for epoch in range(self.epochs):           
                for x, y_target in zip(inputs, outputs):
                    
                    # hidden layer weighted sum
                    hl_input_sum = x.dot(self.weights0) + self.biases[0]
                    # hidden layer activation function output
                    hl_output = self.sigmoid(hl_input_sum)
                    
                    h1_output_sum = np.dot(hl_output, self.weights1) + self.biases[1]
                    # predicted output
                    y_predict = self.sigmoid(h1_output_sum)
                    
                    # error output - the difference between target output and predicted output
                    error_output = sum(y_target - y_predict)
                    average_error = np.mean(abs(error_output))
          
                    if self.verbose == True and epoch % 1000 == 0:
                        print('For epoch: ' + str(epoch + 1) + 'Total Error = ' + str(average_error))
                        error.append(average_error)                    
                        
                    # delta error_total / delta y_predict 
                    delta_error_total_output = -(y_target - y_predict)
                    # delta y_predict / delta y 
                    delta_y_predict = (y_predict - y_predict ** 2)
                    # delta y / delta weights_output
                    delta_y = y_predict * hl_output
                    
                    # delta error_total / delta weight_output = (delta error_total / delta y_predict) * (delta y_predict / delta y) * (delta y / delta weights_output)
                    delta_output = delta_error_total_output * delta_y_predict * delta_y
                    
                    # delta error / delta hl_output = (delta error / delta y) * (delta y / delta hl_output)
                    delta_error_hl = (-(y_target - y_predict) * (y_predict - y_predict ** 2) * self.weights1).T
                    
                    # delta error_total / delta hl_output = sum(delta error / delta hl_output)
                    delta_error_total_hl = sum(delta_error_hl)
                    
                    # delta hl_output / delta hl_input_sum
                    delta_hl_output = self.sigmoid_derivative(hl_output)
                    # delta hl_input_sum / delta weights_input
                    delta_hl_input_sum = x.reshape(len(x),1)
                                      
                    # delta error_total / delta weight_input = (delta error_total / delta hl_output) * (delta hl_output / delta hl_input_sum) * (delta hl_input_sum / delta weights_input)
                    delta_hidden_layer = (delta_error_total_hl * delta_hl_output) * delta_hl_input_sum

                    #weights updating
                    self.weights1 = self.weights1 - (self.eta * delta_output)
                    self.weights0 = self.weights0 - (self.eta * delta_hidden_layer)
                    
        elif self.learning_method == 'batch':
              error = []
              for epoch in range(self.epochs): 
                  for i in range(0, len(outputs), self.batch_size):
                      
                      batch_inputs = inputs[i:(i + self.batch_size)]
                      batch_outputs = outputs[i:(i + self.batch_size)]
                  
                      hl_input_sum = np.dot(batch_inputs, self.weights0) + self.biases[0]
                      hl_output  = self.sigmoid(hl_input_sum)
        
                      h1_output_sum = np.dot(hl_output, self.weights1) + self.biases[1]
                      y_predict = self.sigmoid(h1_output_sum)
        
                      error_output = batch_outputs - y_predict
                      average_error = np.mean(abs(error_output))
          
                      if self.verbose == True and epoch % 1000 == 0:
                          if (i + self.batch_size) < len(outputs) :
                              row_end = i + self.batch_size
                          else :
                              row_end = len(outputs) - 1
                            
                          print('For epoch: ' + str(epoch + 1) + 'and rows from: ' + str(i) + ' to ' + str(row_end) +' Error: ' + str(average_error))
                          error.append(average_error)
        
                      delta_y_predict = self.sigmoid_derivative(y_predict)
                      delta_y = hl_output.T                                    
                      delta_error_predict = error_output * delta_y_predict 
          
                      delta_output_weight = delta_error_predict.dot(self.weights1.T)
                      
                      delta_hl_output = self.sigmoid_derivative(hl_output)
                      delta_hidden_layer = delta_output_weight * delta_hl_output
          
                      delta_output = delta_y.dot(delta_error_predict)
                      delta_hidden_layer = batch_inputs.T.dot(delta_output_weight * delta_hl_output)
                      
                      self.weights1 = self.weights1 + (self.eta * delta_output)
                      self.weights0 = self.weights0 + (self.eta * delta_hidden_layer)
                      
    # function for predicting output
    def predict(self, inputs, outputs):
        
        hl_input_sum = inputs.dot(self.weights0) + self.biases[0]
        hl_output = self.sigmoid(hl_input_sum)
        h1_output_sum = np.dot(hl_output, self.weights1) + self.biases[1]
        y_predict = self.sigmoid(h1_output_sum)
        
        error_output = outputs - y_predict
        accuracy = 1 - np.mean(abs(error_output))
        
        return y_predict, error_output, accuracy
  
def check_data(dataset):
    for column in dataset:
        print(str(column) + ': ' + str(dataset[column].unique()))
                             
# loading iris dataset                   
iris = datasets.load_iris()
inputs = iris.data
outputs = iris.target

# data standarization
scaler = StandardScaler()
scaler.fit(inputs)
inputs = scaler.transform(inputs)

# spliting dataset into test and training sets
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.4)  
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#encoding 3 outputs in training data
encoded_ytrain = []

for i in y_train:
    for x in i:
        if x == 0:
            encoded_ytrain.append([1,0,0])
        elif x == 1:
            encoded_ytrain.append([0,1,0])
        elif x == 2:
            encoded_ytrain.append([0,0,1])
            
encoded_ytrain = np.array(encoded_ytrain)

nn = NeuralNetwork(verbose = False)
nn.fit(X_train, encoded_ytrain)

standarized_inputs = nn.standarize_data(inputs)

#encoding 3 outputs in test data
encoded_ytest = []

for i in y_test:
    for x in i:
        if x == 0:
            encoded_ytest.append([1,0,0])
        elif x == 1:
            encoded_ytest.append([0,1,0])
        elif x == 2:
            encoded_ytest.append([0,0,1])
            
encoded_ytest = np.array(encoded_ytest)

pred_y, error, accuracy = nn.predict(X_test, encoded_ytest)

print('Iris prediction accuracy: ' + str(accuracy))


# loading credit risk data
credit_risk = pd.read_csv('german_credit_data.csv')
# checking the values in the data
'''check_data(credit_risk)'''
# replacing nan values with label 'unknown'
credit_risk.fillna('unknown', inplace=True)

# creating inputs and outputs
credit_risk_inputs = credit_risk.iloc[:,1:-1].values
credit_risk_outputs = credit_risk.iloc[:,-1].values

# encoding categorical values
label_encoder = LabelEncoder()

label_encoder.fit(credit_risk_inputs[:,1])
'''print("Sex column encoded classes: ")
print(label_encoder.classes_)'''
credit_risk_inputs[:,1] = label_encoder.transform(credit_risk_inputs[:,1])

label_encoder.fit(credit_risk_outputs)
'''print("Risk column encoded classes: ")
print(label_encoder.classes_)'''
credit_risk_outputs = label_encoder.transform(credit_risk_outputs)

column_transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [-1])], remainder = 'passthrough')
credit_risk_inputs = np.array(column_transformer.fit_transform(credit_risk_inputs))

column_transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [11])], remainder = 'passthrough')
credit_risk_inputs = np.array(column_transformer.fit_transform(credit_risk_inputs))

column_transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [-3])], remainder = 'passthrough')
credit_risk_inputs = np.array(column_transformer.fit_transform(credit_risk_inputs))

column_transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [-3])], remainder = 'passthrough')
credit_risk_inputs = np.array(column_transformer.fit_transform(credit_risk_inputs))

# data standarization
scaler = StandardScaler()
scaler.fit(credit_risk_inputs)
credit_risk_inputs = scaler.transform(credit_risk_inputs)

# spliting dataset into test and training sets
X_train, X_test, y_train, y_test = train_test_split(credit_risk_inputs, credit_risk_outputs, test_size = 0.4)  
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

nn = NeuralNetwork(verbose = False)
nn.fit(X_train, y_train)

pred_y, error, accuracy = nn.predict(X_test, y_test)

print('Credit risk prediction accuracy: ' + str(accuracy))


# loading heart failure risk data
heart_failure_risk = pd.read_csv('heart_failure_data.csv')
# checking the values in the data
'''check_data(heart_failure_risk)'''
# replacing nan values with label 'unknown'
heart_failure_risk.fillna('unknown', inplace=True)

# creating inputs and outputs
heart_failure_risk_inputs = heart_failure_risk.iloc[:,:-1].values
heart_failure_risk_outputs = heart_failure_risk.iloc[:,-1].values

# data standarization
scaler = StandardScaler()
scaler.fit(heart_failure_risk_inputs)
heart_failure_risk_inputs = scaler.transform(heart_failure_risk_inputs)

# spliting dataset into test and training sets
X_train, X_test, y_train, y_test = train_test_split(heart_failure_risk_inputs, heart_failure_risk_outputs, test_size = 0.4)  
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

nn = NeuralNetwork(verbose = False)
nn.fit(X_train, y_train)

pred_y, error, accuracy = nn.predict(X_test, y_test)

print('Heart failure risk prediction accuracy: ' + str(accuracy))


# loading travel insurance decision data
travel_insurance_decision = pd.read_csv('travel_insurance_data.csv')
# checking the values in the data
'''check_data(travel_insurance_decision)'''
# replacing nan values with label 'unknown'
travel_insurance_decision.fillna('unknown', inplace=True)

# creating inputs and outputs
travel_insurance_decision_inputs = travel_insurance_decision.iloc[:,1:-1].values
travel_insurance_decision_outputs = travel_insurance_decision.iloc[:,-1].values

# encoding categorical values
label_encoder.fit(travel_insurance_decision_inputs[:,1])
travel_insurance_decision_inputs[:, 1] = label_encoder.transform(travel_insurance_decision_inputs[:, 1])

label_encoder.fit(travel_insurance_decision_inputs[:,2])
travel_insurance_decision_inputs[:, 2] = label_encoder.transform(travel_insurance_decision_inputs[:, 2])

label_encoder.fit(travel_insurance_decision_inputs[:,-2])
travel_insurance_decision_inputs[:, -2] = label_encoder.transform(travel_insurance_decision_inputs[:, -2])

label_encoder.fit(travel_insurance_decision_inputs[:,-1])
travel_insurance_decision_inputs[:, -1] = label_encoder.transform(travel_insurance_decision_inputs[:, -1])

# data standarization
scaler = StandardScaler()
scaler.fit(travel_insurance_decision_inputs)
travel_insurance_decision_inputs = scaler.transform(travel_insurance_decision_inputs)

# spliting dataset into test and training sets
X_train, X_test, y_train, y_test = train_test_split(travel_insurance_decision_inputs, travel_insurance_decision_outputs, test_size = 0.4)  
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

nn = NeuralNetwork(verbose = False)
nn.fit(X_train, y_train)

pred_y, error, accuracy = nn.predict(X_test, y_test)

print('Travel insurance decision prediction accuracy: ' + str(accuracy))





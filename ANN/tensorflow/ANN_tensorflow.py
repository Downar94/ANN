import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import itertools

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split  
from sklearn.compose import ColumnTransformer
import sklearn.metrics

import pandas as pd

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    min_delta = 0,
    patience = 2,
    verbose = 0, 
    mode = 'auto',
    restore_best_weights = True
)

def define_cm_callback(log_directory_prefix, log_directory_postfix, cm_postfix):
    log_directory = log_directory_prefix + log_directory_postfix

    # Define a file writer variable for logging purposes
    confusion_matrix_file_writer = tf.summary.create_file_writer(log_directory + cm_postfix)
        
    # Defining the callbacks
    confusion_matrix_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_directory, histogram_freq=1, profile_batch=0)
    
    return tensorboard_callback, confusion_matrix_callback, confusion_matrix_file_writer


def log_confusion_matrix(epoch, logs):
    
    initial_y_predict = ann_model.predict(X_train)
    y_predict = np.argmax(initial_y_predict, axis=1)
    y_target = np.argmax(y_train, axis=1)

    # Confusion matrix evaluation
    confusion_matrix = sklearn.metrics.confusion_matrix(y_target, y_predict)
    
    # Confusion matrix to image summary
    confusion_matrix_figure = create_confusion_matrix_plot(confusion_matrix, class_labels = class_labels)
    confusion_matrix_image = convert_to_image(confusion_matrix_figure)

    with confusion_matrix_file_writer.as_default():
        tf.summary.image("Confusion Matrix", confusion_matrix_image, step=epoch)

def create_confusion_matrix_plot(confusion_matrix, class_labels):
    
    plot = plt.figure(figsize=(14, 14))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_markers = np.arange(len(class_labels))
    plt.xticks(tick_markers , class_labels, rotation=45)
    plt.yticks(tick_markers , class_labels)

    # Confusion matrix normalization
    confusion_matrix = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        color = "white" if confusion_matrix[i, j] > threshold else "black"
        plt.text(j, i, confusion_matrix[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('Target value')
    plt.xlabel('Predicted value')
    
    return plot

def convert_to_image(plot):
    
    # Save the buffer into memory
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')   
    # Closing the figure
    plt.close(plot)  
    buffer.seek(0)      
    # Creating TF image
    img = tf.image.decode_png(buffer.getvalue(), channels=4)  
    # Expand with the batch size
    img = tf.expand_dims(img, 0)
    
    return img

# loading iris dataset                   
iris = datasets.load_iris()
inputs = iris.data[:150]
outputs = iris.target[:150]
class_labels = iris.target_names

encoder =  LabelEncoder()
outputs = encoder.fit_transform(outputs)
outputs = pd.get_dummies(outputs).values

# spliting dataset into test and training sets
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 0.4)  

X_train = np.array(X_train).astype(np.float32)
X_test  = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test  = np.array(y_test).astype(np.float32)

# data standarization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ann_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = 'sigmoid'),
    tf.keras.layers.Dense(10, activation = 'sigmoid'),
    tf.keras.layers.Dense(3, activation = 'softmax'),   
])

ann_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
tensorboard_callback, confusion_matrix_callback, confusion_matrix_file_writer = define_cm_callback("Logs\\fit_iris\\", "iris_fit_n1", '/confusion_matrix')

ann_model.fit(X_train, y_train, batch_size = 36, epochs = 100, verbose = 1, callbacks = [tensorboard_callback, confusion_matrix_callback, early_stopping])

# evaluating the model for iris prediction
loss, accuracy = ann_model.evaluate(X_test, y_test, batch_size = 36)
print('---------------------')
print('iris prediction loss: ' + str(loss) + '\n' +'iris prediction accuracy: ' + str(accuracy))

# loading credit risk data
credit_risk = pd.read_csv('german_credit_data.csv')
# checking the values in the data
'''check_data(credit_risk)'''
# replacing nan values with label 'unknown'
credit_risk.fillna('unknown', inplace=True)

class_labels = credit_risk.Risk.unique()

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

ann_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = 'sigmoid'),
    tf.keras.layers.Dense(10, activation = 'sigmoid'),
    tf.keras.layers.Dense(1, activation = 'softmax'),   
])

ann_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
tensorboard_callback, confusion_matrix_callback, confusion_matrix_file_writer = define_cm_callback("Logs\\fit_credit_risk\\", "redit_risk_fit_n1", '/confusion_matrix')

ann_model.fit(X_train, y_train, batch_size = 26, epochs = 300, verbose = 1, callbacks = [tensorboard_callback, confusion_matrix_callback, early_stopping])

# evaluating the model for credit risk prediction
loss, accuracy = ann_model.evaluate(X_test, y_test, batch_size = 36)
print('---------------------')
print('credit risk prediction loss: ' + str(loss) + '\n' +'credit risk prediction accuracy: ' + str(accuracy))

# loading milk quality data
milk_quality = pd.read_csv('milk_quality_data.csv')
# checking the values in the data
'''check_data(milk_quality)'''
class_labels = milk_quality.Grade.unique()

# creating inputs and outputs
milk_quality_inputs = milk_quality.iloc[:,:-1].values
milk_quality_outputs = milk_quality.iloc[:,-1].values

# encoding categorical values
label_encoder = LabelEncoder()

label_encoder.fit(milk_quality_outputs)
'''print("milk quality encoded classes: ")
print(label_encoder.classes_)'''
milk_quality_outputs = label_encoder.transform(milk_quality_outputs)
milk_quality_outputs = pd.get_dummies(milk_quality_outputs).values

# data standarization
scaler = StandardScaler()
scaler.fit(milk_quality_inputs)
milk_quality_inputs = scaler.transform(milk_quality_inputs)

# spliting dataset into test and training sets
X_train, X_test, y_train, y_test = train_test_split(milk_quality_inputs, milk_quality_outputs, test_size = 0.4)  

ann_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = 'sigmoid'),
    tf.keras.layers.Dense(10, activation = 'sigmoid'),
    tf.keras.layers.Dense(3, activation = 'softmax'),   
])

ann_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
tensorboard_callback, confusion_matrix_callback, confusion_matrix_file_writer = define_cm_callback("Logs\\fit_milk_quality\\", "milk_quality_fit_n1", '/confusion_matrix')

ann_model.fit(X_train, y_train, batch_size = 36, epochs = 300, verbose = 1, callbacks = [tensorboard_callback, confusion_matrix_callback, early_stopping])

# evaluating the model for milk quality prediction
loss, accuracy = ann_model.evaluate(X_test, y_test, batch_size = 36)
print('---------------------')
print('milk quality prediction loss: ' + str(loss) + '\n' +'milk quality prediction accuracy: ' + str(accuracy))
# Loading the Tensorboard extension
'''%load_ext tensorboard
%tensorboard --logdir "logs/fit_milk_quality"'''

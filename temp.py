import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('D:/Projects/ml_project/random forest/trained_model.sav', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

y_pred = loaded_model.predict(input_data_reshaped)
print(y_pred)

if (y_pred[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

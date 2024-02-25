
import numpy as np
import streamlit as st

import pickle

import streamlit as st
import numpy as np
from sklearn.linear_model import Perceptron
import pickle

model = pickle.load(open('per_model-66130701724.sav', 'rb'))
st.title('Iris Flower Prediction App')
 
x1 = st.slider('Sepal Length', 0.0, 10.0, 0.1)
x2 = st.slider('Sepal Width', 0.0, 10.0, 0.1)
x3 = st.slider('Petal Length', 0.0, 10.0, 0.1)
x4 = st.slider('Petal Width', 0.0, 10.0, 0.1)
 
input_data = np.array([[x1, x2, x3, x4]]).reshape(1, -1)
 

predict = model.predict(input_data)
st.write('## Predict Result')
st.write('Species:', predict[0])

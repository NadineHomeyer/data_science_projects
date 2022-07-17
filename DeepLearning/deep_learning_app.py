import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas # https://github.com/andfanilo/streamlit-drawable-canvas

# Definze image size
SIZE = 192
# Load trained model
model = load_model('digit_classification/digit_classification_model.h5')
# Style the page
st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)

# Text on page
st.title('Digit Recognizer')
st.write('This digit recognizer is based on a convolution neuronal network model built based on the MNIST dataset.')
st.write('The MNIST training dataset from the Modified National Institute of Standards and Technology consists of')
st.write('60,000  28x20 pixel grayscale images with handwritten digits in the range 0-9 and the corresponding labels.')
st.write('The dataset is widely used for training deep learning models in the context of image recognition.')
st.write('The implemented model is a CNN-based, classification model determining the likelihood that a drawn.')
st.write('belongs is one of the ten digits 0-9. For the MNIST test set the model showed an accuracy (fraction')
st.write('of correctly classfied digits) of 99.1%. For details and source code see:')
st.write('')
st.write('The canvas used for providing the drawing functionality on this page originates from: https://github.com/andfanilo/streamlit-drawable-canvas ')

st.header('''Try it out!''')
st.markdown('''Draw a digit in the canvas below and then click on 'Predict' ''')

# Canvas for drawing digits
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')

# Draw rescaled canvas that serves as input for the prediction
# Rescaling is necessary in on order to ensure consisteny with trained model
if canvas_result.image_data is not None:
    # Resize in order to match the dimensions of the array used for training
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    # Rescale image for display
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) 
    st.write('Rescaled model input')
    st.image(rescaled)

if st.button('Predict'):
    # Convert BGR colored image to grayscale image
    # Necessary since the MNIST dataset used for
    # model training contains only grayscale images
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform prediction
    y_pred = model.predict(test_x.reshape(1, 28, 28))
    # Show predicted digit
    st.write(f'Predicted digit: {np.argmax(y_pred, axis=1)[0]}')
    
    # Plot bar chart with likeliness distribution
    Y_pred = pd.DataFrame(y_pred.T)
    barchart_figure, ax = plt.subplots()
    ax.bar(Y_pred.index, Y_pred[0], color="skyblue")
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    ax.xaxis.set_ticks([0,1,2,3,4,5,6,7,8,9]) 
    plt.title("Predicted likeliness")
    st.pyplot(fig=barchart_figure)
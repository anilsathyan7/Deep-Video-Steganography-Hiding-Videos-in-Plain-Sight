import numpy as np
import random
import keras
import glob
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from pathlib import Path

'''
Evaluates the trained model on a new dataset
Uses Mean Square Error as evaluation metric
'''

# Path for evaluation dataset
EVAL_PATH = Path(sys.argv[1]).resolve()
BATCH_SIZE=1
TEST_NUM=len(glob.glob(str(EVAL_PATH)))

# Normalize input for evaluation
def normalize_batch(imgs):

    return (imgs -  np.array([0.485, 0.456, 0.406])) /np.array([0.229, 0.224, 0.225])

# Denormalize output for prediction
def denormalize_batch(imgs,should_clip=True):

    imgs= (imgs * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    
    if should_clip:
        imgs= np.clip(imgs,0,1)
    return imgs

# Create a data generator for evaluation
test_imgen = ImageDataGenerator(rescale = 1./255)

# Compute L2 loss(MSE) for secret image
def custom_loss_1(secret,secret_pred):        
    secret_mse = keras.losses.mean_squared_error(secret,secret_pred)
    return secret_mse

# Compute L2 loss(MSE) for cover image
def custom_loss_2(cover,cover_pred):
    cover_mse = keras.losses.mean_squared_error(cover,cover_pred)
    return cover_mse 

# Load the trained model
model=load_model(sys.argv[2], custom_objects={'custom_loss_1': custom_loss_1, 'custom_loss_2': custom_loss_2} )

# Custom data generator
def generate_generator_multiple(generator, direc):
    genX1 = generator.flow_from_directory(direc, target_size=(224,224), batch_size = 12, shuffle=True, seed=3, class_mode=None)
    genX2 = generator.flow_from_directory( direc, target_size=(224, 224), batch_size = 12, shuffle=True, seed=8, class_mode=None)
    
    while True:
            X1i = normalize_batch(genX1.next()) 
            X2i = normalize_batch(genX2.next()) 

            yield( {'secret': X1i, 'cover': X2i}, {'hide_conv_f': X2i, 'revl_conv_f': X1i})  #Yield both images and their mutual label      
    
# Load data using generator 
testgenerator=generate_generator_multiple(test_imgen, direc=str(EVAL_PATH.parent)) 

# Evaluates the model using data generator
score = model.evaluate_generator(testgenerator,steps=TEST_NUM/BATCH_SIZE, verbose=0)

# Print mean square error
print('Mean square error:', score[0])


'''
Test the model on a  random pair of images (test)
Plots the input and output for verification
'''

# Perform prediction for single input
def predict(source,cover):

   # Normalize inputs
   secret=np.array(source/255.0)
   cover=np.array(cover/255.0)

   # Predict output
   coverout, secretout=model.predict([normalize_batch(np.reshape(secret,(1,224,224,3))),normalize_batch(np.reshape(cover,(1,224,224,3)))])

   # Postprocess output cover image
   coverout = denormalize_batch(coverout)
   coverout=np.squeeze(coverout)*255.0
   coverout=np.uint8(coverout)

   # Postprocess output secret image    
   secretout=denormalize_batch(secretout)
   secretout=np.squeeze(secretout)*255.0
   secretout=np.uint8(secretout)

   # Plot output images
   fig_out, ax_out = plt.subplots(1,2, figsize=(10,10))
   fig_out.suptitle('Outputs')
   ax_out[0].title.set_text("Secret output")
   ax_out[0].imshow(secretout)
   ax_out[1].title.set_text("Cover output")
   ax_out[1].imshow(coverout)
   
# Load random test image pairs
sec,cov=random.sample(os.listdir(str(EVAL_PATH)),k=2)

source=np.array(Image.open(str(EVAL_PATH)+'/'+sec))
cover=np.array(Image.open(str(EVAL_PATH)+'/'+cov))

# Plot input images
fig_in, ax_in = plt.subplots(1,2, figsize=(10,10))
fig_in.suptitle('Inputs')
ax_in[0].title.set_text("Secret input")
ax_in[0].imshow(source)
ax_in[1].title.set_text("Cover input")
ax_in[1].imshow(cover)

# Perform prediction
predict(source,cover)

# Sample run: python eval.py dataset/eval_data checkpoints/steg_model-06-0.03.hdf5

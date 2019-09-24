import numpy as np
import keras
import sys
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from random import randint
import imageio
from skimage.util.shape import view_as_blocks
%matplotlib inline

'''
Test the model on sample images (unseen)
Plot the input and output images
'''

# Load test images
test_images=np.load(sys.argv[1])

# Load model
model=load_model(sys.argv[2],compile=False)

# Normalize inputs
def normalize_batch(imgs):
    '''Performs channel-wise z-score normalization'''

    return (imgs -  np.array([0.485, 0.456, 0.406])) /np.array([0.229, 0.224, 0.225])

# Denormalize outputs
def denormalize_batch(imgs,should_clip=True):
    imgs= (imgs * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    
    if should_clip:
        imgs= np.clip(imgs,0,1)
    return imgs

# Load images as batch (batch size -4)
secretin = test_images[np.random.choice(len(test_images), size=4, replace=False)]
coverin = test_images[np.random.choice(len(test_images), size=4, replace=False)]

# Perform batch prediction        
coverout, secretout=model.predict([normalize_batch(secretin),normalize_batch(coverin)])

# Postprocess cover output
coverout = denormalize_batch(coverout)
coverout=np.squeeze(coverout)*255.0
coverout=np.uint8(coverout)

# Postprocess secret output       
secretout=denormalize_batch(secretout)
secretout=np.squeeze(secretout)*255.0
secretout=np.uint8(secretout)
        
# Convert images to UINT8 format (0-255)       
coverin=np.uint8(np.squeeze(coverin*255.0))
secretin=np.uint8(np.squeeze(secretin*255.0))

# Plot the images
def plot(im, title):
    fig = plt.figure(figsize=(20, 20)) 

    for i in range(4):
        sub = fig.add_subplot(1, 4, i + 1)
        sub.title.set_text(title + " " + str(i+1))
        sub.imshow(im[i,:,:,:])

# Plot secret input and output
plot(secretin, "Secret Input")
plot(secretout, "Secret Output")

# Plot cover input and output
plot(coverin, "Cover Input")
plot(coverout, "Cover Output")


# Sample run: python test.py test/testdata.npy checkpoints/steg_model-06-0.03.hdf5

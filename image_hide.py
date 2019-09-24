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
import argparse
from skimage.util.shape import view_as_blocks


# Construct argument parser
parser = argparse.ArgumentParser(description='Use block shuffle')        
parser.add_argument('--shuffle', action='store_true', default=False)
parser.add_argument("--model", required=True, help="path to trained model")
parser.add_argument("--secret_image", required=True, help="path to secret image")
parser.add_argument("--cover_image", required=True, help="path to cover image")
args= vars(parser.parse_args())

'''
Hides secret image inside cover image
Input: Secret Image, Cover Image, Hide Model
Output: Container Image
'''

# Load the model
model_hide=load_model(args['model'],compile=False)

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

# Custom block shuffling
def shuffle(im, inverse = False):
  
  # Configure block size, rows and columns
  blk_size=56
  rows=np.uint8(img.shape[0]/blk_size)
  cols=np.uint8(img.shape[1]/blk_size)

  # Create a block view on image
  img_blks=view_as_blocks(im,block_shape=(blk_size,blk_size,3)).squeeze()
  img_shuff=np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

  # Secret key maps
  map={0:2, 1:0, 2:3, 3:1}
  inv_map = {v: k for k, v in map.items()}

  # Perform block shuffling
  for i in range(0,rows):
    for j in range(0,cols):
     x,y = i*blk_size, j*blk_size
     if(inverse):
      img_shuff[x:x+blk_size, y:y+blk_size] = img_blks[inv_map[i],inv_map[j]]
     else:
      img_shuff[x:x+blk_size, y:y+blk_size] = img_blks[map[i],map[j]]
      
  return img_shuff

# Normalize input images [float: 0-1]
secretin = np.array(Image.open(args['secret_image']).convert('RGB')).reshape(1,224,224,3)/255.0
coverin = np.array(Image.open(args['cover_image']).convert('RGB')).reshape(1,224,224,3)/255.0

# Shuffle secret input
if(args["shuffle"]==True):
    secretin[0]=shuffle(secretin[0], inverse = False)

# Predict the output       
coverout=model_hide.predict([normalize_batch(secretin),normalize_batch(coverin)])

# Postprocess the output
coverout = denormalize_batch(coverout)
coverout=np.squeeze(coverout)*255.0
coverout=np.uint8(coverout)

# Save and plot stego-image output (container image)
imageio.imsave('test/container.png',coverout)
plt.imshow(coverout)

'''
Sample run :-
# Without shuffle
python image_hide.py --model models/hide.h5 --secret_image test/cover.png --cover_image test/secret.png
# With shuffle
python image_hide.py --model models/hide.h5 --secret_image test/cover.png  --cover_image test/cover.png  --shuffle 
'''

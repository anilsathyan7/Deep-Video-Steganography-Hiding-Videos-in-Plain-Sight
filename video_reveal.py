import os
import numpy as np
import keras
import cv2
import sys
import math
from keras.models import Model
from keras.models import load_model
from PIL import Image
import argparse
from scipy.misc import imsave
from skimage.util.shape import view_as_blocks

# Construct argument parser
parser = argparse.ArgumentParser(description='Use block shuffle')        
parser.add_argument('--shuffle', action='store_true', default=False)
parser.add_argument("--model", required=True, help="path to trained model")
parser.add_argument("--container_video", required=True, help="path to container video")
args= vars(parser.parse_args())

# Normalize input images
def normalize_batch(imgs):
    return (imgs -  np.array([0.485, 0.456, 0.406])) /np.array([0.229, 0.224, 0.225])

# Denormalize output images                                                        
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

# Update progress bar
def update_progress(current_frame, total_frames):
    progress=math.ceil((current_frame/total_frames)*100)
    sys.stdout.write('\rProgress: [{0}] {1}%'.format('>'*math.ceil(progress/10), progress))
    
# Load the trained model
model=load_model(args["model"],compile=False)

# Input videos - Container Video
vidcap = cv2.VideoCapture(args["container_video"])

# Start video decoding
print("\nDecoding video ...\n")

# Total secret video frames
num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames in container video:", num_frames)


# Video writer for output
secret_outvid = cv2.VideoWriter('results/secret_outvid_300.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (300,300))

# Temporary buffers for batching
cover_batch=[]
frame = 0

# Process frames as batches
while True:

        # Read frames sequentially
        (success, cover) = vidcap.read()

        if not (success):
            break       
 
        # Preprocess frames
        cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)       

        # Append frames to buffer    
        cover_batch.append(cover)            
        #print("Processing batch: ", batch)
        frame = frame + 1

        # Perform batch prediction
        if frame % 4 == 0  : 
            
            # Convert images to float type 
            cover_batch = np.float32(cover_batch)/255.0
            
            # Save image for testing
            imsave("test.png",cover_batch[0])             

            # Predict outputs
            secretout=model.predict([normalize_batch(cover_batch)])
       
            # Postprocess secret image output
            secretout=denormalize_batch(secretout)
            secretout=np.squeeze(secretout)*255.0
            secretout=np.uint8(secretout)

             # Save secret output video
            for i in range(0,4):
               #imsave("seretout.png",frame)
               if(args["shuffle"]==True):
                 secretout[i]=shuffle(secretout[i], inverse = True)
               secret_outvid.write(cv2.resize(secretout[i][..., ::-1], (300,300), interpolation=cv2.INTER_CUBIC))
            
            # Empty temporary buffers
            cover_batch=[]

            # Update progress
            update_progress(frame, num_frames)

# Finish video decoding
print("\n\nSuccessfully decoded video !!!\n")
              
# Close video capturers
vidcap.release()
cv2.destroyAllWindows()

'''
Sample run :-
# Without shuffle
python video_reveal.py --model models/reveal.h5 --container_video results/cover_outvid_224.avi
# With shuffle
python video_reveal.py --model models/reveal.h5 --container_video results/cover_outvid_224.avi --shuffle 
'''        

import numpy as np
import argparse
import sys
import math
from PIL import ImageFilter, Image
import cv2

'''
Enhance video by applying denoise and sharpen filters
'''

# Construct argument parser
parser = argparse.ArgumentParser(description='Use block shuffle')        
parser.add_argument('--denoise', action='store_true', default=False)
parser.add_argument('--sharpen', action='store_true', default=False)
parser.add_argument("--input_video", required=True, help="path to input video")
args= vars(parser.parse_args())

# Start video enhancement
print('\nEnhancing video ...\n')

# Update progress bar
def update_progress(current_frame, total_frames):
    progress=math.ceil((current_frame/total_frames)*100)
    sys.stdout.write('\rProgress: [{0}] {1}%'.format('>'*math.ceil(progress/10), progress))

# Open the input video
vidcap = cv2.VideoCapture(args['input_video'])

# Total input video frames
num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames in input video:", num_frames)

# Initialize the video writer
enhanced_video = cv2.VideoWriter('results/enhanced_secret_300.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (300,300))

# Initialize the frame buffer
frames=[]

# Load the frames to buffer
while vidcap.isOpened():
  success,image = vidcap.read()
  if success:
     frames.append(image)
  else:
      break

# Set up the start frame index
start_frame=5

# Enhance and save video frame-by-frame
for i in range(start_frame,len(frames)-(start_frame+1)):
   output=frames[i]
   if(args["denoise"]==True):
      output = cv2.fastNlMeansDenoisingColoredMulti(frames, i, 11)
   if(args["sharpen"]==True):
       output = np.array( Image.fromarray(output).filter(ImageFilter.DETAIL) )
   enhanced_video.write(output)
   update_progress(i, num_frames-(start_frame+1))

# Finish video enhancement
print('\n\nSuccessfully enhanced video !!!\n')

'''
Sample run:-
python enhance.py --input_video results/secret_outvid_300.avi --denoise --sharpen
'''

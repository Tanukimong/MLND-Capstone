import numpy as np
import cv2
from PIL.Image import fromarray, BILINEAR
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import model_from_json
import sys
import warnings
warnings.filterwarnings(action='ignore') # 귀찮은 경고 감추기

# Load Keras model
json_file = open('tanuki_model.json', 'r')
json_model = json_file.read()
json_file.close()
model = model_from_json(json_model)
model.load_weights('tanuki_model.h5')

scaler = 6
resized_shape = (1640//scaler, 590//scaler)

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    # Get image ready for feeding into model
    small_img = fromarray(image).resize(resized_shape)
    small_img = np.array(small_img,dtype="uint8")
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    lane_drawn = lane_drawn.astype("uint8")

    # Re-size to match the original image
    lane_image = fromarray(lane_drawn)
    lane_image = lane_image.resize((1280, 720),BILINEAR)
    lane_image = np.asarray(lane_image,dtype="uint8")

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result

lanes = Lanes()

# Where to save the output video
vid_output = sys.argv[2]

# Location of the input video
clip1 = VideoFileClip(sys.argv[1])

vid_clip = clip1.fl_image(road_lines)
vid_clip.write_videofile(vid_output, audio=False)

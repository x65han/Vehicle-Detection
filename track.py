# import statement
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import time, math, glob, cv2
import numpy as np
from helper import Helper

# Initialize Helper
helper = Helper()
helper.train_classifier()

# Define Pipeline
def pipeline(image):
    rectangles = []
    # Windows
    y_starts = [400, 416, 400, 432, 400, 432, 400, 464]
    y_stops = [464, 480, 496, 528, 528, 560, 506, 660]
    scales = [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.5, 3.5]

    for y_start, y_stop, scale in zip(y_starts, y_stops, scales):
        rectangles.append(helper.find_cars(image, y_start, y_stop, scale))

    rectangles = [item for sublist in rectangles for item in sublist] 
    
    heatmap_image = np.zeros_like(image[:,:,0])
    heatmap_image = helper.add_heat(heatmap_image, rectangles)
    heatmap_image = helper.apply_threshold(heatmap_image, 3)
    labels = label(heatmap_image)
    draw_image, rects = helper.draw_labeled_bboxes(np.copy(image), labels)
    return draw_image

# Process Image
input_images = glob.glob('./assets/inputs/*.jpg')
output_images = []
output_titles = []

for index, image_path in enumerate(input_images):
    clean_path = image_path.split('/')[-1]
    image = mpimg.imread(image_path)
    output_image = pipeline(image)
    output_images.append(output_image)
    output_titles.append(clean_path)
    mpimg.imsave('./outputs/' + clean_path, output_image)

# Process Video
video_paths = glob.glob('./assets/inputs/*.mp4')

for video_path in video_paths:
    clean_name = video_path.split('/')[-1]
    print(clean_name)
    project_video = VideoFileClip(video_path)
    output_video = project_video.fl_image(pipeline)
    output_video.write_videofile("./outputs/" + clean_name, audio=False)

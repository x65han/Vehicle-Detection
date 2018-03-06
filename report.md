# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
---
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I created the following function to utilize `skimage.feature.hog` to extract HOG features

```python
from skimage.feature import hog

def convert_to_hog(image, orientations, pixel_per_cell, cell_per_block, visualise=False, feature_vector=True):
    return hog(image, orientations=orientations,
               pixels_per_cell=(pixel_per_cell, pixel_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=False, visualise=visualise,
               feature_vector=feature_vector)
```

Then I use the following parameter to extract HOG image on `a car & a non-car image`.

```python
car_image = mpimg.imread(car_images[50])
_, car_destination = convert_to_hog(car_image[:, :, 2], 9, 8, 8, visualise=True, feature_vector=True)
non_car_image = mpimg.imread(non_car_images[50])
_, non_car_destination = convert_to_hog(non_car_image[:, :, 2], 9, 8, 8, visualise=True, feature_vector=True)
```

<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/hog.jpg?raw=true" /></div>

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and ...
<div align="center"><img  width="1000%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/explore.jpg?raw=true" /></div>

- Considering a combination of
    - hog extraction speed
    - classifier accuracy
- There are a couple great options to choose from, but I ended up using the following parameters.

```python
self.orientations = 11
self.pixel_per_cell = 16
self.cell_per_block = 2
self.colorspace = "YUV" # OR RGB, HSV, LUV, HLS, YUV, YCrCb
self.hog_channel = 'ALL' # OR 0, 1, 2, or "ALL"
self.svc = None
self.feature_vector = True
self.visualise = False
```


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

- I trained a linear SVM using LinearSVC with no other parameters...
- To go through all the images, my classifier takes 0.6 seconds to achieve 97.6% accuracy.

```python
# Define the labels vector
X = np.vstack((car_features, non_car_features)).astype(np.float64)  
y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

# Split up data into randomized training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=np.random.randint(0, 999))

# Use a SVM Classifier (Support Vector Machine)
svc = LinearSVC()
t1 = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
# Metrics
training_time = (round(t2 - t1, 1))
accuracy = round(svc.score(X_test, y_test) * 100, 1)
print("Classifier takes", training_time, "seconds to achieve {}% accuracy.".format(accuracy))
print('Input data has shape {}'.format(X_train[0].shape))
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to have a combination of small to large windows to capture all possible car locations.
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/window_1.jpg?raw=true" /></div>
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/window_2.jpg?raw=true" /></div>
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/window_3.jpg?raw=true" /></div>
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/window_4.jpg?raw=true" /></div>


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I implemented `find_cars` function and `draw_boxes` function to achieve the following result...
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/test1_output.jpg?raw=true" /></div>

Then with my `window search` function, I identify all the potential car images
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/window_mutiple.jpg?raw=true" /></div>

With heatmap, I eliminated all the `false positives` & ended up...
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/output_sample.jpg?raw=true" /></div>

Then I processed all the images
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/output_images.jpg?raw=true" /></div>

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/x65han/Vehicle-Detection/blob/master/outputs/project_video.mp4?raw=true)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I defined the following heatmap function.
```python
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
```

I obtained the following heatmap image
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/heatmap.jpg?raw=true" /></div>

After Applying a threshold of 3, I obtained the follow clean image to avoid `false positives`.
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/heatmap_threshold.jpg?raw=true" /></div>

Then Apply SciPy Labels to heatmap threshold image...
<div align="center"><img  width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/heatmap_scipy.jpg?raw=true" /></div>

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I find this project very clear and straight-forward. I knew exactly the steps I needed to take and why these steps are necessary even before implementing my code. I find scikit-image and scikit-learn to be very powerful.

The pipeline doesn't really fail on any case, but sometimes produces inconsistency and constantly changing window frames. This has to do with window searching. I can spend more time on window searching to provide better drawing frames, but sacarfices on time performance. So it's really a compromise. As long as the vehicle is detected, everything is good. But I wish a better algorithm is taught (where it can exactly identify every contour of the vehicle).

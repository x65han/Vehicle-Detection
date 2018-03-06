# import statement
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import time, math, glob, cv2
import numpy as np

class Helper():
    def __init__(self):
        # Configuration Constants
        self.orientations = 11
        self.pixel_per_cell = 16
        self.cell_per_block = 2
        self.colorspace = "YUV" # OR RGB, HSV, LUV, HLS, YUV, YCrCb
        self.hog_channel = 'ALL' # OR 0, 1, 2, or "ALL"
        self.svc = None
        self.feature_vector = True
        self.visualise = False
        # load data set
        self.car_images = glob.glob('./training_data/vehicles/**/*.png')
        self.total_car_images = len(self.car_images)
        self.non_car_images = glob.glob('./training_data/non-vehicles/**/*.png')
        self.total_non_car_images = len(self.non_car_images)

    def train_classifier(self):
        # Variable Factors
        t1 = time.time()
        car_features = self.extract_features(self.car_images)
        non_car_features = self.extract_features(self.non_car_images)
        t2 = time.time()

        print('It takes {} seconds to run'.format(round(t2 - t1), 2))
        # Define the labels vector
        X = np.vstack((car_features, non_car_features)).astype(np.float64)  
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

        # Split up data into randomized training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(0, 999))

        # Use a SVM Classifier (Support Vector Machine)
        self.svc = LinearSVC()
        t1 = time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        # Metrics
        training_time = (round(t2 - t1, 1))
        accuracy = round(self.svc.score(X_test, y_test) * 100, 1)
        print("Classifier takes", training_time, "seconds to achieve {}% accuracy.".format(accuracy))
        print('Input data has shape {}'.format(X_train[0].shape))

    def convert_to_hog(self, image, visualise=False, feature_vector=True):
        return hog(image, orientations=self.orientations,
               pixels_per_cell=(self.pixel_per_cell, self.pixel_per_cell),
               cells_per_block=(self.cell_per_block, self.cell_per_block),
               transform_sqrt=False, visualise=visualise,
               feature_vector=feature_vector)

    def extract_features(self, images, colorspace='RGB'):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in images:
            # Read in each one by one
            image = mpimage.imread(file)
            # apply color conversion if other than 'RGB'
            if colorspace != 'RGB':
                if colorspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif colorspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif colorspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif colorspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif colorspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)      

            # Call convert_to_hog() with visualise=False, feature_vector=True
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(self.convert_to_hog(feature_image[:,:,channel]))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = self.convert_to_hog(feature_image[:, :, self.hog_channel])
            # Append the new feature vector to the features list
            features.append(hog_features)
        # Return list of feature vectors
        return features

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, image, ystart, ystop, scale):
        
        # array of rectangles where cars were detected
        rectangles = []
        
        image = image.astype(np.float32) / 255
        
        image_tosearch = image[ystart:ystop,:,:]

        # apply color conversion if other than 'RGB'
        if self.colorspace != 'RGB':
            if self.colorspace == 'HSV':
                ctrans_tosearch = cv2.cvtColor(image_tosearch, cv2.COLOR_RGB2HSV)
            elif self.colorspace == 'LUV':
                ctrans_tosearch = cv2.cvtColor(image_tosearch, cv2.COLOR_RGB2LUV)
            elif self.colorspace == 'HLS':
                ctrans_tosearch = cv2.cvtColor(image_tosearch, cv2.COLOR_RGB2HLS)
            elif self.colorspace == 'YUV':
                ctrans_tosearch = cv2.cvtColor(image_tosearch, cv2.COLOR_RGB2YUV)
            elif self.colorspace == 'YCrCb':
                ctrans_tosearch = cv2.cvtColor(image_tosearch, cv2.COLOR_RGB2YCrCb)
        else: ctrans_tosearch = np.copy(image)   
        
        # rescale image if other than 1.0 scale
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
        # select colorspace channel for HOG 
        if self.hog_channel == 'ALL':
            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]
        else: 
            ch1 = ctrans_tosearch[:, :, self.hog_channel]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pixel_per_cell)+1  #-1
        nyblocks = (ch1.shape[0] // self.pixel_per_cell)+1  #-1 
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pixel_per_cell)-1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = self.convert_to_hog(ch1, feature_vector=False)
        if self.hog_channel == 'ALL':
            hog2 = self.convert_to_hog(ch2, feature_vector=False)
            hog3 = self.convert_to_hog(ch3, feature_vector=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                if self.hog_channel == 'ALL':
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog_feat1

                xleft = xpos*self.pixel_per_cell
                ytop = ypos*self.pixel_per_cell
                
                ################ ONLY FOR BIN_SPATIAL AND COLOR_HIST ################

                # Extract the image patch
                #subimage = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
                # Get color features
                #spatial_features = bin_spatial(subimage, size=spatial_size)
                #hist_features = color_hist(subimage, nbins=hist_bins)

                # Scale features and make a prediction
                #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                #test_prediction = svc.predict(test_features)
                
                ######################################################################
                
                hog_features = np.array([hog_features])
                
                test_prediction = self.svc.predict(hog_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                    
        return rectangles

    def draw_boxes(self, image, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(image)
        random_color = False
        # Iterate through the bounding boxes
        for bbox in bboxes:
            if color == 'random' or random_color:
                color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                random_color = True
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, image, labels):
        # Iterate through all detected cars
        rectangles = []
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            rectangles.append(bbox)
            # Draw the box on the image
            cv2.rectangle(image, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image and final rectangleangles
        return image, rectangles
        
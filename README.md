# Vehicle Detection

<img align="right" src="https://cdn.instructables.com/FNN/AZF7/IG2HFKH0/FNNAZF7IG2HFKH0.LARGE.jpg" width="250px" />
<img src="https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg"/>

- **Vehicle Detection + Machine Learning**
- `OpenCV` + `Python` + `scikits-learn` + `scikit-image`
<br>
<hr>
<div align="center">
    <img src="https://kaggle2.blob.core.windows.net/competitions/kaggle/3428/media/scikit-learn-logo.png" width="300px" />
    &emsp;&emsp;&emsp;&emsp;
    <img src="http://scikit-image.org/_static/img/logo.png" width="300px" />
</div>

<hr>

<div align="center"><b>Lane Detection</b>&emsp;|&emsp;<a href="https://github.com/x65han/Vehicle-Detection/blob/master/outputs/project_video.mp4?raw=true">Full Video</a>&emsp;|&emsp;<a href="https://github.com/x65han/Vehicle-Detection/blob/master/report.md">Full Report</a></div><br>
<div align="center"><img width="60%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/sample.gif?raw=true"/></div><br>


## Overview

### Machine Learning Classifier

- I trained a `linear SVM` using `LinearSVC` from **sklearn**
```python 
    from sklearn.svm import LinearSVC
```
- To go through all the training images, my classifier takes **0.6 seconds** to achieve **97.6% accuracy**.
- Here is a sample batch of my training images provided by [GTI Vehicle Image Database](http://www.gti.ssr.upm.es/data/Vehicle_database.html)
- From this awesome vehicle image database, there are two types of data: `car images` & `non car images`
<div align="center"><img width="80%" src="https://github.com/x65han/Vehicle-Detection/blob/master/assets/report/cars.jpg?raw=true"/></div><br>

### Computer Vision Pipeline

<div align="center"><b>Sample Input Image</b></div>
<div align="center"><img width="60%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/illos-home.jpg?raw=true" /></div>
<hr>
<div align="center"><b>Hooray</b></div>
<div align="center"><img width="60%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/illos-home.jpg?raw=true" /></div>
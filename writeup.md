**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
# [image1]: ./examples/car_not_car.png
# [image2]: ./examples/HOG_example.jpg
[image1]: ./output_images/raw_two_classes.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/hog_features_orient.png
[image4]: ./output_images/hog_features_ppc.png
[image5]: ./output_images/hog_features_cpb.png

[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG features are used in training and evaluation time. I extracted the HOG features by using the function `get_hog_features()` in `utils.py` at line 96. In that function, `skimage.feature.hog` is called. Code where I extracted HOG features from the training data are found in the `3.1 HOG Feature Extraction` section of `TrainModel.ipynb`.

I started by loading all the `vehicle` and `non-vehicle` images. Given dataset are classified into "non-vehicle" and "vehicle" classes. Each class has some subclasses. In this project, these subclasses are not necessary, so that those subclasses are combined into "vehicle" and "non-vehicle" classes. Before combining each class.
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
Figure 1: Examples of vehicle (Left) and non-vehicle (Right) class images.

I applied the `get_hog_features()` which is described above to each color channels of loaded images. After some trial-and-error, I decided to use `YCrCb` color space for this feature extraction, since I thought that each channel of that color space well captures different features of images. Examples of visualized HOG features are show in Figure 2.

![alt text][image2]
Figure 2: Vehicle and Non-vehicle images in the 'YCrCb' color space and visualized HOG feature of corresponding channel are shown. Parameters used for Hog features extraction are (orient=11, pix_per_cell=8, cell_per_block=2).

####2. Explain how you settled on your final choice of HOG parameters.

I compared raw image and visualized hog features with different parameters and selected parameters which seem capture the image characteristics well.
Visualized examples of HOG features with different parameters are shown in Figure 3.

![alt text][image3]
![alt text][image4]
![alt text][image5]

In this Figure, HOG features with `orient=3, 5, 11, 21` are shown. With a few orient number `orient <= 5` the visualized feature too simple to characterize the vehicle shape, while the features with `orient = 11` express the vehicle shape well.
I also compared features with `orient = 11` and `orient = 21`, and I couldn't found any qualitative difference in two features. Therefore I thought that the `orient = 11` is sufficient.

Effects of `pix_per_cell` parameter are also given in Figure 3 in the middle row. When the small `pix_per_cell` such as `pix_per_cell=4` the HOG feature depict fine structure for each position, but the orientational variation of features of a single position is limited. On the other hand, features with large `pix_per_cell` such as `pix_per_cell=16, 32` the orientational characteristics expresses fine but positional characteristics are poorly expressed. I thought that the features with `pix_per_cell=8` balanced orientation and position characteristics of the raw image.

I also examined the effect of `cell_per_block`. The lowest row in Figure 3 shows features with different `cell_per_block=1,2,4,6`. I couldn't observed any qualitative difference in this case. Here, I selected `cell_per_block=2` but there is no specific reason for choosing that parameter.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In addition to HOG features, I calculated color and spacial binning features, and I combined them. The code of color and spatial binning feature extraction is found in the `3.2 Color Feature Extraction` and `Spatial Binning Feature Extraction` sections of `TrainModel.ipynb`.

I also applied Data Augmentation techniques to make robust model. I applied image flipping to the vehicle image, while more generic affine transformation is applied to the non-vehicle images. To reduce false positive detection, I introduce larger number of non-vehicle samples than vehicle samples.

|vehicle| non-vehicle|
|-------|-----------|
|26552   |   |


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

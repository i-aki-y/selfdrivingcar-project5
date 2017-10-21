**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/raw_two_classes.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/hog_features_orient.png
[image4]: ./output_images/hog_features_ppc.png
[image5]: ./output_images/hog_features_cpb.png
[image6]: ./output_images/grid_test_image.png
[image7]: ./output_images/pipeline_example.png
[image8]: ./output_images/video_example.png
[image9]: ./output_images/misclassify.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG features are used in training and evaluation time. I extracted the HOG features by using the function `get_hog_features()` in `utils.py` at line 96. In that function, `skimage.feature.hog` is called. The code where I extracted HOG features from the training data is found in the `3.1 HOG Feature Extraction` section of `TrainModel.ipynb`.

I started by loading all the `vehicle` and `non-vehicle` images. Given dataset are classified into "non-vehicle" and "vehicle" classes. Each class has some subclasses. In this project, these subclasses are not necessary, so that those subclasses are combined into "vehicle" and "non-vehicle" classes. Before combining each class.
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
Figure 1: Examples of vehicle (Left) and non-vehicle (Right) class images.

I applied the `get_hog_features()` which is described above to each color channels of loaded images. After some trial-and-error, I decided to use `YCrCb` color space for this feature extraction, since I thought that each channel of that color space well captures different features of images. Examples of visualized HOG features are shown in Figure 2.

![alt text][image2]
Figure 2: Vehicle and Non-vehicle images in the 'YCrCb' color space and visualized HOG feature of corresponding channel are shown. Parameters used for Hog features extraction are (orient=11, pix_per_cell=8, cell_per_block=2).

####2. Explain how you settled on your final choice of HOG parameters.

I compared raw image and visualized hog features with different parameters and selected parameters which seem capture the image characteristics well.
Visualized examples of HOG features with different parameters are shown in Figure 3.

![alt text][image3]
![alt text][image4]
![alt text][image5]
Figure 3: parameter dependency of visualized HOG features. The first row depicts images of different `orient` value. The middle and bottom rows depict images of `pix_per_cell` and `cell_per_block` dependencies, respectively.

In this Figure, HOG features with `orient=3, 5, 11, 21` are shown. With a few orient number `orient <= 5` the visualized feature too simple to characterize the vehicle shape, while the features with `orient = 11` express the vehicle shape well.
I also compared features with `orient = 11` and `orient = 21`, and I couldn't find any qualitative difference in two features. Therefore I thought that the `orient = 11` is sufficient.

Effects of `pix_per_cell` parameter are also given in Figure 3 in the middle row. When the small `pix_per_cell` such as `pix_per_cell=4` the HOG feature depict fine structure for each position, but the orientational variation of features of a single position is limited. On the other hand, features with large `pix_per_cell` such as `pix_per_cell=16, 32` the orientational characteristics expresses fine but positional characteristics are poorly expressed. I thought that the features with `pix_per_cell=8` balanced orientation and position characteristics of the raw image.

I also examined the effect of `cell_per_block`. The lowest row in Figure 3 shows features with different `cell_per_block=1,2,4,6`. I couldn't observe any qualitative difference in this case. Here, I selected `cell_per_block=2` but there is no specific reason for choosing that parameter.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In addition to HOG features, I calculated color and spacial binning features, and I combined them. The code of color and spatial binning feature extraction is found in the `3.2 Color Feature Extraction` and `Spatial Binning Feature Extraction` sections of `TrainModel.ipynb`.

I also applied Data Augmentation techniques to make a robust model. I applied image flipping to the vehicle image, while the more generic affine transformation is applied to the non-vehicle images. To reduce false positive detection, I introduce a larger number of non-vehicle samples than vehicle samples. The numbers of examples for each class are shown in Table 1.

| vehicle | non-vehicle |
|---------|-------------|
| 17584   | 26904       |
Table 1: Numbers of examples for each class.

I split data by using `sklearn.model_selection.train_test_split()` function in `4.4 Split Train and Test` sections of `TrainModel.ipynb`, where I held out 20% of features as test data. After splitting the data, I fitted scaler (`sklearn.preprocessing.StandardScaler`) by using split training data, since test data should be assumed to be unknown.

In order to classify vehicle and non-vehicle, I used SVM model `LinearSVC` from `sklearn.svm` and `GridSearchCV` from `sklearn.model_selection` for hyperparameter search.
I examined the accuracy of the model with different `C` values (`C=[0.5, 1, 5]`), and I obtained same accuracies:
```
[mean: 0.99042, std: 0.00042, params: {'C': 0.5},
 mean: 0.99042, std: 0.00042, params: {'C': 1},
 mean: 0.99042, std: 0.00042, params: {'C': 5}].
```
I applied the trained model with `C=0.5` to test data, and the resulting accuracy is `0.99`.  I think the obtained train and test accuracies is so high and robust that I decide to use the model with `C=0.5`.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I plotted given example image and estimate vehicle size in the image (see Figure 4). The vehicle size seems to be about 50-150 pixels. Since the size of vehicles of training images are 64 pixels, the range of scale parameter will be about 0.8-2.3. From this image, I also assumed that the region where vehicles appear is limited in the rage 400-600 in height.

![alt text][image6]
Figure 4: An example including vehicles with grids is shown.

Sliding window search is implemented in `utils.find_cars()` at line 177 where the same feature extraction process is applied to the given image and returns classification result. Overlap windows are set as 2 cells (16 pixels) per step.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline of the final model is followings:

1. Convert color space to 'YCrCb'
2. Combined 3-channel HOG features,  spatially binned color, and histograms of color in the feature vector.
3. Apply sliding window search to the given image with different scale and region in height.

| Search area in height | scale |
|-----------------------|-------|
| 400 - 500             | 1.0   |
| 400 - 550             | 1.2   |
| 400 - 600             | 1.5   |
| 450 - 650             | 1.7   |
| 450 - 650             | 2.0   |
| 480 - 650             | 2.3   |

4. Voting the count of the detected vehicle area.
5. Ignore the area whose count is less than given threshold value.
6. Redefine the bounding boxes.

The result of pipeline for test image is shown in Figure 5

![alt text][image7]
Figure 5: Final pipeline result for given test images.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video/sol_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code of filtering of false positive detection is implemented in `process_image()` function defined in `VehicleDetection.ipynb`.
First, I collect the positions of positive detections for multiple scales in each frame of the video. I created a heatmap which expresses the count of positive detection for each position. I ignore the heatmap value which is less than a given thresholded in order to reduce the false positive detection. The threshold value is a hyperparameter. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

In the solution video case, I averaged the bounding box collected from past 15 video frames and threshold of the count is set as `threshold=4`.  Figure 6 is an example result showing the collected bounding box and heatmap and labeld region.

![alt text][image8]
Figure 6: Upper left is resized image overlaid by collected bounding box and upper middle is its heatmap and labeld result is upper right. The main area shows final result.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Although the accuracy of trained SVM model seems to be is sufficiently high (99% accuracy), there exists a specific pattern of non-vehicle image which is misclassified. For example in Figure 6 area including yellow lane is misclassified. I think this is because the lane which is diagonal with perspective effect is similar to vehicle edge. But this is completely different from vehicle. This suggests there is much room to improve the model.

![alt text][image9]
Figure 6: Misclassified bounding box

In this project, prepared training images of vehicle class are 4-wheel vehicles. In the real world, there are many types of vehicle such as 2 wheel bite. The model trained in this project will fail to detect such an unknown vehicle.

Collecting much more training data which have many types of vehicle and training model will improve the model, but this approach will take many computational resources. And other types of a model such as CNN which achieves good performance for image recognition will improve the model.

Another problem with this approach is that the sliding window method needs much computational time. In this project, I spend about 30 min to obtained 1 min video. If we want to apply this pipeline to the real world self-driving car system, we need nearly real-time detection. In this case, CNN approach also may be the best candidate. For example, [YOLO](https://pjreddie.com/darknet/yolo/) is known as one of the examples which achieve real-time object detection.

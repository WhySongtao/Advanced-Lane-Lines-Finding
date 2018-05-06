## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/test_image.png "Original"
[image2]: ./output_images/undistorted_test_image.png "Undistorted"
[image3]: ./output_images/undistorted_test2.jpg "Road Image"
[image4]: ./output_images/thresholded_test2.jpg "thresholded Example"
[image5]: ./output_images/Warped_test2.jpg "Warped Image"
[image6]: ./output_images/Window_test2.jpg "Poly Fit"
[image7]: ./output_images/Polygon.jpeg "Polygon"
[video1]: ./project_video_results.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in `./line_utils.py, line 32-62`.

It is important to calibrate the camera before anything else so that the image can reflect real road. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

This is one of the test images that has been undistorted:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this part is in `./line_utils.py, line 65-96`

I used a combination of color and gradient thresholds to generate a binary image. Firstly, I used sobel to calculate the gradient on x and y axis (`abs_sobel_thresh` function). Secondly, I converted the image to hls color space and save the s channel image(`hls_thresh` function). Thirdly, I generated the binary image that has both x gradient and y gradient or has non-zero pixel value in s channel image. Lastly, I used a mask to isolate the area that most likely contains road lane (`region_of_interest` function). 

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this part is in `./line_utils.py, line 159-180`

The code for my perspective transform includes a function called `warp()`, which appears in the 9th code cell of the IPython notebook).  The `warp()` function takes as inputs an image (`img`), and use hardcoded source (`src`) and destination (`dst`) points.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 457      | 300, 0        | 
| 700, 457      | 960, 0        |
| 200, 719      | 300, 719      |
| 1125, 719     | 960, 719      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this part is in `./line_utils.py, line 247-378`

I used sliding window method to position the lane-line pixels(`sliding_window` function). If the previous frame has already found the polynomial fit, I will use that as a starting point to find the lane line within certain margin(`not_sliding_window` function).

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this part is in `./line_utils.py, line 206-219`

I did this by utilising the math equation for calculating the curvature.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this part is in `./line_utils.py, line 222-244`

I ploted the polygon area of the lane back to the road image in function `draw`

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_results.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The issue I was facing in this project is how to choose the right method to be used to detect lines. This involved a lot of fine-tuning and trial and error. Also, how to deal with difficult image is still a challenge to me
if I were going to pursue this project further, I will probably try to use deep learning method to detect lanes. 

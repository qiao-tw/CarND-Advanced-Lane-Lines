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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test2.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./tracked_1.mp4 "Video"
[image7]: ./camera_cal/calibration1.jpg "Before Undistorted"
[image8]: ./output_images/undistorted_calibration1.jpg "After Undistorted"
[image9]: ./output_images/undistorted_test2.jpg "Test After Undistorted"
[image10]: ./output_images/color_grad_thresh_test2.jpg "Test After Color/Gradient Thresholding"
[image11]: ./output_images/warped_test2.jpg "Test After Perspective Transform"
[image12]: ./output_images/road_warped_test2.jpg "Test After Perspective Transform"
[image13]: ./output_images/result2.jpg "sample of results"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `cam_cal.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

<br> sample image before undistorted<br>
![alt text][image7]
<br> sample image after undistorted<br>
![alt text][image8]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

I used `cv2.undistort(image, mtx, dist, None, mtx)` function to recover distortion of test images. Following are before/after example of 1 of test images:

<br> sample image before undistorted <br>
![alt text][image2]
<br> sample image after undistorted <br>
![alt text][image9]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 73 through 82 in `image_gen.py`).  Here's an example of my output for this step.

![alt text][image10]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in line 89-91, 147-148 of image_gen.py.

I didn't hardcode the source and destination points. Instead, I used the code piece from Udacity SDC P4 Q&A session:

```python
  img_size = (img.shape[1], img.shape[0])
  bot_width = 0.76
  mid_width = 0.08
  height_pct = 0.62
  bottom_trim = 0.935
  src = np.float32([[img.shape[1]*(0.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(0.5+mid_width/2),img.shape[0]*height_pct],
      [img.shape[1]*(0.5+bot_width/2),img.shape[0]*bottom_trim], [img.shape[1]*(0.5-bot_width/2),img.shape[0]*bottom_trim]])
  offset = img_size[0]*0.80
  dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

  M = cv2.getPerspectiveTransform(src, dst)
  Minv = cv2.getPerspectiveTransform(dst, src)
  warped = cv2.warpPerspective(preprocessImage,M,img_size,flags=cv2.INTER_LINEAR)
```

The warped image of test2.jpg:
![alt text][image11]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for identifying lane-line pixels and fitting their positions with a polynomial is in line 108 through 158 of image_gen.py.

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 173 through 174 in my code in `image_gen.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 167 through 186 in `image_gen.py`. Here is an example of my result on a test image:

![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./tracked_1.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

There are still many cases when the pipeline has difficuties:
* The color-thresholding sometimes fails to identify both white and yellow lines
* Bright objects near lanes mislead the fitting algorithm to fit wrong polynomials.

More could be done:
* More sophisticated logic to combine different thresholds( e.g. under shadows, edge detection seems to work better then color detection)
* Detect and eliminate other lines: very often we have other lines that run parallel to the lanes (e.g. other lanes, the edge of the road, a mark of the road), sliding windows some can mistake these lines as the main lanes and bias the curve fitting. tracing these lines can help eliminate them and improve the robustness of the pipeline

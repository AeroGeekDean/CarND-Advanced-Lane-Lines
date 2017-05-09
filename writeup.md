## Advanced Lane Finding Project Writeup

---


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

[image1]: ./output_images/distortion_calibration_sample.png "Undistorted"
[image2]: ./output_images/test_images.png "Road Undistorted"
[image3a]: ./output_images/sobel_x_sample.png "Sobel-X Examples"
[image3b]: ./output_images/HLS-S_ch_samples.png "HLS S-ch Examples"
[image3c]: ./output_images/processed_images.png "Processed Examples"
[image4a]: ./output_images/perspectX_calibration1.png "Perspective Calibration 1"
[image4b]: ./output_images/perspectX_calibration2.png "Perspective Calibration 2"
[image4c]: ./output_images/perspectX_calibration3.png "Perspective Calibration 3"
[image5]: ./output_images/pipeline_test_images.png "Pipeline example"
[image6]: ./output_images/final_output.png "Output"
[image6a]: ./output_images/final_output_fault.png "Output"
[video1]: ./my_project_video.mp4 "Video"

--

### Camera Calibration

The code for this step is contained in the code source file `./Calibration.py` and used by the Jupyter notebook located in `./P4-Project_notebook.ipynb`.  The `Calibration.py` file encapsulates the `Calibration` Class. And its usage is demonstrated in the notebook.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

The code for this step is contained in the code cells of the Jupyter notebook. It's the function `find_lane_lines(img)`, located towards the bottom of the notebook. Look for the section **"Let's build all of the above into a pipeline"**

Once the test images are loaded in (via `cv2.imread()` function), I immediately applied:
- Camera distortion correction
- BGR2RGB color space conversion.

Thus avoiding potential downstream mixup by working exclusively in RGB colorspace.

Below are the 6 test images with distortion correction applied:
![alt text][image2]


I used a combination of color and gradient thresholds to generate binary images.

I used:
- Sobel X-gradient with threshold at (20,200)
- HLS S-channel with threshold at (170,255)
- union of above 2 results

Here's an example of my outputs for this step.

Sobel-X outputs:
![alt text][image3a]

HLS S-channel outputs:
![alt text][image3b]

Combined processed outputs:
![alt text][image3c]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is a Class located in the file `'PerspectiveTransform.py'`. Its usage in demonstrated in the notebook under the **Perspective Transform** section.

I manually but judicially chosen the transform polygon coordinates in the straight test images. By utilizing the following dimensional information from US DOT:
- Dashed white lane line = 10 ft long
- Minimum lane width = 12 ft

I marked the screen pixel locations using these as reference points.
```python
# manually draw source polygon...

# Left lane line's X&Y end points
xl = np.array([265, 440])
yl = np.array([695, 565])

# Right lane line's X coordinates. Using same Y coordinates as left lane line
xr = np.array([1055, 860])

# center of the L&R lines
xc = 655

#Flip order of right line endpoints to specify an enclosing polygon
xpoly = np.concatenate((xl, xr[::-1]))
ypoly = np.concatenate((yl, yl[::-1]))
```

```python
#set src coordinates
src_x = xpoly
src_y = ypoly
src_road = np.stack((src_x, src_y), axis=0).T.astype(np.float32)
print(src_road)
```
>[[  265.   695.]
 [  440.   565.]
 [  860.   565.]
 [ 1055.   695.]]

Below, the red box is the source polygon `'src_road'`
![alt text][image4a]

Next I specify a matching occupying box on the birds-eye view screen.

```python
# Now let's determine what the destination coordinate for these area are...
# Note: ASSUME this boxed distance to be:
# - 12 ft wide (min lane width)
# - 10ft long (dashed lane line length)
# Per US Regulations
dst_width = 1280
dst_left_pct = 0.25
dst_right_pct = 0.75
dst_hgt = 720
dst_box_hgt = 80

dst_x = dst_width*np.float32([dst_left_pct, dst_left_pct, dst_right_pct, dst_right_pct])
dst_y = [dst_hgt, dst_hgt-dst_box_hgt, dst_hgt-dst_box_hgt, dst_hgt]

dst_road = np.stack((dst_x, dst_y), axis=0).T.astype(np.float32)
print(dst_road)
```
>[[ 320.  720.]
 [ 320.  640.]
 [ 960.  640.]
 [ 960.  720.]]

And to calibrate the perspective transform object...
```python
# Perspective transform
from PerspectiveTransform import PerspectiveTransform

myXform = PerspectiveTransform()

myXform.set_src(src_road)
myXform.set_dst(dst_road)
myXform.calibrate()
```
Once calibrated, to apply the transform, simply call `'PerspectiveTransform.transform(img)'` method on the class instance.
![alt text][image4b]

Finally I **inverse transform** the image back to driver's-eye view to verify with the original image to verify the transformations were done correctly, by calling `'PerspectiveTransform.inv_transform(img)'`
![alt text][image4c]


The actual finding of lane lines are accomplished using several functions located in `'main.py'`. They are:
- `'find_curvatures()'`
- `'find_window_centroids()'`
- `'window_mask()'`

Coupled with code in the notebook, under the section **Find lane line curvature** where I explored and prototyped the process for a single image.

Here are the results (including calculation of lane curvature and lateral deviation):
![alt text][image5]


The radius of curvature is a function called `'curvature_radius(fit, y)'` located in `'main.py'`.

```python
def curvature_radius(fit, y):
    radius = ((1 + (2*fit[0]*y + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return radius
```


In the final pipeline section **Let's build all of the above into a pipeline**, a `'find_lane_lines(img)'` function is defined. Here is an example of my result pulled from the video:

![alt text][image6]

#### Line validity detection
I also implemented line validity detection by monitoring how many pixels the algorithm has detected and using to predict each lane line. If the number of pixels drops below some threshold, the validity is flagged FALSE, and the display text turns RED.
![alt text][image6a]

Additionally,
- the 2nd degree polynomial lane line tracking from the previous valid frame is used to dead-reckon over the rough spots
- a "Cold Start" on the windowing search for initialized, where a more expansive search is conducted on the next frame.

**Furthermore**, to maintain temporal continuity on the near-end of the lane line from frame-to-frame (ie: the lines near the camera will NOT jump away from its previous positioin discontinuously), **each frame's windowing search at the bottom of the screen will initialize based on the position of the previous frame.**

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my YouTube video](https://www.youtube.com/watch?v=aEZYx9vGwJI)

Here's link to the ["debug" version of the video](https://youtu.be/8tPBpahHpcg), where the algorithm's view is provided.

---

### Discussion

The amount of time it took to trail-n-error the different aspects of this project took up more time than I anticipated. As such, I now only have 1.5 days left to scramble thru project 5 before my deadline on 5/10. If I don't make the deadline, I'll get kicked out of the SDC program. Yes, this is my 2nd extension (the paid one).
:(

The initial computer vision portion of the project were pretty straight forward. What was more difficult was trying to get the implement the windowing search (of lane lines) algorithm, and making it not so fragile. Then after that is the temporal aspect of frame-to-frame dynamics. Luckily, we could take advantage of the continuous nature of the physical world's motion, and devise methods to make the system more robust. I've discussed above what I've implemented. Other improvement possibilities are:
- utilizing **Hough transform** to find pixels that are more likely to be co-linear to improved line detection.
- if I had more time, I would try to implement the polynomial such that the bottom of the screen is the origin of the Y-axis. This way, the constant coefficient term would be exactly the lateral deviation. And the higher ordered terms might be able to map to sharpness of the turn, and possibly help with upcoming impact on trajectory planning / guidance for vehicle/tire dynamic purpose.
- possibly use some sort of **predictive-corrective dynamic filtering** (Kalman or variations there of) for tracking the lanes, to take advantage of the continuous nature of vehicle's forward motion.

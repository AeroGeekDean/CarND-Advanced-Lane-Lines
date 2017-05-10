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

The code for this step is contained in the code source file [`./Calibration.py`](./Calibration.py) and used by the Jupyter notebook located in [`./P4-Project_notebook.ipynb`](./P4-Project_notebook.ipynb).  The `Calibration.py` file encapsulates the `Calibration` Class. And its usage is demonstrated in the notebook.

During class instantiation, the constructor calls the private method `'_prep_objpoints()'` which prepares the "object points" based on the parameters for the chessboard dimensions. Thus automating and abstracting this step away from the class user. The "object points" represents where the chessboard corners are in the world.

The user then informs the object of the 'path' and 'file pattern' for the test image by calling the `'set_img_location(path, pattern)'` method.

After that, the user calls the `'calibrate()'` method, which takes care of loading and identifying the chessboard corners within these images (via `'cv2.findChessboardCorners()'` function), and saves these corner data as the class member data `'imgpionts'`.

Now the object is ready to undistort new images via the `'apply(img)'` method. Below is a sample result.
![alt text][image1]

### Pipeline (single images)

The code for this step is contained in the code cells of the Jupyter notebook. It's the function `find_lane_lines(img)`, located towards the bottom of the notebook. Look for the section **"Let's build all of the above into a pipeline"**

#### Image Processing

Once the test images are loaded in (via `cv2.imread()` function), I immediately applied:
- Camera distortion correction
- BGR2RGB color space conversion

Thus avoiding potential downstream colorspace mixup by working exclusively in RGB.

Below are the 6 test images with distortion correction applied:
![alt text][image2]

I used a combination of color and gradient thresholds to generate binary images.

I used:
- Sobel X-gradient with threshold at (20,200)
- HLS S-channel with threshold at (170,255)
- union ('or') of above 2 results

Here's an example of my outputs for this step.

**Sobel-X outputs:**
![alt text][image3a]

**HLS S-channel outputs:**
![alt text][image3b]

**Combined processed outputs:**
![alt text][image3c]

#### Perspective Transform

The code for my perspective transform is a Class located in the file [`'./PerspectiveTransform.py'`](./PerspectiveTransform.py). Its usage is demonstrated in the notebook under the **Perspective Transform** section.

I manually but judicially chose the transform polygon coordinates in the straight test images, by utilizing the following dimensional information from US DOT:
- Dashed white lane line = 10 ft long
- Minimum lane width = 12 ft

I marked the screen pixel locations using the end points of a white dashed lane line as reference.
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
> [  440.   565.]
> [  860.   565.]
> [ 1055.   695.]]

Below, the reference red box is the source polygon `'src_road'`
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
> [ 320.  640.]
> [ 960.  640.]
> [ 960.  720.]]

And to calibrate the perspective transform object...
```python
# Perspective transform
from PerspectiveTransform import PerspectiveTransform

myXform = PerspectiveTransform()

myXform.set_src(src_road)
myXform.set_dst(dst_road)
myXform.calibrate()
```
Once calibrated, to apply the transform, simply call `'transform(img)'` method on the class instance.

**Note the reference red box is now much smaller in the top-down bird's eye view, per my explicit specification above**
![alt text][image4b]

Finally I **inverse transform** the image **back to driver's-eye view** to verify with the original image that the transformations were done correctly, by calling `'inv_transform(img)'`. **This also allows me to see the visibility limit of the bird's eye view, via the green boarder.**
![alt text][image4c]

#### Finding Lane Lines
The actual "finding of lane lines" were accomplished using several functions located in [`'./main.py'`](./main.py).

They are:
- `'find_curvatures()'`
- `'find_window_centroids()'`
- `'window_mask()'`

Coupled with code in the notebook, under the section **Find lane line curvature** where I explored and prototyped the process for individual test images.

Here are the results (including calculation of lane curvature and lateral deviation):
![alt text][image5]

#### Radius of Curvature
The radius of curvature is a function called `'curvature_radius(fit, y)'` located in `'main.py'`.

```python
def curvature_radius(fit, y):
    radius = ((1 + (2*fit[0]*y + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return radius
```

In the final pipeline section **Let's build all of the above into a pipeline**, a `'find_lane_lines(img)'` function is defined. This is the **actual pipeline function** for the video processing. Here is an example of my result pulled from the video:
![alt text][image6]

#### Line Validity Detection
I also implemented line validity detection by monitoring how many pixels the algorithm has detected and are using to predict each lane line. If the number of pixels drops below some threshold, the validity is flagged FALSE.

When this happens:
- the display text turns RED.
- the lane line from the previous valid frame is used to dead-reckon over the current frame
- a "Cold Start" on the windowing search is initialized, where a more expansive (but more robust) search is conducted on the next frame.

![alt text][image6a]

**Furthermore**, to maintain temporal continuity from frame-to-frame, **each frame's windowing search at the bottom of the screen will initialize based on the position of the previous frame.** (ie: The lines near the camera will NOT jump away from its previous positioin discontinuously)

---

### Pipeline (video)

Here's a [link to my YouTube video](https://www.youtube.com/watch?v=aEZYx9vGwJI)

Here's link to the ["debug" version of the video](https://youtu.be/8tPBpahHpcg), where the algorithm's view is provided.

---

### Discussion

The amount of time it took to trail-n-error the different aspects of this project took up more time than I anticipated.

The initial computer vision portion of the project were pretty straight forward. What was more difficult was trying to get a good working implementation of the (lane lines) windowing search algorithm, and making it "not so fragile" to each individual frame's image. Some frames have horrible shadows that massively interfere with the computer vision algorithm. Luckily, we could take advantage of the continuous nature of the physical world's motion, and devise methods to make the system more robust. I've discussed the fault detection and handling that I implemented (in the **'Line Validity Detection'** section above).

Other possible improvements are:

- Utilize **Hough transform** to find co-linear pixels that are more likely to be lane lines, thus improving detection.

- Invert the Y-axis of the 2nd order fit polynomial
Currently the Y-axis zero is at the top of the screen, with positive Y-axis pointing down (per numpy & general image matrix convention). By inverting, the polynomial coefficients would have more physically representing meanings. (example: the constant term will directly become the lateral offset, while the higher order terms could map to curvature sharpness and thus associated vehicle lateral acceleration)

- possibly use some sort of **predictive-corrective time-based filtering** (Kalman or variations there of) for tracking the lanes, to take advantage of the known continuous nature of vehicle's forward motion.

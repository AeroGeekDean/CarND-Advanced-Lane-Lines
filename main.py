import numpy as np
import cv2
from Calibration import Calibration
from PerspectiveTransform import PerspectiveTransform



# camera distortion handler
# myCal = Calibration()

# perspective transform handler
# myXform = PerspectiveTransform()

def main():

    file_path = ''
    file_pattern = ''

    # initialize
    myCal.set_img_location(file_path, file_pattern)
    myCal.calibrate()

    src_x = [280, 440, 860, 1035]
    src_y = [680, 563, 563, 680]
    dst_x = [1280*0.25, 1280*0.25, 1280*0.75, 1280*0.75]
    dst_y = [720, 600, 600, 720]
    src_road = np.stack((src_x, src_y), axis=0).T.astype(np.float32)
    dst_road = np.stack((dst_x, dst_y), axis=0).T.astype(np.float32)

    myXform.set_src(src_road)
    myXform.set_dst(dst_road)
    myXform.calibrate()

    # run
    find_lane_lines(None)

    exit()

def find_lane_lines(img):

    # correct for distortion
    img_undistorted = myCal.apply(img)

    # process image
    img_gray = cv2.cvtColor(img_undistorted, cv2.COLOR_RGB2GRAY)
    img_sbx = abs_sobel_thresh(img_gray, orient='x',
                               sobel_kernel=3, thresh=(20,200))
    img_sbin = hls_thresh(img_undistorted, ch_sel='s', thresh=(170,255))
    img_combo = (img_sbx | img_sbin)

    # apply perspetive transform (pre-calibrated)
    img_bird = myXform.transform(img_combo)

    # curvature finding
    fit_l, fit_r = find_curvatures(img_bird, margin=50)

    return fit_l, fit_r

#----------------------------
# HLS colorspace + threshold
#----------------------------

def hls_thresh(img_rgb, ch_sel='s', thresh=(0, 255)):
    hls = cv2.cvtColor(np.copy(img_rgb), cv2.COLOR_RGB2HLS)
    if (ch_sel=='h'):
        ch = hls[:,:,0]
    elif (ch_sel=='l'):
        ch = hls[:,:,1]
    elif (ch_sel=='s'):
        ch = hls[:,:,2]
    else:
        ch = hls[:,:,:]
    binary_output = np.zeros_like(ch)
    binary_output[(thresh[0]<=ch)&(ch<thresh[1])] = 1
    return binary_output

#-----------------
# Sobel functions
#-----------------

def abs_sobel_thresh(gray_img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient=='x':
        sobel = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    elif orient=='y':
        sobel = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # Apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(thresh[0]<=scaled_sobel) & (scaled_sobel<thresh[1])] = 1
    return grad_binary

def mag_thresh(gray_img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    sbx = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sby = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    sbm = np.sqrt(sbx**2 + sby**2)
    scaled_sbm = np.uint8(255*sbm/np.max(sbm))
    # Apply threshold
    mag_binary = np.zeros_like(scaled_sbm)
    mag_binary[(thresh[0]<=scaled_sbm) & (scaled_sbm<thresh[1])] = 1
    return mag_binary

def dir_thresh_abs(gray_img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sbx = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sby = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    sbdir = np.arctan2(sby, sbx)
    # Apply threshold
    dir_binary = np.zeros_like(sbdir)
    dir_binary[(thresh[0]<=sbdir) & (sbdir<thresh[1])] = 1
    return dir_binary

def dir_thresh(gray_img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sbx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sby = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sbdir = np.arctan2(sby, sbx)
    # Apply threshold
    dir_binary = np.zeros_like(sbdir)
    dir_binary[(thresh[0]<=sbdir) & (sbdir<thresh[1])] = 1
    return dir_binary

#--------------------
# Curveature finding
#--------------------

# class init_centroid(object):
#     def __init__(self):
#         self.value = None
#         return
#
#     def set(self, ic_tuple):
#         self.value = ic_tuple
#         return
#
# ic = init_centroid()

def find_curvatures(binary_img, w_width=80, w_hgt=80, margin=100):
    '''
    binary_img : binary image
    w_width : window width
    w_hgt : window height
    margin : how much to slide each successive layer left/right for searching
    '''
    # verify input image is binary!
    # if (binary_img.shape[])

    window_centroids = find_window_centroids(binary_img, w_width, w_hgt, margin)

    # found any centers
    if (len(window_centroids)>0):
        # Points used to draw the lane pixels within left and right windows
        l_points = np.zeros_like(binary_img)
        r_points = np.zeros_like(binary_img)
        # Go through each level and draw the lane pixels
        for level, centroids in enumerate(window_centroids):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(w_width, w_hgt, binary_img, centroids[0], level)
            r_mask = window_mask(w_width, w_hgt, binary_img, centroids[1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[((l_mask==1) & (binary_img==1)) | (l_points==1)] = 1
            r_points[((r_mask==1) & (binary_img==1)) | (r_points==1)] = 1

        # Identify the x and y positions of all drawn pixels in the images
        left_y, left_x = l_points.nonzero()
        right_y, right_x = r_points.nonzero()

        # Fit a second order polynomial to each
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        # update init_centroid for next frame
        # ic.set(window_centroids[0])
    else:
        # return straight lines
        left_fit = right_fit = [0.0, 0.0, 0.0]
    return left_fit, right_fit

def find_window_centroids(image, window_width, window_height, margin):

    window_centroids = [] # Store the (L,R) window centroid positions per level
    window = np.ones(window_width) # window template for convolution use

    # First find the two starting positions for the left and right lane by using
    # np.sum to get the vertical image slice and then np.convolve the vertical
    # image slice with the window template

    # -- initialize the L & R centroid positions --

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(image.shape[0]*3/4):,
                         :int(image.shape[1]/2)], axis=0) # bottom 1/4 hgt, left side
    l_center = np.argmax(np.convolve(window,l_sum)) -window_width/2
    # l_center = np.argmax(np.convolve(window,l_sum)) -window_width/2
    # Note - conv signal Ref is R side of window, thus +half_width

    r_sum = np.sum(image[int(image.shape[0]*3/4):,
                         int(image.shape[1]/2):], axis=0) # bottom 1/4 hgt, right side
    r_center = np.argmax(np.convolve(window,r_sum)) -window_width/2 +int(image.shape[1]/2)

    # Add the init layer centroid pos
    window_centroids.append( (l_center,r_center) )

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        top_idx    = int(image.shape[0]-(level+1)*window_height)
        bottom_idx = int(image.shape[0]-level*window_height)

        image_layer = np.sum(image[top_idx:bottom_idx,:], axis=0)
        conv_signal = np.convolve(window, image_layer)

        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is
        #   at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) +l_min_index -offset

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) +r_min_index -offset

        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
    return window_centroids

def window_mask(width, height, img_ref, center, level):
    # width = window width
    # height = window hgt
    # img_ref = for img.shape
    # center = window center (X-axis)
    # level = current vertical level (0-based)

    # index for window sides
    top_idx    = int(img_ref.shape[0]-(level+1)*height)
    bottom_idx = int(img_ref.shape[0]-level*height)
    left_idx   = max(0,int(center-width/2))
    right_idx  = min(int(center+width/2),img_ref.shape[1])

    output = np.zeros_like(img_ref)
    output[top_idx:bottom_idx, left_idx:right_idx] = 1
    return output

def curvature_radius(fit, y):
    radius = ((1 + (2*fit[0]*y + fit[1])**2)**1.5) / np.absolute(2*fit[0])
    return radius


if __name__ == '__main__':
    main()

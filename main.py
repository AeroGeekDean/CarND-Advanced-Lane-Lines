import numpy as np
import cv2
from Calibration import Calibration
from PerspectiveTransform import PerspectiveTransform



# camera distortion handler
myCal = Calibration()

# perspective transform handler
myXform = PerspectiveTransform()

#----------------------------
# HLS colorspace + threshold
#----------------------------

def cvtColor_thresh(img_rgb, cvt=cv2.COLOR_RGB2HLS, ch_sel=2, thresh=(0, 255)):
    new = cv2.cvtColor(img_rgb, cvt)
    ch = new[:,:,ch_sel]
    binary_output = np.zeros_like(ch)
    binary_output[(thresh[0]<=ch)&(ch<thresh[1])] = 1
    return binary_output

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

def find_curvatures(binary_img, start_pts, w_width=80, w_hgt=80, margin=40):
    '''
    binary_img : binary image
    w_width : window width
    w_hgt : window height
    margin : how much to slide each successive layer left/right for searching
    '''
    # verify input image is binary!
    # if (binary_img.shape[2])

    window_centroids = find_window_centroids(binary_img, start_pts, w_width, w_hgt, margin)

    # Points used to draw the lane pixels within left and right windows
    l_points = np.zeros_like(binary_img)
    r_points = np.zeros_like(binary_img)

    win_img = np.zeros_like(binary_img) # DEBUGGING

    # Go through each level and draw the lane pixels
    for level, centroids in enumerate(window_centroids):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(w_width, w_hgt, binary_img, centroids[0], level)
        r_mask = window_mask(w_width, w_hgt, binary_img, centroids[1], level)

        win_img[((win_img==1) | (l_mask==1))] = 1 # DEBUGGING
        win_img[((win_img==1) | (r_mask==1))] = 1 # DEBUGGING

        # Add graphic points from window mask here to total pixels found
        l_points[((l_mask==1) & (binary_img==1)) | (l_points==1)] = 1
        r_points[((r_mask==1) & (binary_img==1)) | (r_points==1)] = 1

    # Identify the x and y positions of all drawn pixels in the images
    left_y, left_x = l_points.nonzero()
    right_y, right_x = r_points.nonzero()

    fit_valid = True

    # Fit a second order polynomial to each
    if (left_y.size<1000):
        print('Fit FAIL: left_fit')
        fit_valid = False
        left_fit = [0.0, 0.0, 0.0]
    else:
        left_fit = np.polyfit(left_y, left_x, 2)

    if (right_y.size<1000):
        print('Fit FAIL: right_fit')
        fit_valid = False
        right_fit= [0.0, 0.0, binary_img.shape[1]/2]
    else:
        right_fit = np.polyfit(right_y, right_x, 2)

    # DEBUG - generate image of window (green) and found points (white)
    zero_ch = np.zeros_like(binary_img)
    win_image = cv2.merge((zero_ch,zero_ch,win_img))*255 # blue window boxes
    pts_image = cv2.merge((l_points,r_points,zero_ch))*255 # left red, right green
    binary_image = cv2.merge((binary_img,binary_img,binary_img))*255
    points_img = cv2.addWeighted(binary_image, 0.5, pts_image, 1, 0.0)
    output_img = cv2.addWeighted(points_img, 1, win_image, 0.5, 0.0)

    return fit_valid, left_fit, right_fit, window_centroids[0], output_img, (left_y.size, right_y.size)

def find_window_centroids(image, start_pts, window_width, window_height, margin):

    # Note - algorithm will search ALONG the path of prev_centroids
    #        with +/-margin wiggle

    window_centroids = [] # Store the (L,R) window centroid positions per level
    window = np.ones(window_width) # window template for convolution use

    img_hgt, img_wid = image.shape
    n_levels = int(img_hgt/window_height)

    # First find the two starting positions for the left and right lane by using
    # np.sum to get the vertical image slice and then np.convolve the vertical
    # image slice with the window template

    # -- initialize the L & R centroid positions --

    # Cold start. Search img bottom for starting window locations
    if start_pts == (None, None):
        print('find_window_centroids(): Cold starting window centroid search...')
        # Sum bottom 1/4 of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(img_hgt*3/4):,:int(img_wid/2)], axis=0)
        r_sum = np.sum(image[int(img_hgt*3/4):,int(img_wid/2):], axis=0)

        # Note - conv signal Ref is R side of window, thus +half_width
        l_center = np.argmax(np.convolve(window,l_sum)) -window_width/2
        r_center = np.argmax(np.convolve(window,r_sum)) -window_width/2 +int(img_wid/2)
    else:
        (l_center, r_center) = start_pts
        # limit each line start point to their own halves of the image
        l_center = min(max(l_center,0),img_wid/2)
        r_center = min(max(r_center,img_wid/2),img_wid)

    # Go through each layer looking for max pixel locations
    for level in range(n_levels):
        # convolve the window into the vertical slice of the image
        top_idx    = int(img_hgt-(level+1)*window_height)
        bottom_idx = int(img_hgt-level*window_height)

        image_layer = np.sum(image[top_idx:bottom_idx,:], axis=0)
        conv_signal = np.convolve(window, image_layer)

        # Use window_width/2 as offset because convolution signal reference is
        #   at right side of window, not center of window
        offset = window_width/2

        # Find the best left centroid by using past left center as a reference
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,img_wid/2))
        conv_l = conv_signal[l_min_index:l_max_index]
        if np.sum(conv_l)==0:
            pass # if no points found, then keep prev frame value.
        else:
            l_center = np.argmax(conv_l) +l_min_index -offset # update l_center

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,img_wid/2))
        r_max_index = int(min(r_center+offset+margin,img_wid))
        conv_r = conv_signal[r_min_index:r_max_index]
        if np.sum(conv_r)==0:
            pass # if no points found, then keep prev value.
        else:
            r_center = np.argmax(conv_r) +r_min_index -offset # update r_center

        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
    return window_centroids

def find_window_centroids_saved(image, start_pts, window_width, window_height, margin):

    # Note - algorithm will search ALONG the path of prev_centroids
    #        with +/-margin wiggle

    window_centroids = [] # Store the (L,R) window centroid positions per level
    window = np.ones(window_width) # window template for convolution use

    img_hgt, img_wid = image.shape
    n_levels = int(img_hgt/window_height)

    # First find the two starting positions for the left and right lane by using
    # np.sum to get the vertical image slice and then np.convolve the vertical
    # image slice with the window template

    # -- initialize the L & R centroid positions --

    # Cold start. Search img bottom for starting window locations
    if start_pts == (None, None):
        print('find_window_centroids(): Cold starting window centroid search...')
        # Sum bottom 1/4 of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(img_hgt*3/4):,:int(img_wid/2)], axis=0)
        r_sum = np.sum(image[int(img_hgt*3/4):,int(img_wid/2):], axis=0)

        # Note - conv signal Ref is R side of window, thus +half_width
        l_center = np.argmax(np.convolve(window,l_sum)) -window_width/2
        r_center = np.argmax(np.convolve(window,r_sum)) -window_width/2 +int(img_wid/2)
    else:
        (l_center, r_center) = start_pts
        # limit each line start point to their own halves of the image
        l_center = min(max(l_center,0),img_wid/2)
        r_center = min(max(r_center,img_wid/2),img_wid)

    # Go through each layer looking for max pixel locations
    for level in range(n_levels):
        # convolve the window into the vertical slice of the image
        top_idx    = int(img_hgt-(level+1)*window_height)
        bottom_idx = int(img_hgt-level*window_height)

        image_layer = np.sum(image[top_idx:bottom_idx,:], axis=0)
        conv_signal = np.convolve(window, image_layer)

        # Use window_width/2 as offset because convolution signal reference is
        #   at right side of window, not center of window
        offset = window_width/2

        # Find the best left centroid by using past left center as a reference
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,img_wid/2))
        conv_l = conv_signal[l_min_index:l_max_index]
        if np.sum(conv_l)==0:
            pass # if no points found, then keep prev frame value.
        else:
            l_center = np.argmax(conv_l) +l_min_index -offset # update l_center

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,img_wid/2))
        r_max_index = int(min(r_center+offset+margin,img_wid))
        conv_r = conv_signal[r_min_index:r_max_index]
        if np.sum(conv_r)==0:
            pass # if no points found, then keep prev value.
        else:
            r_center = np.argmax(conv_r) +r_min_index -offset # update r_center

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

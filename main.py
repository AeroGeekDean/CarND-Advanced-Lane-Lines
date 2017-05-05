import numpy as np
import cv2
from Calibration import Calibration
from PerspectiveTransform import PerspectiveTransform



# calibrate camera distortion
myCal = Calibration()

# calibrate perspective transform
myXform = PerspectiveTransform()

def main():

    find_lane_lines(None)

    exit()

def find_lane_lines(img):

    print(myCal.cwidth)

    # correct for distortion

    # isolate out the lane lines

    # apply perspetive transform (pre-calibrated)

    return

#-----------------
# HLS + threshold
#-----------------

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



if __name__ == '__main__':
    main()

import numpy as np
import cv2
import glob
# import matplotlib.pyplot as plt

class Calibration(object):
    """
    This class handles the calibration of a camera.

    After instantiation, call set_img_location() with path/pattern to image files.
    Then call calibrate() for it to evaluate distortion calibration parameters.
    Fianlly call apply() with image to undistort.
    """
    def __init__(self, width=9, hgt=6):
        self.cwidth = width
        self.cheight = hgt

        self.file_path = None
        self.file_pattern = None

        self.objp = self._prep_objpoints()
        self.imgpoints = [] # list of image points

        self.mtx = None # Camera Matrix
        self.dist = None # Distortion Coefficients
        self.rvecs = None # Rotation Vectors
        self.tvecs = None # Translation Vectors

    def _prep_objpoints(self):
        # prep object points, ex: (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros(((self.cwidth*self.cheight), 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.cwidth, 0:self.cheight].T.reshape(-1,2)
        return objp

    def set_img_location(self, path, pattern):
        self.file_path = path
        self.file_pattern = pattern
        return

    def calibrate(self):
        image_files = glob.glob(self.file_path+'/'+self.file_pattern)
        print('{} image files to process...'.format(str(len(image_files))))

        # grab img pixel size from 1st image. Note format is (width, hgt)
        img_size = cv2.imread(image_files[0]).shape[:2][::-1]

        for idx, fname in enumerate(image_files):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the corners
            ret, corners = cv2.findChessboardCorners(gray, (self.cwidth, self.cheight), None)

            # if found, add image point to list
            if ret == True:
                self.imgpoints.append(corners)

        print('Distortion calibration complete. {} images used'.format(str(len(self.imgpoints))))

        # create list of objectpoints, the same length as imgpoints
        objpoints = [self.objp for i in self.imgpoints]

        [ret, self.mtx, self.dist, self.rvecs, self.tvecs] = \
            cv2.calibrateCamera(objpoints, self.imgpoints, img_size, None, None)
        return

    def apply(self, img):
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst

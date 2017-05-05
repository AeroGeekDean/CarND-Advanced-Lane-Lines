import numpy as np
import cv2
import glob
# import matplotlib.pyplot as plt

class PerspectiveTransform(object):
    """
    docstring for .
    """
    def __init__(self):
        self.M = None
        self.M_inv = None
        self.src = None # source quadrilateral verticies
        self.dst = None # destination quadrilateral verticies

    def set_src(self, src):
        self.src = src
        return

    def set_dst(self, dst):
        self.dst = dst
        return

    def calibrate(self):
        if (self.src is None) | (self.dst is None):
            print('ERROR: src or dst quadrilateral verticies not set!')
            return
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)
        return

    def transform(self, img):
        if self.M is None:
            self.calibrate()
        img_size = img.shape[:2][::-1]
        warped = cv2.warpPerspective(img, self.M, img_size,
                                     flags=cv2.INTER_LINEAR)
        return warped

    def inv_transform(self, img):
        if self.M_inv is None:
            self.calibrate()
        img_size = img.shape[:2][::-1]
        unwarped = cv2.warpPerspective(img, self.M_inv, img_size,
                                       flags=cv2.INTER_LINEAR)
        return unwarped

    def transform_pts(self, pts):
        if self.M is None:
            self.calibrate()
        return cv2.perspectiveTransform(pts, self.M)

    def inv_transform_pts(self, pts):
        if self.M_inv is None:
            self.calibrate()
        return cv2.perspectiveTransform(pts, self.M_inv)

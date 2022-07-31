from __future__ import division
import numpy as np
import cv2

class LogoExtractor:

    def __init__(self, srcImg):
        self.optLogoArea = 0
        self.supportLogoRatio = 2
        self.source = cv2.imread(srcImg)
        self.x, self.y, self.w, self.h = 0, 0, 0, 0
        self.logoEstimate = self.logoIsolator()

    def morphologicalOps(self, img):
        # maybe use sobel operator to detect the edges...
        struc = cv2.getStructuringElement(0, (5, 5))
        return cv2.erode(cv2.dilate(img, struc, iterations=3), struc, iterations=3)

    def isOptimalArea(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        if (w/h<self.supportLogoRatio or h/w<self.supportLogoRatio) and w*h > self.optLogoArea:
            self.optLogoArea = w*h
            return True
        else:
            return False

    def logoIsolator(self):
        contours, hierarchy = cv2.findContours(self.plateIsolator(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.x, self.y, self.w, self.h = cv2.boundingRect(contours[0])
        # times 3 to allow some tolerance (e.g some car's logo sits very high above the licence plate)
        logo = self.source[self.y-self.h*3:self.y, self.x:self.x+self.w]
        return logo

    def refineIsolation(self):
        estimate = self.logoEstimate
        estimate = cv2.cvtColor(cv2.resize(estimate, (2 * estimate.shape[1], 2 * estimate.shape[0])), cv2.COLOR_BGR2GRAY)
        threshold, estimate = cv2.threshold(estimate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        estimate = self.morphologicalOps(cv2.Sobel(estimate, cv2.CV_8U, 1, 0, ksize=3))
        contours, hierarchy = cv2.findContours(estimate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        result = list(filter(self.isOptimalArea, contours))
        x, y, w, h = cv2.boundingRect(result[-1])
        x_start = int(self.x + x/2)
        y_start = int(self.y-self.h*3+y/2)
        return self.source[y_start:int(y_start+h/2), x_start:int(x_start+w/2)]

    def plateIsolator(self):
        # use HSV to filter out the area that is not licence plate...
        source = cv2.cvtColor(self.source, cv2.COLOR_BGR2HSV)
        bottom  = np.array([90, 105, 80])
        top  = np.array([133, 255, 255])
        filtered = cv2.inRange(source, bottom , top )
        return self.morphologicalOps(filtered)

    def getLogo(self):
        return self.refineIsolation()
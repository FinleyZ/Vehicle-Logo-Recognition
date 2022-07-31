import cv2
import os
import sys
import src.LocalBinaryPatternUtil as Util
import matplotlib.pyplot as plt


class Classifier:

    def __init__(self, cache, paramters , knownCarMakeList):
        self.cache = cache
        self.paramters  = paramters 
        self.knownCarMakeList = knownCarMakeList

        self.prevGuess = ""
        self.prevMinDiff = 0
        self.prevLBP = []

    def guessBrand(self, srcImg):
        histogram, self.prevLBP = Util.computeLBPHistogram(srcImg, self.paramters )
        currMin = cv2.compareHist(self.cache.histogram[0], histogram, cv2.HISTCMP_BHATTACHARYYA)
        print('brand:%s curr min difference:%f' % (self.cache.brand[0], currMin))
        guess = ""
        for IDX in range(1, len(self.cache.histogram), 1):
            dif = cv2.compareHist(self.cache.histogram[IDX], histogram, cv2.HISTCMP_BHATTACHARYYA)
            if currMin > dif:
                currMin = dif
                guess = self.cache.brand[IDX]
                print('brand:%s curr min difference:%f' % (guess, currMin))

        if guess == "":
            guess=self.cache.brand[0]
        self.prevGuess = guess
        self.prevMinDiff = currMin
        return guess, self.prevLBP


    def printGuess(self):
        if self.prevGuess == "":
            print("Something went wrong, unable to identify the brand")
        else:
            print("The vehicle brand is {}".format(self.prevGuess))
        # reset cache
        self.prevGuess = ""
        self.prevMinDiff = 0
        self.prevLBP = []


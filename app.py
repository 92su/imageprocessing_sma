"""
Created on 20 December 2022

@author su mon aung https://www.geeksforgeeks.org/python-blood-cell-identification-using-image-processing/

"""

import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import time
import cv2
from PIL import Image
import matplotlib.pyplot as plt


#st.title("Digital Image Processing Using Python")

st.set_page_config(layout="wide")


def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Image Processing','Video Processing')
    )

    if selected_box == 'Welcome':
        welcome()
    if selected_box == 'Image Processing':
        photo()
    if selected_box == 'Video Processing':
        video()


def welcome():

    st.title('Digital Image Processing using Python')


    st.subheader('Detection of White Blood Cell and Red Blood Cell is very useful for various medical applications, like counting of WBC, disease diagnosis, etc. Circle detection is the most suitable approach.'
             + 'This article is the implementation of suitable image segmentation and feature extraction techniques for blood cell identification, on the obtained enhanced images. ' +
             'For explaining the working and use of Image Enhancement and Edge Detection, this article is using the image')


col1,col2,col3,col4,col5,col6,col7,col8 = st.columns(8)
def photo():

    # read original image
    image = cv2.imread('cell.png')
    col1.image(image,caption='Original Cell Image')


    # convert image to gray scale image

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Gray.png',gray)
    col2.image(gray,caption='Gray Scale Image')

    # apply medain filter for smoothing

    blurM = cv2.medianBlur(gray,5)
    cv2.imwrite('BlurredM.png',blurM)
    col3.image(blurM,caption='Blurred Image M')

    # apply gaussian filter for smoothing
    blurG = cv2.GaussianBlur(gray,(9,9),0)
    cv2.imwrite('BlurredG.png',blurG)
    col4.image(blurG,caption='Blurred Image G')

    # histogram equalization
    histoNorm = cv2.equalizeHist(gray)
    cv2.imwrite('HistoNorm.png',histoNorm)





    #create a CLAHE object for
    # contrast limited adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit = 2.0,tileGridSize=(8,8))
    claheNorm = clahe.apply(gray)
    cv2.imwrite('ClaheNorm.png',claheNorm)
    col5.image(claheNorm,caption='Clahe Image')





    # contrast stretching

    def pixelVal(pix,r1,s1,r2,s2):
        if (0 <= pix and pix <= r1):
            return (s1/r1) * pix
        elif (r1< pix and pix <=r2):
            return ((s2-s1)/(r2-r1))*(pix-r1)+s1
        else:
            return ((255-s2)/(255-r2))*(pix-r2)+s2

    # define parameter.
    r1 = 70
    s1 = 0
    r2 = 200
    s2 = 255

    # vectorize the function to apply it to each value in the Numpy arrary
    pixelVal_vec = np.vectorize(pixelVal)

    # Apply contrast stretching.
    contrast_stretched = pixelVal_vec(gray,r1,s1,r2,s2)
    contrast_stretched_blurM = pixelVal_vec(blurM,r1,s1,r2,s2)

    cv2.imwrite('contrast_stretch.png',contrast_stretched)
    cv2.imwrite('contrast_stretched_blurM.png',contrast_stretched_blurM)

    # edge detection using canny edge detector
    edge = cv2.Canny(gray,100,200)
    cv2.imwrite('edge.png',edge)
    col6.image(edge,caption='Canny Edge Image')

    edgeG = cv2.Canny(blurG,100,200)
    cv2.imwrite('edgeG.png',edgeG)
    col7.image(edgeG,caption='Blurred G Edge Image')

    edgeM = cv2.Canny(blurM,100,200)
    cv2.imwrite('edgeM.png',edgeM)
    col8.image(edgeM,caption='Blurred M Edge Image')


col1,col2,col3,col4,col5,col6,col7,col8 = st.columns(8)
def video():

    img = cv2.imread('Gray.png',0)
    col1.image(img,caption='Gray Scale Image')

    #morphological operations
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations =10)
    closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #x =st.slider('Change Threshold Value',min_value=10,max_value=255)
    ret4,th4 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th4 = th4.astype(np.float64)
    col2.image(th4,clamp=True,caption='Thresholded Image')

    # initialize the list
    Cell_count,x_count,y_count = [],[],[]

    # read original image, to display the circel and center Detectiondisp
    display = cv2.imread('cell.png')

    #hough transform with modified circular parameters
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 20,
                               param1 = 50, param2 = 28, minRadius = 1, maxRadius = 20)
    # Circle detection and labeling using hough transformation

#    if circles is not None:
        # convert the (x,y )coordinates and radius of the circels to integers
#        circles = np.round(circles[0,:]).astype("int")

        # Loop over the (x,y) coordinates and radius of the HoughCircle
#        for (x,y,r) in circles:
#            cv2.circle(display,(x,y),r,(0,255,0),2)
#            cv2.rectangle(display,(x-2,y-2),(x+2,y+2),(0,128,255),-1)

#            Cell_count.append(r)
#            x_count.append(x)
#            y_count.append(y)
#            # show the output image
#    cv2.imshow('detected circles',display)  #https://github.com/opencv/opencv/issues/22602
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    # display the count of white blood cells
    print(len(Cell_count))
    # total number of radius
    print(Cell_count)
    # X coordinate of circles
    print(x_count)

    # Y coordinate of circle
    print(y_count)


if __name__ == "__main__":
    main()

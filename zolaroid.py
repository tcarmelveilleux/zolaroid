# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 19:59:58 2017

@author: tennessee
"""

import cv2
import time
import math
import numpy as np
import random

FRAME_OFF_X_IN = 0.54
FRAME_OFF_Y_IN = 0.65
FRAME_WIDTH_IN = 2.75
FRAME_HEIGHT_IN = 4.6
FRAME_DPI = 203
FRAME_MIN_MARGIN_IN = 0.08

def viewfinder_bounds(img, height_ratio, aspect_ratio):
    im_height, im_width, _ = img.shape
    cap_height = int(im_height * height_ratio)
    cap_width = int(cap_height * aspect_ratio)
    
    ul = ((im_width / 2) - (cap_width / 2), (im_height / 2) - (cap_height / 2))
    br = (ul[0] + cap_width - 1, ul[1] + cap_height - 1)
    
    return cap_width, cap_height, ul, br

def extract_rectangle(img, ul, br):
    return img[ul[1]:(br[1]+1), ul[0]:(br[0]+1)]

def capture_picture(cam_id=0, mirror=True, rotate=90, aspect_ratio=4.0/3.0, height_ratio=0.5, **kwargs):
    cam = cv2.VideoCapture(cam_id)
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    
    if "exposure" in kwargs:
        cam.set(cv2.CAP_PROP_EXPOSURE, kwargs["exposure"])
        
    if "contrast" in kwargs:
        cam.set(cv2.CAP_PROP_CONTRAST, kwargs["contrast"])
    
    # brightness 60-300, 60 default
    if "brightness" in kwargs:
        cam.set(cv2.CAP_PROP_BRIGHTNESS, kwargs["brightness"])
    
    img_src = cv2.imread("test_img.jpg")
    
    while True:
        #ret_val, img = cam.read()
        img = img_src[:,:]         
        if rotate == 90:
            img = cv2.transpose(img)

        if mirror: 
            img = cv2.flip(img, -1)

        # Extract a rectangle that has ratio `max_dim` of the height and is sized
        # according to `aspect_ratio`
        cap_width, cap_height, ul, br = viewfinder_bounds(img, height_ratio, aspect_ratio)
        captured = extract_rectangle(img, ul, br)
    
        # Preview is always max 480 pixels to fit the screen easy
        preview_height = 480
        preview_width = int(preview_height * aspect_ratio)
        preview_img = cv2.resize(captured, (preview_width, preview_height))
        
        cv2.imshow('Webcam Capture. Press <SPACE> to take picture', preview_img)
        # Space to capture
        if cv2.waitKey(1) == 32: 
            break
            
    cv2.destroyAllWindows()
    
    return captured

def sat_add(a, b):
    the_sum = a + b
    if the_sum > 255:
        return 255
    elif the_sum < 0:
        return 0
    else:
        return the_sum

def dither_floyd_steinberg(img):
    # Adapted from
    # https://en.wikipedia.org/wiki/Floyd-Steinberg_dithering
    # https://github.com/kehribar/Dithering-OpenCV/blob/master/main.cpp

    width = img.shape[1]
    height = img.shape[0]
    new_img = img[:, :]
        
    for y in xrange(height):
        for x in xrange(width):
            if new_img[y, x] > 127:
                err = np.uint8(new_img[y, x]) - 255 + random.randint(-10, 10)
                new_img[y, x] = 255
            else:
                err = np.uint8(new_img[y, x]) - 0 + random.randint(-10, 10)
                new_img[y, x] = 0

            a = (err * 7) / 16
            b = (err * 1) / 16
            c = (err * 5) / 16
            d = (err * 3) / 16
              
            if (y != (height - 1)) and (x != 0) and (x != (width - 1)):
                new_img[y  ,x+1] = sat_add(new_img[y  ,x+1], a)
                new_img[y+1,x+1] = sat_add(new_img[y+1,x+1], b)
                new_img[y+1,x  ] = sat_add(new_img[y+1,x  ], c)
                new_img[y+1,x-1] = sat_add(new_img[y+1,x-1], d)
            
    
    return new_img

def blit_centered(dst, src):
    src_height, src_width = src.shape[0:2]
    dst_height, dst_width = dst.shape[0:2]
    
    ul = ((dst_height / 2)-(src_height / 2), (dst_width / 2)-(src_width / 2))
    lr = ((dst_height / 2)+(src_height / 2), (dst_width / 2)+(src_width / 2))
    
    dst[ul[0]:lr[0], ul[1]:lr[1]] = src[:, :]

def blit_offset(dst, src, offset):
    offset_x, offset_y = offset
    
    src_height, src_width  = src.shape[0:2]
    dst_height, dst_width  = dst.shape[0:2]
    
    ul = (offset_y, offset_x)
    lr = (offset_y + src_height, offset_x + src_width)
    
    dst[ul[0]:lr[0], ul[1]:lr[1]] = src[:, :]

class ZolaroidProcessor(object):
    def __init__(self, source, paper_width_in=4.0, paper_height_in=6.0,
                 frame_width_in=FRAME_WIDTH_IN, frame_height_in=FRAME_HEIGHT_IN,
                 printer_dpi=203, **kwargs):
        
        self._source = source
        self._paper_width_pix = int(paper_width_in * printer_dpi) / 8 * 8
        self._paper_height_pix = int(paper_height_in * printer_dpi) / 8 * 8
        self._frame_width_pix =  int(frame_width_in * printer_dpi) / 8 * 8
        self._frame_height_pix = int(frame_height_in * printer_dpi) / 8 * 8
        self._printer_dpi = printer_dpi
        self._background = kwargs.get("background")
        
        self._frame_off_x_pix = kwargs.get("frame_off_x_in", (paper_width_in-frame_width_in) / 2.0)
        self._frame_off_x_pix = int((self._frame_off_x_pix * printer_dpi) / 8 * 8)
                
        self._frame_off_y_pix = kwargs.get("frame_off_y_in", (paper_height_in-frame_height_in) / 2.0)
        self._frame_off_y_pix = int((self._frame_off_y_pix * printer_dpi) / 8 * 8)
        
    def process_until_done(self):
        image_width = self._paper_width_pix - int(2.0 * FRAME_MIN_MARGIN_IN * self._printer_dpi)
        image_height = self._paper_height_pix - int(2.0 * FRAME_MIN_MARGIN_IN * self._printer_dpi)
        
        img = np.ones((image_height, image_width), dtype=np.uint8)
        
        if self._background is not None:
            background = cv2.imread(self._background)
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            background = cv2.resize(background, (image_width, image_height))
            background[background < 10] = 0
            background[background > 245] = 255
            
            blit_centered(img, background)

        while True:
            gray = cv2.cvtColor(self._source, cv2.COLOR_BGR2GRAY)
            lightened = cv2.multiply(gray, np.array([1.2]))
            resized = cv2.resize(lightened, (self._frame_width_pix, self._frame_height_pix))
            blit_offset(img, resized, (self._frame_off_x_pix, self._frame_off_y_pix))
            dithered = dither_floyd_steinberg(img)
            
            #preview_img = cv2.resize(img, (image_width / 2, image_height / 2))
            preview_img = dithered
            
            cv2.imshow('Frame preview. Press <SPACE> to end', preview_img)
            # Space to end
            if cv2.waitKey(1) == 32: 
                break
            
        cv2.destroyAllWindows()
    
def main():
    captured = capture_picture(cam_id=1, mirror=True, aspect_ratio=FRAME_WIDTH_IN/FRAME_HEIGHT_IN,
                               brightness=120, exposure=1, height_ratio=0.9)
    
    processor = ZolaroidProcessor(source=captured, background="backgrounds/frame1.png",
                                  frame_off_x_in=FRAME_OFF_X_IN, frame_off_y_in=FRAME_OFF_Y_IN)
    processor.process_until_done()
    
if __name__ == '__main__':
    main()
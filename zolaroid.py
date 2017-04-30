#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zebra-printer-based, MIDI-controlled Photo Booth :D

Created on Fri Apr 07 19:59:58 2017

@author: tennessee

Copyright 2017, Tennessee Carmel-Veilleux.
https://github.com/tcarmelveilleux/zolaroid
"""

import cv2
import threading
import numpy as np
import random

try:
    from midi_driver import MidiControllerDriver
    MIDI_SUPPORT = True
except:
    MIDI_SUPPORT = False

FRAME_OFF_X_IN = 0.54
FRAME_OFF_Y_IN = 0.65
FRAME_WIDTH_IN = 2.75
FRAME_HEIGHT_IN = 4.6
FRAME_DPI = 203
FRAME_MIN_MARGIN_IN = 0.08

PLAY_AXIS = 41
ALPHA_AXIS = 16
BETA_AXIS = 17
BRIGHTNESS_AXIS = ALPHA_AXIS
EXPOSURE_AXIS = BETA_AXIS


def viewfinder_bounds(img, height_ratio, aspect_ratio):
    im_height, im_width, _ = img.shape
    cap_height = int(im_height * height_ratio)
    cap_width = int(cap_height * aspect_ratio)
    
    ul = ((im_width / 2) - (cap_width / 2), (im_height / 2) - (cap_height / 2))
    br = (ul[0] + cap_width - 1, ul[1] + cap_height - 1)
    
    return cap_width, cap_height, ul, br


def extract_rectangle(img, ul, br):
    return img[ul[1]:(br[1]+1), ul[0]:(br[0]+1)]


class PictureCapture(object):
    def __init__(self, **kwargs):
        self._lock = threading.Lock()
        self._done = False
        self._brightness = kwargs.get("brightness", 60)
        self._exposure = kwargs.get("exposure", 0)
        
    def handle_event(self, event):
        if event.get("event", "") == "control_change":
            with self._lock:
                if event["control"] == PLAY_AXIS:
                    self._done = True
                elif event["control"] == BRIGHTNESS_AXIS:
                    self._brightness = 60 + int(event["value"] * 240)
                    print "Set brightness to %d" % self._brightness
                elif event["control"] == EXPOSURE_AXIS:
                    self._exposure = -6 + int((event["value"] * 7.0) + 0.001)
                    print "Set exposure to %d" % self._exposure
        
    def capture(self, cam_id=0, mirror=True, rotate=90, aspect_ratio=4.0/3.0, height_ratio=0.5, **kwargs):
        cam = cv2.VideoCapture(cam_id)
        
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        
        if "contrast" in kwargs:
            cam.set(cv2.CAP_PROP_CONTRAST, kwargs["contrast"])
        
        #img_src = cv2.imread("test_img.jpg")
        
        while True:
            with self._lock:
                brightness = self._brightness
                exposure = self._exposure
                
            # brightness 60-300, 60 default
            if brightness is not None:
                cam.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            
            if exposure is not None:
                cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
            
            ret_val, img = cam.read()
            #img = img_src[:,:]         
            if rotate == 90:
                img = cv2.transpose(img)
    
            if mirror: 
                img = cv2.flip(img, -1)
    
            # Extract a rectangle that has ratio `max_dim` of the height and is sized
            # according to `aspect_ratio`
            cap_width, cap_height, ul, br = viewfinder_bounds(img, height_ratio, aspect_ratio)
            captured = extract_rectangle(img, ul, br)
        
            # Preview is always max 480 pixels to fit the screen easy
            preview_height = 600
            preview_width = int(preview_height * aspect_ratio)
            preview_img = cv2.resize(captured, (preview_width, preview_height))
            
            cv2.imshow('Capture. <SPACE>/PLAY to take grab', preview_img)
            
            # Space or PLAY midi button to capture
            with self._lock:
                done = self._done
                
            if done or cv2.waitKey(1) == 32: 
                break
                
        cv2.destroyAllWindows()
        
        return captured


def dither_floyd_steinberg(img):
    # Adapted from
    # https://en.wikipedia.org/wiki/Floyd-Steinberg_dithering
    # https://github.com/kehribar/Dithering-OpenCV/blob/master/main.cpp

    width = img.shape[1]
    height = img.shape[0]
    new_img = img[:, :]
        
    for y in xrange(height):
        if (y % 10) == 0:
            print("Floyd steinberg %d%%" % (y * 100 / height))
            
        for x in xrange(width):
            if new_img[y, x] > 127:
                err = np.uint8(new_img[y, x]) - 255
                new_img[y, x] = 255
            else:
                err = np.uint8(new_img[y, x]) - 0
                new_img[y, x] = 0

            a = (err * 7) / 16
            b = (err * 1) / 16
            c = (err * 5) / 16
            d = (err * 3) / 16
              
            if (y != (height - 1)) and (x != 0) and (x != (width - 1)):
                w = new_img[y  ,x+1] + a
                w = 255 if w > 255 else (0 if w < 0 else w)
                new_img[y  ,x+1] = w
                
                w = new_img[y+1,x+1] + b
                w = 255 if w > 255 else (0 if w < 0 else w)
                new_img[y+1,x+1] = w
                
                w = new_img[y+1,x  ] + c
                w = 255 if w > 255 else (0 if w < 0 else w)
                new_img[y+1,x  ] = w
                
                w = new_img[y+1,x-1] + d
                w = 255 if w > 255 else (0 if w < 0 else w)
                new_img[y+1,x-1] = w
            
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


def convert_to_epl_string(img, offset_x_pix, offset_y_pix):
    """
    Syntax GWp1,p2,p3,p4,DATA
    Parameters This table identifies the parameters for this format:
    Parameters Details
    p1 = Horizontal start
    position
    Horizontal start position (X) in dots.
    p2 = Vertical start position Vertical start position (Y) in dots.
    p3 = Width of graphic Width of graphic in bytes. Eight (8) dots = one (1) byte of
    data.
    p4 = Length of graphic Length of graphic in dots (or print lines)
    DATA Raw binary data without graphic file formatting. Data must
    be in bytes. Multiply the width in bytes (p3) by the number of
    print lines (p4) for the total amount of graphic data. The
    printer automatically calculates the exact size of the data
    block based upon this formula.
    """
    num_bytes_per_line = img.shape[1] / 8
    num_lines = img.shape[0]

    data_bytes = np.zeros((num_bytes_per_line * num_lines,), dtype=np.uint8)    
    for y in xrange(num_lines):
        for byte_idx in xrange(num_bytes_per_line):
            x = byte_idx * 8
            byte =  0x80 if (img[y, x + 0] > 128) else 0x00
            byte |= 0x40 if (img[y, x + 1] > 128) else 0x00
            byte |= 0x20 if (img[y, x + 2] > 128) else 0x00
            byte |= 0x10 if (img[y, x + 3] > 128) else 0x00
            byte |= 0x08 if (img[y, x + 4] > 128) else 0x00
            byte |= 0x04 if (img[y, x + 5] > 128) else 0x00
            byte |= 0x02 if (img[y, x + 6] > 128) else 0x00
            byte |= 0x01 if (img[y, x + 7] > 128) else 0x00
            
            data_bytes[y * num_bytes_per_line + byte_idx] = byte
    
    epl_cmd = "GW%d,%d,%d,%d,%s\r\n" % (offset_x_pix, offset_y_pix, num_bytes_per_line, num_lines, "".join([chr(b) for b in data_bytes]))
    
    return epl_cmd


def generate_epl_file(img, filename, **kwargs):
    print_data = { "label_length_pix": 1218, "label_width_pix": 831, "gap_length_pix": 24,
                  "speed_idx":1, "darkness": 4, "ref_x_pix": 0, "ref_y_pix": 0}
    
    print_data.update(kwargs)
    print_data["image_data"] = convert_to_epl_string(img, 0, 0)
    template_header = """    
I8,A,001


Q%(label_length_pix)d,%(gap_length_pix)03d
q%(label_width_pix)d
rN
S%(speed_idx)d
D%(darkness)d
ZT
JF
O
R%(ref_x_pix)d,%(ref_y_pix)d
f100

N
"""
    template_footer = """
P1
"""
    with open(filename, "wb+") as fout:
        fout.write(template_header % print_data)
        fout.write(print_data["image_data"])
        fout.write(template_footer % print_data)
    

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

        self._lock = threading.Lock()
        self._done = False
        self._alpha = kwargs.get("alpha", 1.0)
        self._beta = kwargs.get("beta", 0.0)
        
        self._result_image = None
        
    def handle_event(self, event):
        if event.get("event", "") == "control_change":
            with self._lock:
                if event["control"] == PLAY_AXIS:
                    self._done = True
                elif event["control"] == ALPHA_AXIS:
                    self._alpha = 0.5 + (event["value"] * (5.0 - 0.5))
                    print "Set alpha to %.3f" % self._alpha
                elif event["control"] == BETA_AXIS:
                    self._beta = 0.0 + event["value"] * 100.0
                    print "Set beta to %.3f" % self._beta
        
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
        
        gray = cv2.cvtColor(self._source, cv2.COLOR_BGR2GRAY)
        while True:
            with self._lock:
                alpha = self._alpha
                beta = self._beta
                
            adjusted = gray.copy()
            cv2.convertScaleAbs(gray, adjusted, alpha, beta)
            resized = cv2.resize(adjusted, (self._frame_width_pix, self._frame_height_pix))
            blit_offset(img, resized, (self._frame_off_x_pix, self._frame_off_y_pix))
            #dithered = dither_floyd_steinberg(img)
            
            preview_img = cv2.resize(img, (image_width / 2, image_height / 2))
            
            cv2.imshow('Adjust, <SPACE>/PLAY to print', preview_img)
            
            # Space or PLAY midi button to end
            with self._lock:
                done = self._done
                
            if done or cv2.waitKey(1) == 32: 
                self._result_image = img
                break
            
        cv2.destroyAllWindows()
        
    def print_image(self, filename, format="epl", **kwargs):
        print "Saving result image to %s (will take a while)" % filename
        dithered = dither_floyd_steinberg(self._result_image)
        # TODO: Support ZPL
        generate_epl_file(dithered, filename, **kwargs)
        print "Done saving result"
        
        if "printer" in kwargs:
            with open(filename, "rb") as infile:
                with open(kwargs["printer"], "wb") as printer:
                    data = infile.read()
                    printer.write(data)
                    print "Done printing" 

    
class Controller(object):
    def __init__(self):
        self._observers = set()
        
    def add_observer(self, observer):
        self._observers.add(observer)
        
    def remove_observer(self, observer):
        if observer in self._observers:
            self._observers.remove(observer)
    
    def handle_event(self, event):
        for observer in self._observers:
            observer.handle_event(event)
    
        
def main():
    # Init MIDI driver    
    controller = Controller()
    
    if MIDI_SUPPORT:
        axes_configs = {
            ALPHA_AXIS: {"name": "alpha", "centered": False},
            BETA_AXIS: {"name": "beta", "centered": False},
            PLAY_AXIS:{"name": "button_play", "centered": False, "button_down_only": True}
        }
        # midi_port = None # Default
        midi_port = "nanoKONTROL2:nanoKONTROL2 MIDI 1 20:0"
        midi_driver = MidiControllerDriver(midi_port, 0, controller.handle_event, axes_configs)
        print "MIDI Support Enabled!"
    else:
        midi_driver = None
        print "NO MIDI Support!"

    # Capture picture from webcam
    pic_capture = PictureCapture()
    controller.add_observer(pic_capture)
    captured = pic_capture.capture(cam_id=0, mirror=True, aspect_ratio=FRAME_WIDTH_IN/FRAME_HEIGHT_IN, height_ratio=0.9)
    controller.remove_observer(pic_capture)
    
    # Adjust
    processor = ZolaroidProcessor(source=captured, background="backgrounds/frame1.png",
                                  frame_off_x_in=FRAME_OFF_X_IN, frame_off_y_in=FRAME_OFF_Y_IN)
    controller.add_observer(processor)
    processor.process_until_done()
    controller.remove_observer(processor)

    # Print
    processor.print_image(filename="out.prn", format="epl", ref_x_pix=int(0.25*203), printer="/dev/usb/lp0")
    
    # Clean-up
    if midi_driver is not None:
        midi_driver.shutdown()


if __name__ == '__main__':
    main()

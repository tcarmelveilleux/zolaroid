#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zebra-printer-based, MIDI-controlled Photo Booth :D

Created on Fri Apr 07 19:59:58 2017

@author: tennessee

Copyright 2017, Tennessee Carmel-Veilleux.
https://github.com/tcarmelveilleux/zolaroid
"""
from __future__ import print_function
import os
import sys
import cv2
import threading
import numpy as np
import argparse
import tempfile

try:
    from midi_driver import MidiControllerDriver, get_midi_devices_list
    MIDI_SUPPORT = True
except:
    MIDI_SUPPORT = False

DEFAULT_WIDTH = 4.0
DEFAULT_HEIGHT = 6.0
DEFAULT_DPI = 203

FRAME_OFF_X_IN = 0.54
FRAME_OFF_Y_IN = 0.65
FRAME_WIDTH_IN = 2.75
FRAME_HEIGHT_IN = 4.6
FRAME_DPI = DEFAULT_DPI
FRAME_MIN_MARGIN_IN = 0.08

PLAY_AXIS = 41
ALPHA_AXIS = 16
BETA_AXIS = 17
BRIGHTNESS_AXIS = ALPHA_AXIS
EXPOSURE_AXIS = BETA_AXIS

RETCODE_BAD_ARG = 1
RETCODE_FILE_NOT_FOUND = 2
RETCODE_PARSE_ERROR = 3

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
                    print("Set brightness to %d" % self._brightness)
                elif event["control"] == EXPOSURE_AXIS:
                    self._exposure = -6 + int((event["value"] * 7.0) + 0.001)
                    print("Set exposure to %d" % self._exposure)
        
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
        # Print progress once in a while
        if (y % 50) == 0 or y == (height - 1):
            print("Floyd steinberg %d%%" % (y * 100 / (height - 1)))
            
        for x in xrange(width):
            if new_img[y, x] > 127:
                err = new_img[y, x] - 255
                new_img[y, x] = 255
            else:
                err = np.uint8(new_img[y, x] - 0)
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


def dither_ordered(img):
    dith_mat = np.array([[0, 8, 2, 10],[12, 4, 14, 6],[3,11,1,9],[15,7,13,5]], dtype='uint8') * 16
    width = img.shape[1]
    height = img.shape[0]
    new_img = img[:, :]

    for y in xrange(height):
        # Print progress once in a while
        if (y % 50) == 0 or (y == (height - 1)):
            print("Ordered dither %d%%" % (y * 100 / (height - 1)))

        for x in xrange(width):
            new_img[y, x] = 255 if (new_img[y, x] >= dith_mat[y&3, x&3]) else 0

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


def generate_epl_file(img, image_file, **kwargs):
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
    image_file.write(template_header % print_data)
    image_file.write(print_data["image_data"])
    image_file.write(template_footer % print_data)
    

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
                    print("Set alpha to %.3f" % self._alpha)
                elif event["control"] == BETA_AXIS:
                    self._beta = 0.0 + event["value"] * 100.0
                    print("Set beta to %.3f" % self._beta)
        
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
            
            preview_img = cv2.resize(img, (image_width / 2, image_height / 2))
            
            cv2.imshow('Adjust, <SPACE>/PLAY to print', preview_img)
            
            # Space or PLAY midi button to end
            with self._lock:
                done = self._done
                
            if done or cv2.waitKey(1) == 32: 
                self._result_image = img
                break
            
        cv2.destroyAllWindows()
        
    def print_image(self, filename=None, image_type="epl", use_floyd_steinberg=False, **kwargs):
        print("Saving result image to %s (will take a while)" % filename)

        if use_floyd_steinberg:
            dithered = dither_floyd_steinberg(self._result_image)
        else:
            dithered = dither_ordered(self._result_image)

        # TODO: Support ZPL
        if filename is None:
            output_file = tempfile.NamedTemporaryFile(delete=False)
            output_filename = output_file.name
        else:
            output_file = open(filename, "wb+")
            output_filename = filename

        with output_file:
            generate_epl_file(dithered, output_file, **kwargs)
            print("Done saving result")
        
        if kwargs.get("printer") is not None:
            try:
                if sys.platform == "win32":
                    # On win32, use the COPY /B command to print directly to a raw port. Somehow,
                    # I could not make normal IO work (as the case below), which works under Linux
                    os.system("COPY /B %s %s" % (output_filename, kwargs["printer"]))
                else:
                    # On non-windows, simply dump the data to the printer spool path
                    with open(output_filename, "rb") as infile:
                        with open(kwargs["printer"], "wb+") as printer:
                            data = infile.read()
                            printer.write(data)
                print("Done printing")
            except IOError as e:
                print("Error while trying to print: %s", str(e))


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


def print_and_die(message, retcode):
    print(message, file=sys.stderr)
    sys.exit(retcode)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--printer', metavar="PATH_TO_PRINTER", action="store", help='Path for printer which will output')
    parser.add_argument('-o', '--output', metavar="FILENAME", action="store", help='Path where to save output file')
    parser.add_argument('-t', '--type', metavar="FORMAT", action="store", default="epl", help='Printer type, one of ["zpl", "epl"]')
    parser.add_argument('-l', '--list', action="store_true", help='List MIDI devices and exit')
    parser.add_argument('-n', '--midi-name', action="store", help='Full name of MIDI input device or "default" for default')
    parser.add_argument('-m', '--midi-config', metavar="JSON_FILENAME", action="store", help='Load MIDI config from given JSON (otherwise, default to KORG NanoKontrol2)')
    parser.add_argument('-c', '--camera', metavar="CAMERA_NUM", action="store", default=0, type=int, help='Set camera number to use (default: 0)')
    parser.add_argument('-s', '--floyd-steinberg', action="store_true", default=False, help='Use Floyd-Steinberg dithering instead of ordered dithering')
    parser.add_argument('-W', '--width', metavar="WIDTH_INCHES", action="store", type=float,
                        default=DEFAULT_WIDTH, help='Set label width in inches (default:%.1f)' % DEFAULT_WIDTH)
    parser.add_argument('-H', '--height', metavar="HEIGHT_INCHES", action="store", type=float,
                        default=DEFAULT_WIDTH, help='Set label height in inches (default:%.1f)' % DEFAULT_HEIGHT)
    parser.add_argument('-d', '--dpi', metavar="DPI", action="store", type=int,
                        default=FRAME_DPI, help='Set printer DPI (default:%d)' % FRAME_DPI)

    # TODO: Add background config
    # TODO: Add MIDI input as index
    # TODO: Add MIDI test mode hook from midi_driver.py

    args = parser.parse_args()

    # Handle listing MIDI devices
    if args.list and MIDI_SUPPORT:
        print("MIDI Input devices:")
        for idx, midi_port in enumerate(get_midi_devices_list()):
            print("  * %d: '%s'" % idx, midi_port)
        sys.exit(0)

    # Make sure at least one output action is provided
    if not args.printer and not args.output:
        parser.print_usage(file=sys.stderr)
        print_and_die("Need at least one of -p/--printer or -o/--output!", RETCODE_BAD_ARG)

    return args

def main():
    args = parse_args()

    # Init MIDI driver    
    controller = Controller()
    
    if MIDI_SUPPORT and args.midi_name is not None:
        axes_configs = {
            ALPHA_AXIS: {"name": "alpha", "centered": False},
            BETA_AXIS: {"name": "beta", "centered": False},
            PLAY_AXIS:{"name": "button_play", "centered": False, "button_down_only": True}
        }

        if args.midi_name == "default":
            midi_port = None
        else:
            midi_port = args.midi_name #"nanoKONTROL2:nanoKONTROL2 MIDI 1 20:0"
        midi_driver = MidiControllerDriver(midi_port, 0, controller.handle_event, axes_configs)
        print("MIDI Support Enabled!")
    else:
        midi_driver = None
        print("NO MIDI Support!")

    # Capture picture from webcam
    pic_capture = PictureCapture()
    controller.add_observer(pic_capture)
    captured = pic_capture.capture(cam_id=args.camera, mirror=True, aspect_ratio=FRAME_WIDTH_IN/FRAME_HEIGHT_IN, height_ratio=0.9)
    controller.remove_observer(pic_capture)
    
    # Adjust
    processor = ZolaroidProcessor(source=captured, background="backgrounds/frame1.png",
                                  frame_off_x_in=FRAME_OFF_X_IN, frame_off_y_in=FRAME_OFF_Y_IN)
    controller.add_observer(processor)
    processor.process_until_done()
    controller.remove_observer(processor)

    # Print
    processor.print_image(filename=args.output, image_type=args.type, use_floyd_steinberg=args.floyd_steinberg,
                          ref_x_pix=int(0.25*203), printer=args.printer)
    
    # Clean-up
    if midi_driver is not None:
        midi_driver.shutdown()


if __name__ == '__main__':
    main()

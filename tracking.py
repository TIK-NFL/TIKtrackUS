#!/usr/bin/python3
# TIKtrackUS is a tracking Software developed for the usage in a Lecture Capture Enviroment
# Copyright (C) 2018 Brian Barnhart
# This program is free software; you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
# GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA

import cv2
import sys
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression
import time

# Load Config File
dic = {}
bbox = (0, 0, 0, 0)
fail = 0
counter = 0
file = open("config", 'r')
# Read config to dictionary
for line in file:
    key, val = line.split(',')
    print(key)
    val = val.strip('\n')
    val = val.strip('"')
    dic[key] = val
#Necessary for Pysca usage delete if not needed
dic['port'] = int(dic['port'])
dic['myport'] = int(dic['myport'])



# Detection using HOGdesicriptor
def detect():
    global bbox, image
    global y, x, w, h
    image = frame[800:1080, 1:1920]
    # image = cv2.resize(image[900:1080, 350:1520],(0,0),fx=2, fy=2,interpolation=cv2.INTER_CUBIC)
    orig = image.copy()
    # Optimization possibilities
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    y = y + 800
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.05)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    cv2.imshow("After NMS", image)
    bbox = (x, y, w, h)


# Detection using BackgroundSubstraction
def detect_fallback():
    global gray, bbox, image, fgmask, y, x, h, w
    image = frame[800:1080, 1:1920]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    fgmask = fgbg.apply(gray)
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        y = y + 800
        bbox = (x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        break
    cv2.imshow('fgmask', image)
    cv2.imshow('frame', fgmask)


if __name__ == '__main__':

    # Set up tracker.
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()

    # Read video
    # V3801
    # video = cv2.VideoCapture("rtsp://root@172.25.35.15:554/live1.sdp")
    # office
    # video = cv2.VideoCapture("rtsp://root@129.69.58.27:554/live1.sdp")
    # if stream URL is given use Stream, otherwise use file
    if dic['URI'] == '':
        video = cv2.VideoCapture(dic['file'])
    else:
        video = cv2.VideoCapture(dic['URI'])
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    # Initialise HOGDescritpro and BackgroundSubstractor
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    fgbg = cv2.createBackgroundSubtractorMOG2()
    # Crop, greyscale and gaus for background substraction
    image = frame[800:1080, 1:1920]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    fgmask = fgbg.apply(gray)
    detect_fallback()
    # if HOGDescriptor fails -> Fallback
    try:
        detect()
        cv2.imshow("After NMS", image)
        cv2.waitKey(0)
    except NameError:   # If HOGDescriptor does not find anything y = y + 800 in detect() will throw a NameError
        detect_fallback()

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    # Tracking loop
    while True:
        # Read a new frame
        check = bbox
        ok, frame = video.read()
        if not ok:
            break
        # Start timer
        timer = cv2.getTickCount()
        # Update tracker
        ok, bbox = tracker.update(frame)
        if (bbox[0] - 1) <= check[0] <= (bbox[0] + 1) and (bbox[1] - 1) <= check[1] <= (bbox[1] + 1):
            counter = counter + 1

        if counter > 15:
            detect()
            counter = 0
            tracker = cv2.TrackerKCF_create()
            ok = tracker.init(frame, bbox)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            fail = 0
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        elif fail < 5:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            try:
                detect()
            except NameError:
                pass
            tracker = cv2.TrackerKCF_create()
            ok = tracker.init(frame, bbox)
            if ok:
                fail = 0
            else:
                fail = fail + 1
        else:
            detect_fallback()
            tracker = cv2.TrackerKCF_create()
            ok = tracker.init(frame, bbox)


        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        print(counter)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

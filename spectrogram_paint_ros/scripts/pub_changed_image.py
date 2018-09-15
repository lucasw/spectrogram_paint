#!/usr/bin/env python
# load and publish an image from file, republish if the image on disk changes

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pyaudio
import rospy
import time
import sys

from cv_bridge import CvBridge, CvBridgeError
from scipy import signal
from sensor_msgs.msg import Image
from skimage import io


class PubChangedImage:
    def __init__(self):
        self.pub = rospy.Publisher("image", Image, queue_size=3)
        rospy.sleep(1.0)
        filename = rospy.get_param("~image","image.tif")
        rospy.loginfo(filename)

        self.bridge = CvBridge()
        old_stamp = None
        im = None

        while not rospy.is_shutdown():
            # try:
            if im is not None:
                cv2.imshow("im", im)
                cv2.waitKey(5)
            if True:
                stamp = os.stat(filename).st_mtime
                if stamp == old_stamp:
                    rospy.sleep(0.1)
                    continue
                old_stamp = stamp
                im = io.imread(filename)
                # cv bridge can actually handle 2 channels, but imshow can't
                if len(im.shape) > 2 and not im.shape[2] in [1, 3, 4]:
                    rospy.logwarn('warning phase image unexpected multiple layers ' +
                                  str(im.shape[2]))
                    # TODO(lucasw) collapse them or publish them on different topics?
                    im = im[:, :, 0]
                print im.shape
                msg = self.bridge.cv2_to_imgmsg(im)
                # TODO(lucasw) or convert the st_mtime to ros time
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = 'map'
                # msg.height = im.shape[0]
                # msg.width = im.shape[1]
                # msg.step = image.
                print filename, stamp, im.shape, np.min(im), np.max(im)
                self.pub.publish(msg)
            if False:  # except Exception as ex:
                rospy.logerr("something went wrong " + str(ex))
                # TODO(lucasw) catch the exception here and print something
                rospy.sleep(0.1)

if __name__ == '__main__':
    rospy.init_node('pub_changed_image')
    pub_changed_image = PubChangedImage()
    rospy.sleep()

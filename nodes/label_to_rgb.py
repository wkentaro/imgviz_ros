#!/usr/bin/env python

import imgviz

import cv_bridge
import rospy
from sensor_msgs.msg import Image
from topic_tools import LazyTransport
import message_filters


class LabelToRgb(LazyTransport):
    def __init__(self):
        super().__init__()

        self._approximate_sync = rospy.get_param("~approximate_sync", False)
        self._queue_size = rospy.get_param("~queue_size", 50)
        self._slop = rospy.get_param("~slop", 0.1)

        self._use_image = rospy.get_param("~use_image", False)
        self._alpha = rospy.get_param("~alpha", 0.5)

        self._label_offset = rospy.get_param("~label_offset", 0)
        self._label_astype = rospy.get_param("~label_astype", "int32")

        self._pub = self.advertise("~output", Image, queue_size=1)

    def subscribe(self):
        sub_label = message_filters.Subscriber(
            "~input/label", Image, queue_size=1, buff_size=2 ** 24
        )

        if self._use_image:
            sub_image = message_filters.Subscriber(
                "~input/image", Image, queue_size=1, buff_size=2 ** 24
            )
            self._subscribers = [sub_label, sub_image]
            if self._approximate_sync:
                sync = message_filters.ApproximateTimeSynchronizer(
                    self._subscribers, queue_size=self._queue_size, slop=self._slop
                )
            else:
                sync = message_filters.TimeSynchronizer(
                    self._subscribers, queue_size=self._queue_size
                )
            sync.registerCallback(self._subscribe_callback)
        else:
            self._subscribers = [sub_label]
            sub_label.registerCallback(self._subscribe_callback)

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _subscribe_callback(self, label_msg, image_msg=None):
        bridge = cv_bridge.CvBridge()

        label = bridge.imgmsg_to_cv2(label_msg)
        label = label + self._label_offset
        label = label.astype(self._label_astype)

        if image_msg is None:
            image = None
        else:
            image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")

        out = imgviz.label2rgb(label=label, image=image, alpha=self._alpha)
        out_msg = bridge.cv2_to_imgmsg(out, encoding="rgb8")
        out_msg.header = label_msg.header
        self._pub.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("label_to_rgb")
    LabelToRgb()
    rospy.spin()

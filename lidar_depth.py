import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
from time import time, sleep


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

try:
   while True:
      try:
         frames = pipeline.wait_for_frames()

         if not frames:
            continue

         depth_frame = frames.get_depth_frame() 
         
         depth_image = np.asanyarray(depth_frame.get_data())

         current_t = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
         
         cv2.imshow("Depth Camera", depth_image)

         key = cv2.waitKey(1)

         if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

         sleep(0.005)
      except KeyboardInterrupt:
            break

finally:
   pipeline.stop()

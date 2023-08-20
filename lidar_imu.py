import time
import pyrealsense2 as rs


imu_pipe = rs.pipeline()
imu_config = rs.config()

imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

imu_pipe.start(imu_config)

start_t = time.time()
previous_t = time.time() * 1000
elapsed_t = 0

num_samples = 0

try:
    while True:
        try:
            elapsed_t = time.time() - start_t
            
            if elapsed_t > 25: break
            
            motion_frames: rs.composite_frame = imu_pipe.poll_for_frames()

            if motion_frames:
                current_t = motion_frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
                
                if current_t - previous_t < 1: continue
                
                accel_frame = motion_frames[0].as_motion_frame()
                gyro_frame = motion_frames[1].as_motion_frame()

                num_samples += 1
                
                print("Number of IMU samples: ", num_samples, end="\r")
                
        except KeyboardInterrupt:
            break
finally:
    imu_pipe.stop()


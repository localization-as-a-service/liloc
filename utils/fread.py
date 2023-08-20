import os
import numpy as np


def get_timstamps(dir_path, ext=".secondary.npz"):
    seq_ts = [int(f.split(".")[0]) for f in os.listdir(dir_path) if f.endswith(ext)]
    return np.array(sorted(seq_ts))


def get_timstamps_from_images(dir_path, ext=".color.png"):
    seq_ts = [int(f.split("-")[1].split(".")[0])for f in os.listdir(dir_path) if f.endswith(ext)]
    return np.array(sorted(seq_ts))


def sample_timestamps(sequence_ts, frame_rates):
    sequence_ets = (sequence_ts - sequence_ts[0])
    frame_ids = sequence_ets * frame_rates // 1e3
    to_drop = np.where(frame_ids[1:] - frame_ids[:-1] == 0)[0]
    return np.delete(sequence_ts, to_drop)


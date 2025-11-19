import os
import numpy as np
from .lib.utils import dotify

cu_dir = os.path.dirname(__file__)

config = {
    "SEED": 7,
    "N": 128,
    "DT": 1.0,
    "THREADS": 128,
    "GPU_HEAP_SIZE_BYTES": 500 * 1024 * 1024,
    "THRESHOLD": 0.8,
    "sensor": {
        "RANGE": 35,    # Stereo camera max range
        "FOV": 0.389*np.pi,   # Lidar FoV i.e 70°
        "MISS_PROB": 0,
        "VARIANCE": [0.015**2, np.deg2rad(0.5)**2],
        "MAX_MEASUREMENTS": 100
    },
    "GROUND_TRUTH": np.load(os.path.join(cu_dir,"simulation/odom.npy")).astype(np.float64),
    "CONTROL": np.load(os.path.join(cu_dir,"simulation/control.npy")).astype(np.float64),
    "CONTROL_VARIANCE": [np.deg2rad(5.0) ** 2, 0.15 ** 2],
    "LANDMARKS": np.load(os.path.join(cu_dir,"simulation/landmarks.npy")).astype(np.float64), # landmark positions
    "MAX_LANDMARKS": 100,  # upper bound on the total number of landmarks in the environment
    "START_POSITION": np.array([2, 6, 0], dtype=np.float64)  
}
 
config = dotify(config)

config.sensor.COVARIANCE = np.diag(config.sensor.VARIANCE).astype(np.float64)
config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 7*config.MAX_LANDMARKS

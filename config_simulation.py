import os
import numpy as np
from .lib.utils import dotify

cu_dir = os.path.dirname(__file__)

config = {
    "SEED": 7,
    "N": 512,
    "DT": 1.0,
    "THREADS": 128,
    "GPU_HEAP_SIZE_BYTES": 500 * 1024 * 1024,
    "THRESHOLD": 2.3,
    "sensor": {
        "RANGE": 35,    # Stereo camera max range
        "FOV": 0.7*np.pi,   # FoV i.e 100°
        "MISS_PROB": 0.05,
        "VARIANCE": [0.15**2, np.deg2rad(1.0)**2],
        "MAX_MEASUREMENTS": 50
    },
    # MOTION MODEL (ODOMETRY)
    # Tuned based on your ZED2i /odom and /imu topics.
    # ZED reports very low covariance (1e-9), but we must keep some variance 
    # for the Particle Filter to work (otherwise all particles die).
    
    # Index 0: Rotation Noise (Angular Z)
    #   IMU says cov is ~0.0002. We use ~0.0003 (1.0 degree) to allow for wheel slip.
    #   Previous Sim Value: 5.0 degrees
    #   New Real Value: 1 degree
    
    # Index 1: Linear Noise (Linear X)
    #   ZED Odom is very precise. We reduce linear error to 5cm.
    #   Previous Sim Value: 0.15m
    #   New Real Value: 0.05m
    "CONTROL_VARIANCE": [np.deg2rad(0.0) ** 2, 0.00 ** 2],

    #"GROUND_TRUTH": np.load(os.path.join(cu_dir,"simulation/odom.npy")).astype(np.float64),
    #"CONTROL": np.load(os.path.join(cu_dir,"simulation/control.npy")).astype(np.float64),
    #"LANDMARKS": np.load(os.path.join(cu_dir,"simulation/landmarks.npy")).astype(np.float64), # landmark positions
    "MAX_LANDMARKS": 600,  # upper bound on the total number of landmarks in the environment
    "START_POSITION": np.array([0, 0, 0], dtype=np.float64)         # Will be overwritten by first Odom msg 
}
 
config = dotify(config)

config.sensor.COVARIANCE = np.diag(config.sensor.VARIANCE).astype(np.float64)
config.PARTICLES_PER_THREAD = config.N // config.THREADS
config.PARTICLE_SIZE = 6 + 8*config.MAX_LANDMARKS

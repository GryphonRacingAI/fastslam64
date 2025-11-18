#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
import traceback

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda._driver import LogicError as CudaLogicError

from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from common_msgs.msg import ConeArray, Cone

DEBUG_GPU = True  # Kept True for your debugging

# --- FastSLAM Core Library Imports ---
try:
    from .config_simulation import config
    from .lib.particle3 import FlatParticle
    from .lib.common import CUDAMemory, rescale, get_pose_estimate, resample
    from .cuda.fastslam import load_cuda_modules
except Exception as e:
    raise SystemExit(f"Failed to import FastSLAM core components: {e}")


# ---------- Helpers ----------
def make_grid(n_items: int, block_x: int):
    if block_x <= 0:
        raise ValueError("block_x must be > 0")
    grid_x = (n_items + block_x - 1) // block_x
    return (max(1, int(grid_x)), 1, 1)


def ensure_contiguous(arr: np.ndarray, dtype):
    if arr is None:
        return None
    return np.ascontiguousarray(arr, dtype=dtype)


def print_gpu_stats(prefix=""):
    if not DEBUG_GPU:
        return
    free, total = cuda.mem_get_info()
    # Optional: reduce spam by only printing if free memory is low
    # if free > 1000 * 1024 * 1024: return 
    print(f"[GPU DEBUG] {prefix} | Free: {free/1e6:.2f} MB / Total: {total/1e6:.2f} MB")


def check_and_memcpy_htod(dev_ptr, host_arr, name=""):
    if host_arr is None:
        raise RuntimeError(f"Host array for {name} is None")
    host_arr = ensure_contiguous(host_arr, host_arr.dtype)
    try:
        cuda.memcpy_htod(dev_ptr, host_arr)
        print_gpu_stats(f"HTOD {name}")
    except Exception as e:
        raise RuntimeError(f"memcpy_htod failed for {name}: {e}")


def check_and_memcpy_dtoh(host_arr, dev_ptr, name=""):
    if host_arr is None:
        raise RuntimeError(f"Host target for {name} is None")
    try:
        cuda.memcpy_dtoh(host_arr, dev_ptr)
        print_gpu_stats(f"DTOH {name}")
    except Exception as e:
        raise RuntimeError(f"memcpy_dtoh failed for {name}: {e}")


KERN_FLOAT = np.float32
WEIGHT_DTYPE = np.float32
INT_DTYPE = np.int32


# ---------- Utility Functions ----------
def quaternion_to_yaw(q_x, q_y, q_z, q_w):
    siny_cosp = 2.0 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1.0 - 2.0 * (q_y*q_y + q_z*q_z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    cy = math.cos(yaw*0.5)
    sy = math.sin(yaw*0.5)
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = sy
    q.w = cy
    return q


def xy_to_range_bearing(x, y):
    r = math.sqrt(x**2 + y**2)
    beta = math.atan2(y, x)
    return r, beta


def get_cone_color(cone_type: int):
    color = {"r":0.0, "g":0.0, "b":0.0, "a":1.0}
    if cone_type == Cone.YELLOW:
        color["r"], color["g"], color["b"] = 1.0, 1.0, 0.0
    elif cone_type == Cone.BLUE:
        color["r"], color["g"], color["b"] = 0.0, 0.0, 1.0
    elif cone_type in [Cone.ORANGE, Cone.LARGE_ORANGE]:
        color["r"], color["g"], color["b"] = 1.0, 0.5, 0.0
    else:
        # Gray for unknown
        color["r"], color["g"], color["b"] = 0.5, 0.5, 0.5
    return color


# ---------- FastSLAM Core ----------
class FastSLAMCore:
    def __init__(self, config):
        self.config = config
        self.N = int(config.N)
        self.THREADS = int(config.THREADS)
        
        # --- FIX: Robust Integer Safety ---
        # Handle missing attribute OR explicit None
        raw_limit = getattr(config, 'MAX_SENSOR_MEASUREMENTS', 100)
        self.MAX_MEASUREMENTS = 100 if raw_limit is None else int(raw_limit)

        print(f"[DEBUG] Sensor Limit set to: {self.MAX_MEASUREMENTS}")

        self.particles = FlatParticle.get_initial_particles(
            self.N, config.MAX_LANDMARKS, config.START_POSITION, sigma=0.2
        ).astype(KERN_FLOAT)

        self.cuda_modules = load_cuda_modules(
            THREADS=self.THREADS,
            PARTICLE_SIZE=int(config.PARTICLE_SIZE),
            N_PARTICLES=self.N
        )

        self.memory = CUDAMemory(config)

        check_and_memcpy_htod(self.memory.cov, ensure_contiguous(config.sensor.COVARIANCE, KERN_FLOAT), name="covariance")
        check_and_memcpy_htod(self.memory.particles, ensure_contiguous(self.particles, KERN_FLOAT), name="particles")

        try:
            init_rng_func = self.cuda_modules["predict"].get_function("init_rng")
            grid = make_grid(self.N, self.THREADS)
            init_rng_func(np.int32(config.SEED), block=(self.THREADS,1,1), grid=grid)
            cuda.Context.synchronize()
            print_gpu_stats("RNG init")
        except Exception as e:
            raise RuntimeError(f"init_rng failed: {e}")

        self.weights = np.zeros(self.N, dtype=WEIGHT_DTYPE)

    def execute_step(self, v_linear: float, omega_angular: float, measurements_rb: np.ndarray):
        # --- CRASH FIX: Truncate Measurements ---
        if measurements_rb is None or len(measurements_rb) == 0:
            measurements_rb = np.zeros((0,2), dtype=KERN_FLOAT)
        else:
            # Truncate to prevent buffer overflow
            if len(measurements_rb) > self.MAX_MEASUREMENTS:
                # Optional: warn once or periodically if this happens often
                # print(f"[WARN] Truncating cones: {len(measurements_rb)} -> {self.MAX_MEASUREMENTS}")
                measurements_rb = measurements_rb[:self.MAX_MEASUREMENTS]
                
            measurements_rb = ensure_contiguous(np.array(measurements_rb), KERN_FLOAT)
            
        N_measurements = len(measurements_rb)

        # Reset Weights
        try:
            func = self.cuda_modules["resample"].get_function("reset_weights")
            func(self.memory.particles, block=(self.THREADS,1,1), grid=make_grid(self.N,self.THREADS))
            cuda.Context.synchronize()
            print_gpu_stats("reset_weights")
        except Exception as e:
            raise RuntimeError(f"reset_weights kernel failed: {e}")

        # Prediction
        try:
            func = self.cuda_modules["predict"].get_function("predict_from_model")
            func(
                self.memory.particles,
                np.float32(omega_angular),
                np.float32(v_linear),
                np.float32(np.sqrt(self.config.CONTROL_VARIANCE[0])),
                np.float32(np.sqrt(self.config.CONTROL_VARIANCE[1])),
                np.float32(self.config.DT),
                block=(self.THREADS,1,1),
                grid=make_grid(self.N,self.THREADS)
            )
            cuda.Context.synchronize()
            print_gpu_stats("predict_from_model")
        except Exception as e:
            raise RuntimeError(f"predict_from_model failed: {e}")

        # Measurement Update
        if N_measurements > 0:
            try:
                check_and_memcpy_htod(self.memory.measurements, measurements_rb, name="measurements")
                block_size = min(self.N,32)
                func = self.cuda_modules["update"].get_function("update")
                func(
                    self.memory.particles,
                    np.int32(self.N//block_size),
                    self.memory.scratchpad,
                    np.int32(self.memory.scratchpad_block_size),
                    self.memory.measurements,
                    np.int32(self.N),
                    np.int32(N_measurements),
                    np.float32(self.config.sensor.COVARIANCE[0,0]),
                    np.float32(self.config.THRESHOLD),
                    np.float32(self.config.sensor.RANGE),
                    np.float32(self.config.sensor.FOV),
                    np.int32(self.config.MAX_LANDMARKS),
                    block=(block_size,1,1),
                    grid=make_grid(self.N, block_size)
                )
                cuda.Context.synchronize()
                print_gpu_stats("update kernel")
            except Exception as e:
                raise RuntimeError(f"update kernel or HtoD failed: {e}\n{traceback.format_exc()}")

        # Rescale & Pose Estimate
        try:
            rescale(self.cuda_modules, self.config, self.memory)
            estimated_pose = get_pose_estimate(self.cuda_modules, self.config, self.memory)
            cuda.Context.synchronize()
            print_gpu_stats("get_pose_estimate")
        except Exception as e:
            raise RuntimeError(f"get_pose_estimate failed: {e}")

        # Compute Neff & Resample
        try:
            func = self.cuda_modules["weights_and_mean"].get_function("get_weights")
            func(self.memory.particles, self.memory.weights, block=(self.THREADS,1,1), grid=make_grid(self.N,self.THREADS))
            cuda.Context.synchronize()
            check_and_memcpy_dtoh(self.weights, self.memory.weights, name="weights")
            neff = FlatParticle.neff(self.weights.astype(np.float64))
            if neff < 0.6*self.N:
                resample(self.cuda_modules, self.config, self.weights, self.memory, 0.5)
                cuda.Context.synchronize()
                print_gpu_stats("resample due to low Neff")
        except Exception as e:
            raise RuntimeError(f"weights_and_mean / resample failed: {e}")

        # Extract Map
        try:
            check_and_memcpy_dtoh(self.particles, self.memory.particles, name="particles")
            best_idx = int(np.argmax(FlatParticle.w(self.particles)))
            best_landmarks = FlatParticle.get_landmarks(self.particles, best_idx)
            best_covariances = FlatParticle.get_covariances(self.particles, best_idx)
        except Exception as e:
            raise RuntimeError(f"postprocessing particles failed: {e}")

        final_map_data = []
        for lm, cov in zip(best_landmarks, best_covariances):
            if not np.all(cov == 0) and not np.linalg.norm(lm) > 0.01:
                final_map_data.append({"position": lm, "covariance": cov})

        return estimated_pose, final_map_data


# ---------- ROS 2 Node ----------
class FastSLAMNode(Node):
    def __init__(self):
        super().__init__("fastslam_node")
        self.get_logger().info("FastSLAM Node Initializing...")

        self.slam_core = FastSLAMCore(config)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, "/slam/pose", 1)
        self.map_pub = self.create_publisher(MarkerArray, "/slam/cone_map", 1)
        
        self.odom_sub = Subscriber(self, Odometry, "/odom")
        self.cone_sub = Subscriber(self, ConeArray, "/perfect_cone_array")

        self.ts = ApproximateTimeSynchronizer([self.odom_sub, self.cone_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.slam_callback)
        self.get_logger().info("Subscribers synchronized. Waiting for data...")

    def slam_callback(self, odom_msg: Odometry, cone_array_msg: ConeArray):
        v_linear = odom_msg.twist.twist.linear.x
        omega_angular = odom_msg.twist.twist.angular.z

        measurements_rb, cone_types = self._process_cones(cone_array_msg)
        
        if v_linear == 0.0 and omega_angular == 0.0 and len(measurements_rb) == 0:
            return

        try:
            estimated_pose, final_map_data = self.slam_core.execute_step(v_linear, omega_angular, measurements_rb)
        except Exception as e:
            self.get_logger().error(f"SLAM execution error: {e}")
            self.get_logger().debug(traceback.format_exc())
            return

        self._publish_transforms(odom_msg, estimated_pose)
        self._publish_map_markers(final_map_data)
        self._publish_pose_stamped(estimated_pose, odom_msg.header.stamp)

    def _process_cones(self, cone_array_msg: ConeArray):
        all_cones_rb = []
        all_cone_types = []
        cone_mapping = [
            (cone_array_msg.yellow_cones, Cone.YELLOW),
            (cone_array_msg.blue_cones, Cone.BLUE),
            (cone_array_msg.orange_cones, Cone.ORANGE),
            (cone_array_msg.large_orange_cones, Cone.LARGE_ORANGE),
            (cone_array_msg.unknown_cones, Cone.UNKNOWN),
        ]
        for cone_list, cone_type in cone_mapping:
            for cone in cone_list:
                x = cone.position.x
                y = cone.position.y
                if abs(x) < 1e-3 and abs(y) < 1e-3:
                    continue
                r, beta = xy_to_range_bearing(x, y)
                all_cones_rb.append([r, beta])
                all_cone_types.append(cone_type)
        return np.array(all_cones_rb, dtype=KERN_FLOAT), all_cone_types

    def _publish_transforms(self, odom_msg: Odometry, estimated_pose: np.ndarray):
        x_map, y_map, theta_map = estimated_pose[0], estimated_pose[1], estimated_pose[2]
        odom_x = odom_msg.pose.pose.position.x
        odom_y = odom_msg.pose.pose.position.y
        odom_q = odom_msg.pose.pose.orientation
        odom_theta = quaternion_to_yaw(odom_q.x, odom_q.y, odom_q.z, odom_q.w)

        # Calculate Map -> Odom transform
        # T_map_odom = T_map_base * T_base_odom
        delta_theta = theta_map - odom_theta
        
        t_x = x_map - (odom_x * math.cos(delta_theta) - odom_y * math.sin(delta_theta))
        t_y = y_map - (odom_x * math.sin(delta_theta) + odom_y * math.cos(delta_theta))
        
        q_corr_msg = yaw_to_quaternion(delta_theta)

        t = TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = t_x
        t.transform.translation.y = t_y
        t.transform.translation.z = 0.0
        t.transform.rotation = q_corr_msg
        self.tf_broadcaster.sendTransform(t)

    def _publish_pose_stamped(self, estimated_pose: np.ndarray, timestamp):
        x, y, theta = estimated_pose[0], estimated_pose[1], estimated_pose[2]
        q_msg = yaw_to_quaternion(theta)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.orientation = q_msg
        self.pose_pub.publish(pose_msg)

    def _publish_map_markers(self, final_map_data):
        marker_array = MarkerArray()
        for i, lm_data in enumerate(final_map_data):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cones"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(lm_data["position"][0])
            marker.pose.position.y = float(lm_data["position"][1])
            marker.pose.position.z = 0.5
            
            # Use generic color for map landmarks to avoid flickering
            cone_type = Cone.UNKNOWN 
            
            color = get_cone_color(cone_type)
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 1.0
            marker.color.a = color["a"]
            marker.color.r = color["r"]
            marker.color.g = color["g"]
            marker.color.b = color["b"]
            marker_array.markers.append(marker)
        self.map_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = FastSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("SLAM node shut down by user.")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
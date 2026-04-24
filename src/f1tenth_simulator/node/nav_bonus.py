#!/usr/bin/env python
from __future__ import print_function
from lib2to3.pytree import Node
import sys
import math
from tokenize import Double
import numpy as np
import time

from  numpy import array, dot
from quadprog import solve_qp
#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


# realsense depth frame: z forward, x right, y down
# base_link: x forward, y left, z up
# so: base_x = cam_z, base_y = -cam_x, base_z = -cam_y
R_OPTICAL_TO_BASE = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)


class GapBarrier:
    def __init__(self):
     
        self.kp = rospy.get_param("~k_p")
        self.kd = rospy.get_param("~k_d")
        self.desired_velocity = rospy.get_param("~vehicle_velocity")
        self.delta_theta_deg = rospy.get_param("~delta_theta_deg")
        self.d_stop = rospy.get_param("~d_stop")
        self.d_tau = rospy.get_param("~d_tau")
        self.delta_max = rospy.get_param("~delta_max")

        self.wheelbase = rospy.get_param("~wheelbase")
        self.safe_distance = rospy.get_param("~safe_distance")

        lidarscan_topic = rospy.get_param("~scan_topic")
        odom_topic = rospy.get_param("~odom_topic")
        drive_topic = rospy.get_param("~nav_drive_topic")

        # Initialize variables
        self.vel = 0.5
        self.fov_angle_deg = rospy.get_param("~fov_angle_deg")
        self.barrier_bl_deg = rospy.get_param("~barrier_bl_deg")
        self.barrier_al_deg = rospy.get_param("~barrier_al_deg")
        self.barrier_br_deg = rospy.get_param("~barrier_br_deg")
        self.barrier_ar_deg = rospy.get_param("~barrier_ar_deg")
        self.n_pts_r = rospy.get_param("~n_pts_r")
        self.n_pts_l = rospy.get_param("~n_pts_l")

        # camera stuff for bonus
        self.use_camera = bool(rospy.get_param("~use_camera", 0))

        # params are in mm so divide by 1000

        self.min_depth = float(rospy.get_param("~min_depth", 100.0))  / 1000.0
        self.max_depth = float(rospy.get_param("~max_depth", 5000.0)) / 1000.0

        self.roi_u_lower    = int(rospy.get_param("~roi_u_lower",   250))
        self.roi_u_upper    = int(rospy.get_param("~roi_u_upper",   548))
        self.roi_v_lower    = int(rospy.get_param("~roi_v_lower",   150))
        self.roi_v_upper    = int(rospy.get_param("~roi_v_upper",   470))
        self.depth_u_stride = int(rospy.get_param("~depth_u_stride",  4))
        self.depth_v_stride = int(rospy.get_param("~depth_v_stride",  4))

        cam_tx = float(rospy.get_param("~cam_tx", 0.292))
        cam_ty = float(rospy.get_param("~cam_ty", 0.0  ))
        cam_tz = float(rospy.get_param("~cam_tz", 0.086))
        self.t_cam_to_base = np.array([[cam_tx], [cam_ty], [cam_tz]], dtype=np.float64)

        self.x_base_min = float(rospy.get_param("~x_base_min",  0.05))
        self.x_base_max = float(rospy.get_param("~x_base_max",  2.5 ))
        self.y_base_min = float(rospy.get_param("~y_base_min", -0.60))
        self.y_base_max = float(rospy.get_param("~y_base_max",  0.60))
        self.z_base_min = float(rospy.get_param("~z_base_min",  0.03))  # anything below this is probably the floor
        self.z_base_max = float(rospy.get_param("~z_base_max",  0.60))

        self.camera_hold_time     = float(rospy.get_param("~camera_hold_time",     0.25))
        self.camera_range_padding = float(rospy.get_param("~camera_range_padding", 0.03))

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.has_intrinsics = False

        self.cam_ranges     = np.empty(0, dtype=np.float64)
        self.cam_angles     = np.empty(0, dtype=np.float64)
        self.last_cam_stamp = None

        # Add your subscribers for LiDAR scan and Odometry here
        rospy.Subscriber(lidarscan_topic, LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)

        if self.use_camera:
            cam_info_topic = rospy.get_param("~camera_info", "/camera/depth/camera_info")
            depth_topic    = rospy.get_param("~img_raw",     "/camera/depth/image_rect_raw")
            rospy.Subscriber(cam_info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
            rospy.Subscriber(depth_topic,    Image,      self.depth_callback,    queue_size=1)

        # Add your publisher for Drive topic here
        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=1)

    
    def get_front_distance(self, scan_msg):
        half_angle = math.radians(self.delta_theta_deg)
        min_dist = 999.0

        for i, r in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            if abs(angle) <= half_angle:
                if not math.isinf(r) and not math.isnan(r):
                    min_dist = min(min_dist, r)

        return min_dist

    # Optional function to pre-process LiDAR by considering only those LiDAR returns within a FOV in front of the vehicle
    def get_best_direction(self, scan_msg):
        
        fov_half_angle = math.radians(self.fov_angle_deg)

        fov_ranges = []
        fov_angles = []

        for i, r in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment

            if abs(angle) <= fov_half_angle:
                if math.isinf(r) or math.isnan(r) or r <= 0.0:
                    r_use = 0.0
                elif r < self.safe_distance:
                    r_use = 0.0
                else:
                    r_use = r

                fov_ranges.append(r_use)
                fov_angles.append(angle)

        if len(fov_ranges) == 0:
            return 0

        best_start = 0
        best_len = 0
        curr_start = 0
        curr_len = 0

        for i, r in enumerate(fov_ranges):
            if r > 0.0:
                curr_len += 1
                if curr_len > best_len:
                    best_len = curr_len
                    best_start = curr_start
            else:
                curr_start = i + 1
                curr_len = 0

        if best_len == 0:
            return 0

        gap_ranges = fov_ranges[best_start:best_start + best_len]
        gap_angles = fov_angles[best_start:best_start + best_len]

        max_idx = int(np.argmax(gap_ranges))
        theta_des = gap_angles[max_idx]

        return theta_des

    def angle_in_relative_sector(self, angle, theta_des, low_rel, high_rel):
        rel = angle - theta_des
        while rel > math.pi: rel -= 2 * math.pi
        while rel < -math.pi: rel += 2 * math.pi
        return low_rel <= rel <= high_rel

    def laser_to_base_link_xy(self, r, angle):
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        return x, y

    def solve_barriers(self, scan_msg, theta_des):

        # Relative sector limits around theta_des
        # Right sector: from br to ar relative to forward/right reference
        right_low  = math.radians(self.barrier_br_deg - 180.0)  # 90-180 = -90
        right_high = math.radians(self.barrier_ar_deg - 180.0)  # 140-180 = -40

        # Left sector: from al to bl relative to forward/left reference
        left_low   = math.radians(self.barrier_al_deg - 180.0)  # 220-180 = +40
        left_high  = math.radians(self.barrier_bl_deg - 180.0)  # 270-180 = +90

        pts_r = []
        pts_l = []

        for i, r in enumerate(scan_msg.ranges):
            if math.isinf(r) or math.isnan(r) or r <= 0.0:
                continue

            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            x, y = self.laser_to_base_link_xy(r, angle)

            if self.angle_in_relative_sector(angle, theta_des, right_low, right_high):
                pts_r.append([x, y])

            if self.angle_in_relative_sector(angle, theta_des, left_low, left_high):
                pts_l.append([x, y])

        if len(pts_r) == 0 or len(pts_l) == 0:
            rospy.loginfo("Not enough points for virtual barriers")
            return None, None

        pts_r = np.array(pts_r)
        pts_l = np.array(pts_l)

        def subsample(pts, n):
            idx = np.linspace(0, len(pts)-1, n, dtype=int)
            return pts[idx]

        pts_r = subsample(pts_r, self.n_pts_r)
        pts_l = subsample(pts_l, self.n_pts_l)

        

        nr = pts_r.shape[0]
        nl = pts_l.shape[0]

        rospy.loginfo("sample pts_r[0]=%.3f,%.3f pts_l[0]=%.3f,%.3f", pts_r[0][0], pts_r[0][1], pts_l[0][0], pts_l[0][1])

        rospy.loginfo("pts_r count=%d pts_l count=%d right_low=%.3f right_high=%.3f left_low=%.3f left_high=%.3f theta_des=%.3f", len(pts_r), len(pts_l), right_low, right_high, left_low, left_high, theta_des)

        # QP:
        # min 1/2 x^T G x - a^T x
        # s.t. C^T x >= b
        #
        # x = [w1, w2, s]^T

        G = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0001]
        ], dtype=float)

        a = np.zeros(3, dtype=float)

        # Right obstacle constraints: w^T p_i + s >= 1
        C_r = np.vstack([
            pts_r[:, 0],
            pts_r[:, 1],
            np.ones(nr)
        ])

        # Left obstacle constraints: -(w^T p_j + s) >= 1
        C_l = np.vstack([
            -pts_l[:, 0],
            -pts_l[:, 1],
            -np.ones(nl)
        ])

        # Keep origin between barriers: -1 + eps <= s <= 1 - eps
        # For quadprog C^T x >= b:
        #   s >= -0.99
        #  -s >= -0.99   -> s <= 0.99
        C_s_lower = np.array([[0.0], [0.0], [1.0]])
        C_s_upper = np.array([[0.0], [0.0], [-1.0]])

        C = np.hstack([C_r, C_l, C_s_lower, C_s_upper])

        b = np.concatenate([
            np.ones(nr),
            np.ones(nl),
            np.array([-0.99, -0.99])
        ])

        try:
            sol = solve_qp(G, a, C, b, 0)[0]
            w = sol[:2]
            s = sol[2]
            return w, s
        
        except Exception as e:
            rospy.loginfo("QP failed: %s", str(e))
            return None, None

    def cam_info_callback(self, msg):
        # only need this once
        if self.has_intrinsics:
            return
        self.fx = float(msg.K[0])
        self.fy = float(msg.K[4])
        self.cx = float(msg.K[2])
        self.cy = float(msg.K[5])
        self.has_intrinsics = True

    def depth_callback(self, msg):
        if not self.has_intrinsics:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "Depth decode error: %s", exc)
            return

        if img is None or img.size == 0:
            return

        # realsense gives uint16 in mm, but some modes give float in metres
        # if max > 100 its mm
        depth_raw = img.astype(np.float64)
        if msg.encoding in ("16UC1", "mono16") or depth_raw.max() > 100.0:
            depth_m = depth_raw / 1000.0
        else:
            depth_m = depth_raw

        # only look at the middle chunk of the image, full res is too slow
        h, w = depth_m.shape[:2]
        u0 = max(self.roi_u_lower, 0);  u1 = min(self.roi_u_upper, w)
        v0 = max(self.roi_v_lower, 0);  v1 = min(self.roi_v_upper, h)

        u_idx = np.arange(u0, u1, max(self.depth_u_stride, 1))
        v_idx = np.arange(v0, v1, max(self.depth_v_stride, 1))
        if u_idx.size == 0 or v_idx.size == 0:
            return

        uu, vv = np.meshgrid(u_idx, v_idx)
        z_raw  = depth_m[vv, uu]

        valid = np.isfinite(z_raw) & (z_raw >= self.min_depth) & (z_raw <= self.max_depth)
        if not np.any(valid):
            self.cam_ranges     = np.empty(0, dtype=np.float64)
            self.cam_angles     = np.empty(0, dtype=np.float64)
            self.last_cam_stamp = msg.header.stamp
            return

        z_c = z_raw[valid]
        u_v = uu[valid].astype(np.float64)
        v_v = vv[valid].astype(np.float64)

        # pinhole deprojection from lecture notes
        # x = (u - cx) * z / fx, same for y
        x_c = (u_v - self.cx) * z_c / self.fx
        y_c = (v_v - self.cy) * z_c / self.fy

        pts_cam  = np.vstack((x_c, y_c, z_c))

        # rotate into base_link frame then add the camera offset
        pts_base = np.dot(R_OPTICAL_TO_BASE, pts_cam) + self.t_cam_to_base

        x_b = pts_base[0]
        y_b = pts_base[1]
        z_b = pts_base[2]

        # filter to just the box in front of the car, also drop ground points
        # tried using a plane fit for ground removal but the box filter was good enough
        mask = (np.isfinite(x_b) & np.isfinite(y_b) & np.isfinite(z_b)
                & (x_b >= self.x_base_min) & (x_b <= self.x_base_max)
                & (y_b >= self.y_base_min) & (y_b <= self.y_base_max)
                & (z_b >= self.z_base_min) & (z_b <= self.z_base_max))

        if not np.any(mask):
            self.cam_ranges     = np.empty(0, dtype=np.float64)
            self.cam_angles     = np.empty(0, dtype=np.float64)
            self.last_cam_stamp = msg.header.stamp
            return

        x_k = x_b[mask]
        y_k = y_b[mask]

        # project down onto lidar plane, just need range and angle
        self.cam_ranges     = np.hypot(x_k, y_k)
        self.cam_angles     = np.arctan2(y_k, x_k)
        self.last_cam_stamp = msg.header.stamp

    def fuse_scan_with_camera(self, scan_msg):
        ranges = list(scan_msg.ranges)

        if (not self.use_camera
                or self.cam_ranges.size == 0
                or self.last_cam_stamp is None):
            return ranges

        # ignore camera data if its too old
        age = abs((scan_msg.header.stamp - self.last_cam_stamp).to_sec())
        if age > self.camera_hold_time:
            return ranges

        # for each camera point find the nearest lidar beam and lower it if camera is closer
        # tried doing this with numpy indexing but had issues with duplicate indices overwriting
        for angle, r_cam in zip(self.cam_angles, self.cam_ranges):
            idx = int(round((angle - scan_msg.angle_min) / scan_msg.angle_increment))
            if 0 <= idx < len(ranges):
                r_padded = max(r_cam - self.camera_range_padding, 0.0)
                if math.isnan(ranges[idx]) or math.isinf(ranges[idx]) or r_padded < ranges[idx]:
                    ranges[idx] = r_padded

        return ranges

    def lidar_callback(self, data):
        # fuse before doing anything else
        fused_ranges = self.fuse_scan_with_camera(data)

        class FusedScan(object): #lol
            pass
        fused = FusedScan()
        fused.ranges          = fused_ranges
        fused.angle_min       = data.angle_min
        fused.angle_increment = data.angle_increment
        fused.range_max       = data.range_max
        fused.header          = data.header

        theta_des = self.get_best_direction(fused)
        w, s = self.solve_barriers(fused, theta_des)

        if w is None:
            return

        v_ctrl = max(abs(self.vel), 0.3)

        if abs(s - 1.0) < 1e-6 or abs(s + 1.0) < 1e-6:
            rospy.loginfo("Barrier solution singular: s = %f", s)
            return

        wr = w / (s - 1.0)
        wl = w / (s + 1.0)

        wr_norm_sq = float(np.dot(wr, wr))
        wl_norm_sq = float(np.dot(wl, wl))

        if wr_norm_sq < 1e-12 or wl_norm_sq < 1e-12:
            rospy.loginfo("Invalid barrier normals")
            return

        dr = 1.0 / math.sqrt(wr_norm_sq)
        dl = 1.0 / math.sqrt(wl_norm_sq)

        w_hat_r = dr * wr
        w_hat_l = dl * wl

        d_dot_r = np.dot(np.array([v_ctrl, 0.0]), w_hat_r)
        d_dot_l = np.dot(np.array([v_ctrl, 0.0]), w_hat_l)
        d_dot_lr = d_dot_l - d_dot_r

        cos_alpha_l = np.dot(np.array([0.0, -1.0]), w_hat_l)
        cos_alpha_r = np.dot(np.array([0.0,  1.0]), w_hat_r)

        #dlr = dr - dl
        dlr = dl - dr

        denom = (v_ctrl ** 2) * (cos_alpha_r + cos_alpha_l)

        if abs(denom) < 1e-6:
            delta = 0.0

        else:
            delta = math.atan((-self.wheelbase / denom) * (-self.kp * dlr - self.kd * d_dot_lr))

        delta = max(-self.delta_max, min(self.delta_max, delta))

        front_dist = self.get_front_distance(fused)
        speed = self.desired_velocity * (1.0 - math.exp(-max(front_dist - self.d_stop, 0.0) / self.d_tau))

        rospy.loginfo("theta_des=%.3f s=%.3f dl=%.3f dr=%.3f delta=%.3f front=%.3f", theta_des, s, dl, dr, delta, front_dist)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.steering_angle = delta
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        

    def odom_callback(self, odom_msg):
        self.vel = odom_msg.twist.twist.linear.x


def main(args):
    rospy.init_node("GapWallFollow_node", anonymous=True)
    wf = GapBarrier()
    rospy.sleep(0.1)
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

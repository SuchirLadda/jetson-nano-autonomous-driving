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
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class GapBarrier:
    def __init__(self):
        # Read algorithm parameters from params.yaml
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

        # Add your subscribers for LiDAR scan and Odometry here
        rospy.Subscriber(lidarscan_topic, LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)

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
        """
        Checks whether beam angle is inside [theta_des + low_rel, theta_des + high_rel]
        using wrapped relative angles, so wraparound is handled safely.
        """
        rel = angle - theta_des
        while rel > math.pi: rel -= 2 * math.pi
        while rel < -math.pi: rel += 2 * math.pi
        return low_rel <= rel <= high_rel

    def laser_to_base_link_xy(self, r, angle):
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        return x, y

    def solve_barriers(self, scan_msg, theta_des):
        """
        Collect left/right sector points relative to theta_des,
        then solve the QP for parallel separating barriers.
        """

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

    def lidar_callback(self, data):
        theta_des = self.get_best_direction(data)
        w, s = self.solve_barriers(data, theta_des)

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

        front_dist = self.get_front_distance(data)
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

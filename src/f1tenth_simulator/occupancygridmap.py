#!/usr/bin/env python

import numpy as np
import rospy
import math
import sys

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from tf.transformations import euler_from_quaternion


class OccupancyGridMap:
    def __init__(self):
        # Topics
        lidarscan_topic = rospy.get_param('~scan_topic')
        odom_topic = rospy.get_param('~odom_topic')
        self.map_topic = rospy.get_param('~occ_map_topic')

        # Frames
        self.map_frame = rospy.get_param('~odom_frame')

        # Laser parameters
        self.max_lidar_range = rospy.get_param('~scan_range')
        self.scan_beams = rospy.get_param('~scan_beams')
        self.lidar_offset_x = rospy.get_param('~scan_distance_to_base_link')
        self.lidar_offset_y = 0.0
        self.lidar_yaw_offset = math.pi

        # Occupancy map parameters
        self.map_width = rospy.get_param('~map_width')
        self.map_height = rospy.get_param('~map_height')
        self.map_resolution = rospy.get_param('~map_res')

        # Probabilities
        self.p_occ = rospy.get_param('~p_occ')
        self.p_free = rospy.get_param('~p_free')
        self.p_un = 0.5

        # Thresholds for publishing occupancy values
        self.p_occ_thresh = rospy.get_param('~p_occ_thresh')
        self.p_free_thresh = rospy.get_param('~p_free_thresh')

        # Log-odds values
        self.l_occ = math.log(self.p_occ / (1.0 - self.p_occ))
        self.l_free = math.log(self.p_free / (1.0 - self.p_free))
        self.l_un = math.log(self.p_un / (1.0 - self.p_un))  # = 0

        # Map origin so the map is centered around odom origin
        self.map_origin_x = -(self.map_width * self.map_resolution) / 2.0
        self.map_origin_y = -(self.map_height * self.map_resolution) / 2.0

        # Internal log-odds map
        self.log_odds_map = np.zeros((self.map_height, self.map_width), dtype=np.float64)

        # Robot pose from odometry
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_received = False

        # OccupancyGrid message
        self.map_occ_grid_msg = OccupancyGrid()
        self.map_occ_grid_msg.header.frame_id = self.map_frame
        self.map_occ_grid_msg.info.resolution = self.map_resolution
        self.map_occ_grid_msg.info.width = self.map_width
        self.map_occ_grid_msg.info.height = self.map_height

        self.map_occ_grid_msg.info.origin.position.x = self.map_origin_x
        self.map_occ_grid_msg.info.origin.position.y = self.map_origin_y
        self.map_occ_grid_msg.info.origin.position.z = 0.0
        self.map_occ_grid_msg.info.origin.orientation.x = 0.0
        self.map_occ_grid_msg.info.origin.orientation.y = 0.0
        self.map_occ_grid_msg.info.origin.orientation.z = 0.0
        self.map_occ_grid_msg.info.origin.orientation.w = 1.0

        self.map_occ_grid_msg.data = [-1] * (self.map_width * self.map_height)

        # ROS interfaces
        self.lidar_sub = rospy.Subscriber(lidarscan_topic, LaserScan, self.lidar_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.map_pub = rospy.Publisher(self.map_topic, OccupancyGrid, queue_size=1)

    def wrap_to_pi(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def logodds_to_prob(self, l):
        return 1.0 - 1.0 / (1.0 + math.exp(l))

    def cell_center(self, row, col):
        x = self.map_origin_x + (col + 0.5) * self.map_resolution
        y = self.map_origin_y + (row + 0.5) * self.map_resolution
        return x, y

    def odom_callback(self, odom_msg):
        self.x = odom_msg.pose.pose.position.x
        self.y = odom_msg.pose.pose.position.y

        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w

        (_, _, self.yaw) = euler_from_quaternion([qx, qy, qz, qw])
        self.odom_received = True

    def lidar_callback(self, data):
        if not self.odom_received:
            return

        # Compute lidar pose in odom frame
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)

        lidar_x = self.x + c * self.lidar_offset_x - s * self.lidar_offset_y
        lidar_y = self.y + s * self.lidar_offset_x + c * self.lidar_offset_y
        lidar_yaw = self.yaw + self.lidar_yaw_offset

        angle_min = data.angle_min
        angle_increment = data.angle_increment
        ranges = data.ranges
        n_ranges = len(ranges)

        if n_ranges == 0:
            return

        # Fix 1: use half the cell diagonal as hit tolerance
        hit_tol = (math.sqrt(2.0) / 2.0) * self.map_resolution
        l_min = -5.0
        l_max = 5.0

        for row in range(self.map_height):
            for col in range(self.map_width):
                cell_x, cell_y = self.cell_center(row, col)

                dx = cell_x - lidar_x
                dy = cell_y - lidar_y

                cell_range = math.sqrt(dx * dx + dy * dy)
                cell_bearing = self.wrap_to_pi(math.atan2(dy, dx) - lidar_yaw)

                if cell_range > self.max_lidar_range:
                    continue

                beam_idx = int(round((cell_bearing - angle_min) / angle_increment))
                if beam_idx < 0 or beam_idx >= n_ranges:
                    continue

                beam_angle = angle_min + beam_idx * angle_increment
                if abs(self.wrap_to_pi(cell_bearing - beam_angle)) > (angle_increment):
                    continue

                meas_range = ranges[beam_idx]

                if math.isnan(meas_range) or math.isinf(meas_range):
                    continue
		if meas_range < data.range_min:
	            continue
                if meas_range > self.max_lidar_range:
                    meas_range = self.max_lidar_range

                if abs(cell_range - meas_range) <= hit_tol:
                    l_inv = self.l_occ
                elif cell_range < (meas_range - hit_tol):
                    l_inv = self.l_free
                else:
                    continue

                self.log_odds_map[row, col] = self.log_odds_map[row, col] + l_inv - self.l_un
                self.log_odds_map[row, col] = np.clip(self.log_odds_map[row, col], l_min, l_max)
        # Convert log-odds map to OccupancyGrid values
        map_data = []
        for row in range(self.map_height):
            for col in range(self.map_width):
                p = self.logodds_to_prob(self.log_odds_map[row, col])

                if p > self.p_occ_thresh:
                    map_data.append(100)
                elif p < self.p_free_thresh:
                    map_data.append(0)
                else:
                    map_data.append(-1)

        self.map_occ_grid_msg.header.stamp = rospy.Time.now()
        self.map_occ_grid_msg.info.map_load_time = rospy.Time.now()
        self.map_occ_grid_msg.data = map_data
        self.map_pub.publish(self.map_occ_grid_msg)


def main(args):
    rospy.init_node("occupancygridmap", anonymous=True)
    OccupancyGridMap()
    rospy.sleep(0.1)
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

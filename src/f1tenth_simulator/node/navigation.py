#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry


class WallFollow:
    def __init__(self):
        self.kp = rospy.get_param("~k_p")
        self.kd = rospy.get_param("~k_d")
        self.desired_velocity = rospy.get_param("~vehicle_velocity")
        self.delta_theta_deg = rospy.get_param("~delta_theta_deg")
        self.d_stop = rospy.get_param("~d_stop")
        self.d_tau = rospy.get_param("~d_tau")
        self.delta_max = rospy.get_param("~delta_max")

        self.angle_bl_deg = rospy.get_param("~angle_bl_deg")
        self.angle_al_deg = rospy.get_param("~angle_al_deg")
        self.angle_br_deg = rospy.get_param("~angle_br_deg")
        self.angle_ar_deg = rospy.get_param("~angle_ar_deg")

        self.wheelbase = rospy.get_param("~wheelbase")

        lidarscan_topic = rospy.get_param("~scan_topic")
        odom_topic = rospy.get_param("~odom_topic")
        drive_topic = rospy.get_param("~nav_drive_topic")

        self.vel = 0.5

        rospy.Subscriber(lidarscan_topic, LaserScan, self.lidar_callback, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)

        self.drive_pub = rospy.Publisher(drive_topic, AckermannDriveStamped, queue_size=1)

    def get_range(self, scan_msg, angle_deg):
        angle_rad = - math.radians(angle_deg)

        while angle_rad > math.pi:
            angle_rad -= 2 * math.pi

        while angle_rad < -math.pi:
            angle_rad += 2 * math.pi

        idx = int(round((angle_rad - scan_msg.angle_min) / scan_msg.angle_increment))
        idx = max(0, min(idx, len(scan_msg.ranges) - 1))

        r = scan_msg.ranges[idx]
        if math.isinf(r) or math.isnan(r):
            return None
        return r

    def get_front_distance(self, scan_msg):
        half_angle = math.radians(self.delta_theta_deg)
        min_dist = 999.0

        for i, r in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            if abs(angle) <= half_angle:
                if not math.isinf(r) and not math.isnan(r):
                    min_dist = min(min_dist, r)

        return min_dist
    
    

    def lidar_callback(self, data):
        bl = self.get_range(data, self.angle_bl_deg)
        al = self.get_range(data, self.angle_al_deg)
        br = self.get_range(data, self.angle_br_deg)
        ar = self.get_range(data, self.angle_ar_deg)

        #rospy.loginfo("raw bl = %s al = %s br = %s ar = %s", bl ,al, br, ar)

        if bl is None or al is None or br is None or ar is None:
            return

	def wrap(a):
		while a > math.pi: a -= 2 * math.pi
		while a < -math.pi: a += 2 * math.pi
		return a

        scan_br = wrap(math.radians(self.angle_br_deg))
        scan_ar = wrap(math.radians(self.angle_ar_deg))
        scan_bl = wrap(math.radians(self.angle_bl_deg))
        scan_al = wrap(math.radians(self.angle_al_deg))

        theta_r = scan_ar - scan_br
        theta_l = scan_bl - scan_al

        beta_r = math.atan2(ar * math.cos(theta_r) - br, ar * math.sin(theta_r))
        beta_l = math.atan2(al * math.cos(theta_l) - bl, al * math.sin(theta_l))

        alpha_r = beta_r + math.pi / 2.0 - scan_br
        alpha_l = -beta_l + 3.0 * math.pi / 2.0 - scan_bl   
        
        dr = br * math.cos(beta_r)
        dl = bl * math.cos(beta_l)

        dlr = dr - dl #simulator
	#dlr = dl - dr #expirement
        
        v_ctrl = max(abs(self.vel), 0.3)
        d_dot_lr = -v_ctrl * math.sin(alpha_l) - v_ctrl * math.sin(alpha_r)
        denom = (v_ctrl ** 2) * (math.cos(alpha_r) + math.cos(alpha_l))

        if abs(denom) < 1e-6:
            delta = 0.0
        else:
            delta = -math.atan((-self.wheelbase / denom) *(-self.kp * dlr - self.kd * d_dot_lr))

        if delta > self.delta_max:
            delta = self.delta_max

        elif delta < -self.delta_max:
            delta = -self.delta_max

        front_dist = self.get_front_distance(data)

        speed = self.desired_velocity * (1.0 - math.exp(-max(front_dist - self.d_stop, 0.0) / self.d_tau))

        rospy.loginfo("front=%.2f dl=%.2f dr=%.2f dlr=%.2f delta=%.2f | bl=%.2f al=%.2f br=%.2f ar=%.2f",front_dist, dl, dr, dlr, delta, bl, al, br, ar)

        #rospy.loginfo("dl=%.2f dr=%.2f delta=%.2f front=%.2f", dl, dr, delta, front_dist)

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.steering_angle = delta
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def odom_callback(self, odom_msg):
        self.vel = odom_msg.twist.twist.linear.x


def main(args):
    rospy.init_node("WallFollow_node", anonymous=True)
    WallFollow()
    rospy.sleep(0.1)
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

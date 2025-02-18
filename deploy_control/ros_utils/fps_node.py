import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, DurabilityPolicy
import time
import numpy as np

class FpsNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.create_timer(1, self.count_fps)
        self.counter = {}
        self.counter_time = time.time()
    
    def count(self, key):
        self.counter[key] = self.counter.get(key, 0) + 1

    def count_fps(self):
        t = time.time() - self.counter_time
        self.counter_time = time.time()
        self.cnt = 0
        self.get_logger().info(f"[INFO] [{self.get_name()}] FPS: {[(k, np.round(v/t, 2)) for k, v in self.counter.items()]}")
        self.counter = {k: 0 for k in self.counter.keys()}
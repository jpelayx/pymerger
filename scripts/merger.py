#!/usr/bin/env python
# coding=utf-8

'''

'''
import rospy 
from nav_msgs.msg import OccupancyGrid
import tf_conversions
from geometry_msgs.msg import TransformStamped
import tf2_ros

import cv2 as cv
import numpy as np

import threading 
from itertools import permutations

class Map(object):
    #  default values for OccupancyGrid and image
    free_img = 255
    free_map = 0
    free_map_thresh = 25
    occupied_img = 0
    occupied_map = 100 
    occupied_map_thresh = 75 
    unexplored_img = 205
    unexplored_map = -1

    def __init__(self):
        self.data_lock = threading.Lock()
        self.h_lock = threading.Lock()
        self.data = None
        self.parent = None
        self.src = None
        self.dst = None
        self.H = None
    
    def get_data(self):
        with self.data_lock:
            if (self.data is None) or (len(self.data) == 0):
                self.compute_data()
            return self.data
    
    def get_occ_grid_data(self):
        # converts image to OccupancyGrid, returns OccupancyGrid
        img = self.get_data()
        if (img is None) or (len(img) == 0):
            return []
        img_flat = np.array(np.transpose(img).flatten(), dtype='uint8')
        v = self.unexplored_img
        occ = np.piecewise(img_flat, [img_flat<v,img_flat==v,img_flat>v], [100,-1,0])
        occ = np.array(occ, dtype='int8')
        return occ.tolist()
    
    def compute_data(self):
        # recursively  puts resulting map image together
        s,d = self.src.get_data(), self.dst.get_data()
        if (len(s) == 0) or (len(d) == 0):
            return
        if (s is None) or (d is None):
            return

        h,w = d.shape
        src_trans = cv.warpAffine(s, self.get_H(), (w,h)) # applies rigid transform

        alpha = 0.5
        beta  = 1.0 - alpha 
        res = cv.addWeighted(d, alpha, src_trans, beta, 0.0) # adds imgs
        v = self.unexplored_img
        res = np.piecewise(res, [res<v,res==v,res>v], [0,205,255]) 
        # for each cell e in the map:
        # if e < unexplored, e is occupied
        # if e = unexplored, e is unexplored
        # if e > unexplored, e is free
        self.data = res
    
    def update_match(self):
        self.dst.update_match()
        new_h = match(self.src, self.dst)
        if good_match(new_h, self.src, self.dst):
            self.add_H(new_h)
        else:
            print('update caused bad h, keeping old one')
        self.compute_data()
    
    def add_src(self, src):
        self.src = src
    
    def add_dst(self, dst):
        self.dst = dst
        self.info = dst.info

    def add_H(self, H):
        with self.h_lock:
            self.H = H
    
    def get_H(self):
        with self.h_lock:
            return self.H
    
    def get_transforms(self):
        # recursively generates a list of tuples (ns, h)
        # with ns: map namespace
        #       h: transform to global map 
        dst_ts = self.dst.get_transforms()
        _, last_h = dst_ts[-1]
        ts = []
        ts.extend(dst_ts)
        ts.append((self.src.ns, last_h*self.get_H()))
        return ts

    def add_parent(self, parent):
        self.parent = parent
    
class SubMap(Map):
# class that receives map info directly from the robot's map topic
    def __init__(self, ns, merger=None):
        super(SubMap, self).__init__()
        self.merger = merger
        self.ns = ns
        self.sub = rospy.Subscriber(ns+'/map', OccupancyGrid, self.update_map)

    def update_map(self, m):
        # map topic callback
        print("map update")
        with self.data_lock:
            self.info = m.info
            w = m.info.width 
            h = m.info.height
            if self.data is None:
                self.data = np.empty((w,h), dtype=np.uint8)
            if self.data.shape != (w,h):
                self.data = np.empty((w,h), dtype=np.uint8)

            for i in range(len(m.data)):
                if m.data[i] == self.unexplored_map:
                    self.data[i%w][(int)(i/w)] = self.unexplored_img
                elif m.data[i] >= self.occupied_map_thresh:
                    self.data[i%w][(int)(i/w)] = self.occupied_img
                else :
                    self.data[i%w][(int)(i/w)] = self.free_img
            self.merger.update()

    def update_match(self):
        return 

    def get_data(self):
        with self.data_lock:
            if self.data is not None:
                return self.data
            else:
                return []
    
    def get_transforms(self):
        # id 
        return [(self.ns, np.eye(2,3,dtype=np.double))]

    def valid_data(self):
        if self.data is None:
            return False
        else:
            return True

def match(src_map, dst_map):
    # map matching
    if (src_map is None) or (dst_map is None):
        return None

    src, dst = src_map.get_data(), dst_map.get_data()
    if (src is None) or (dst is None):
        return None 
    if (len(src)==0) or (len(dst)==0):
        return None
    
    # joining free and unexplored regions so that only obstacles influence matches
    _,src_bin = cv.threshold(src, src_map.unexplored_img+1, 255, cv.THRESH_BINARY)
    _,dst_bin = cv.threshold(dst, dst_map.unexplored_img+1, 255, cv.THRESH_BINARY)

    # key point detection and descriptor computation with ORB
    orb = cv.ORB_create()
    kps, dess = orb.detectAndCompute(src_bin, None)
    kpd, desd = orb.detectAndCompute(dst_bin, None)

    # matching descriptors with BruteForce Matcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(dess,desd)

    # choose 30% best matches (or at least 6)
    matches = sorted(matches, key = lambda x:x.distance)
    numGoodMatches = int(len(matches) * 0.30) 
    if numGoodMatches < 6:
        numGoodMatches = 6
    good = matches[:numGoodMatches]

    points1 = np.zeros((len(good), 2), dtype=np.float32)
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    for i, match in enumerate(good):
        points1[i,:] = kps[match.queryIdx].pt
        points2[i,:] = kpd[match.trainIdx].pt

    # rigid (or partial affine) transform estimation 
    h,_ = cv.estimateAffinePartial2D(points1, points2, method=cv.RANSAC)
    return h

def good_match(h, src, dst):
    if h is None:
        return False
    if len(h) == 0:
        return False

    # by the definition of a rigid transform
    theta = np.arctan(h[1,0]/h[0,0])
    s = h[0,0]/np.cos(theta)

    # the ground truth for the scale value can be infered with the resolution info from the maps
    s_truth = np.sqrt(dst.info.resolution/src.info.resolution)

    # filter transforms that diverge from said ground truth
    tolerance = 0.01
    if (s >= s_truth - tolerance) and (s <= s_truth + tolerance) :
        return True 
    else:
        return False


class Merger():
    # manages map tree 
    def __init__(self, sub_list):
        self.lock = threading.Lock()
        self.updated = False
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)
        self.map_seq = 0
        self.unmatched = [SubMap(ns, self) for ns in sub_list]
        self.global_map = None
        
    def update(self):
        with self.lock:
            print('Local maps updated')
            self.updated = True
        
    def run(self):
        print('Running merger')
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            with self.lock:
                has_updated = self.updated
            if has_updated:
                if self.global_map is not None:
                    self.global_map.update_match()
                self.add_unmatched()
                with self.lock:
                    self.updated = False
                if self.global_map is not None:
                    print('writing global map')
                    og = OccupancyGrid()
                    og.data = self.global_map.get_occ_grid_data()
                    og.info = self.global_map.info
                    # og.header.seq = self.map_seq
                    # self.map_seq += 1
                    og.header.stamp = rospy.Time.now()
                    # og.header.frame_id = '/map'
                    self.map_pub.publish(og)
                    # cv.imwrite('/home/julia/catkin_ws/src/pymerger/media/gmap.png', d)
            if self.global_map is not None:
                self.tf_broadcast()
            rate.sleep()

    def add_unmatched(self):
        if self.global_map is None: 
            for s, d in permutations(list(range(len(self.unmatched))),2):
                h = match(self.unmatched[s],self.unmatched[d])
                if good_match(h,self.unmatched[s],self.unmatched[d]):
                    self.global_map = Map()
                    if s>d:
                        self.global_map.add_src(self.unmatched.pop(s))
                        self.global_map.add_dst(self.unmatched.pop(d))
                    else:
                        self.global_map.add_dst(self.unmatched.pop(d))
                        self.global_map.add_src(self.unmatched.pop(s))
                    self.global_map.add_H(h)
                    print('starting global map with maps ' + str(s) + ' and ' + str(d))
                    break
        if self.global_map is None:
            print('Couldnt start global map')
            return
        
        print(str(len(self.unmatched)) + ' maps left')
        matched = []
        for i,m in enumerate(self.unmatched):
            h = match(m, self.global_map)
            if good_match(h,m,self.global_map):
                print('adding map ' + str(i))
                new_global = Map()
                new_global.add_src(m)
                new_global.add_dst(self.global_map)
                new_global.add_H(h)
                self.global_map = new_global
                matched.append(i)
        matched.reverse()
        for i in matched:
            self.unmatched.pop(i)
    
    def tf_broadcast(self):
        br = tf2_ros.TransformBroadcaster()
        ts = self.global_map.get_transforms()
        print(ts)
        for ns, h in ts:
            t = TransformStamped()

            t.header.stamp = rospy.Time.now()
            t.header.frame_id = 'map'
            t.child_frame_id  = ns + '/map'

            t.transform.translation.x = h[0][2]
            t.transform.translation.y = h[1][2]
            t.transform.translation.z = 0.0

            # rotacao em relacao ao eixo z..
            theta = np.arctan(h[1,0]/h[0,0])
            q = tf_conversions.transformations.quaternion_from_euler(0.0,0.0,theta)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            br.sendTransform(t)

if __name__ == '__main__':
    rospy.init_node('merger', anonymous=True)

    if not rospy.has_param('merger/maps'):
        rospy.logerr("No maps specified")
    else:

        sub_list = rospy.get_param('merger/maps').split()
        rospy.loginfo("Subscribing to: " + str(sub_list))
        merger = Merger(sub_list)
        merger_thread = threading.Thread(target=merger.run())
        
        merger_thread.start()
        rospy.spin()
# -*- coding: utf-8 -*-
"""--------------------------------
BirdCameraのクラスやその他関数など

@author: Hisa
--------------------------------"""

""" import """
import BirdCamera as Bcamera
import numpy as np
import cv2
from matplotlib import pyplot as plt


""" Const Numbers """
RAD2DEG = 180.0 / np.pi 
DEG2RAD = np.pi / 180.0


"""============================================================================
    センサーデータクラス
============================================================================"""
class SensorData:
    """ Class for sensordata """
    def __init__(self, txtfilename):
        #print('\nBirdCamera_others:SensorData\nREAD:\n' + txtfilename + '\n')
        f = open(txtfilename)
        lines = f.readlines()
        self.gps_latitude       = float(lines[0])
        self.gps_longitude      = float(lines[1])
        self.gps_altitude       = float(lines[2])
        self.gps_acclacy        = float(lines[3])
        self.direction          = float(lines[4])
        self.accelerometer_X    = float(lines[5])
        self.accelerometer_Y    = float(lines[6])
        self.accelerometer_Z    = float(lines[7])
        f.close()
        self.calcCamPose()
        
    def __str__(self):
        return \
        '===================\n' + \
        ' SENSOR DATA VALUE \n' + \
        '===================\n' + \
        ' GPS-Latitude      : ' + str(self.gps_latitude) + '\n' + \
        ' GPS-Longitude     : ' + str(self.gps_longitude) + '\n' + \
        ' GPS-Altitude      : ' + str(self.gps_altitude) + '\n' + \
        ' GPS-Accuracy      : ' + str(self.gps_acclacy) + '\n' + \
        ' Compass-direction : ' + str(self.direction)	+ '\n' + \
        ' Accelerometer-X   : ' + str(self.accelerometer_X) + '\n' + \
        ' Accelerometer-Y   : ' + str(self.accelerometer_Y) + '\n' + \
        ' Accelerometer-Z   : ' + str(self.accelerometer_Z) + '\n' + \
        ' Roll angle [deg]  : ' + str(self.roll*180.0/np.pi) + '\n' + \
        ' Pitch angle [deg] : ' + str(self.pitch*180.0/np.pi) + '\n'

    def calcCamPose(self):
        aX = self.accelerometer_X
        aY = self.accelerometer_Y
        aZ = self.accelerometer_Z
        
        if(aY == 0 or aZ == 0):
            print('[ERROR] CamPose is uncorrect.\n')
            exit(-1)
        else:
            self.roll  = np.arctan2(aX, aY)
            self.pitch = np.arctan2(aZ, np.sqrt(aX**2 + aY**2))


"""============================================================================
    位置姿勢情報クラス
============================================================================"""
class LocRotData:
    """ Class for Location & Direction Data """
    def __init__(self, _R, _t):
        self.rX_rad   = np.arcsin(_R[2,1])
        self.rY_rad   = np.arctan2(-_R[2,0], _R[2,2])
        self.rZ_rad   = np.arctan2(-_R[0,1], _R[1,1])
        self.rX_deg   = self.rX_rad * RAD2DEG
        self.rY_deg   = self.rY_rad * RAD2DEG
        self.rZ_deg   = self.rZ_rad * RAD2DEG
        self.location = - np.linalg.inv(_R).dot(_t)

    def __str__(self):
        return "\n" + \
        '==================================\n' + \
        ' LOCATION & ROTATION DATE VALUES \n' + \
        '==================================\n' + \
        ' rX_deg (pitch) : ' + str(self.rX_deg) + '\n' + \
        ' rY_deg (yaw)   : ' + str(self.rY_deg)	+ '\n' + \
        ' rZ_deg (roll)  : ' + str(self.rZ_deg) + '\n' + \
        ' location_X     : ' + str(self.location[0]) + '\n' + \
        ' location_Y     : ' + str(self.location[1]) + '\n' + \
        ' location_Z     : ' + str(self.location[2]) + '\n'
        #' rX_rad (pitch) : ' + str(self.rX_rad) + '\n' + \
        #' rY_rad (yaw)   : ' + str(self.rY_rad) + '\n' + \
        #' rZ_rad (roll)  : ' + str(self.rZ_rad) + '\n' + \


"""============================================================================
    読み込んだ画像を確認する関数
============================================================================"""
def dispImg(_winname, _imgarr, size=[640,480]):
    cv2.namedWindow(_winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_winname, size[0], size[1])
    cv2.imshow(_winname, _imgarr)


"""============================================================================
    Pyplotlibで読み込んだ画像を確認する関数
============================================================================"""
def dispImg2(_imgarr, title="image", num=0):
    plt.figure(num)
    plt.title(title)
    plt.imshow(cv2.cvtColor(_imgarr, cv2.COLOR_BGR2RGB))


    
"""============================================================================
============================================================================"""
if __name__ == "__main__":
    """" main()関数呼び出し """
    Bcamera.main()
    
# -*- coding: utf-8 -*-
"""--------------------------------
BirdCamera- virtual top-view
Virtual Top-View image

Created on Sat Jul  2 22:42:59 2016
@author: h1sat0r1
--------------------------------"""


""" Import """
import BirdCamera as Bcamera
import numpy as np
import cv2


""" Constant Numbers """
CAM_HEIGHT = 150.0  # Camera height from ground[cm]
RES_AERIAL = 1.0     # Spatial resolution of aerial image[cm/pix]
PHI1       = np.arctan2(1224.0, 2610.066) # Half of vertical angle of view
PHI2       = np.arctan2(1632.0, 2612.083) # Half of holizontal angle of view
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi


"""============================================================================
    decideArea() function
============================================================================"""
def decideArea(_res, _height, _camheight, _sensor):
    #area = -1.0
    #psi = np.pi/2.0 - _sensor.pitch
    area = 1.2 + (_sensor.pitch*180.0/np.pi)/90.0
    return area


"""============================================================================
    makeVTV() function
============================================================================"""
def makeVTV(_camimg, _sensor):
    
    """ Getting the image size """
    height,width,color = _camimg.shape

    
    """ Correcting roll angle """
    center = (width/2.0, height/2.0)
    rMat = cv2.getRotationMatrix2D(center, -_sensor.roll*RAD_TO_DEG, 1.0)
    rotImg = cv2.warpAffine(_camimg, rMat, (width,height))

    
    """ Deciding area """
    area = decideArea(RES_AERIAL, height, CAM_HEIGHT, _sensor)


    """ Variables """
    psi          = np.pi/2.0 - _sensor.pitch
    tan_psi      = np.tan(psi)
    tan_psimphi1 = np.tan(psi - PHI1)
    tan_phi2     = np.tan(PHI2)

    
    """ Pts of source """
    pts1 = np.float32([[0, height/2.0],\
                       [0, height],\
                       [width-1, height],\
                       [width-1, height/2.0]])
    

    """ Pts for destination """
    p0 = width/2.0 - (1.0/area) * height * tan_phi2 / (np.sin(psi) - tan_psimphi1 * np.cos(psi))
    p1 = width/2.0 - (1.0/area) * height * np.cos(PHI1) * tan_phi2 / (tan_psi * np.cos(psi - PHI1) - np.sin(psi - PHI1))
    p2 = width/2.0 + (1.0/area) * height * np.cos(PHI1) * tan_phi2 / (tan_psi * np.cos(psi - PHI1) - np.sin(psi - PHI1))
    p3 = width/2.0 + (1.0/area) * height * tan_phi2 / (np.sin(psi) - tan_psimphi1 * np.cos(psi))
    pts2 = np.float32([[p0, (1.0-1.0/area)*height],\
                       [p1, height-1],\
                       [p2, height-1],\
                       [p3, (1.0-1.0/area)*height]])
                       

    """ Finding homography referring to the 4 pts """
    pMat = cv2.getPerspectiveTransform(pts1,pts2)
    

    """ Affin mat to Homography mat """    
    rMat3x3 = np.vstack((rMat, np.array([0,0,1])))
   

    """ Generating vtv mat """    
    vtvimg = cv2.warpPerspective(rotImg, pMat, (width,height))
    

    return [vtvimg, pMat.dot(rMat3x3)]
    

 
"""============================================================================
============================================================================"""
if __name__ == "__main__":
    """" Call main() """
    Bcamera.main()
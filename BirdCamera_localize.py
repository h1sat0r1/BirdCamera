# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
   BirdCamera-localization
    Created on Mon Aug  8 19:41:34 2016
    @author: h1sat0r1
----------------------------------------------------------------------------"""


""" Import """
import BirdCamera as Bcamera
import BirdCamera_others as Bo
import numpy as np
import cv2


"""============================================================================
    localize func 
============================================================================"""
def localize(_H, _K):
    """ Find homography aerial img to mobile img """
    iH = np.linalg.inv(_H)


    """-------------------------------
      How to localize:
        P = K[R t] = K[r1 r2 r3 t]
        [r1 r2 t] = DOT(K.inv(), iH)
        r3 = CROSS(r1, r2)
        [R t] = [r1 r2 r3 t]
    -------------------------------"""
    
    """ Calculating r1, r2, t """
    X  = np.linalg.inv(_K).dot(iH)
    r1 = X[:,0]
    r2 = X[:,1]
    t  = X[:,2]
    

    """ Calculating norm """
    norm1 = np.linalg.norm(r1)
    norm2 = np.linalg.norm(r2)
    

    """ Calculating r3, t """
    r1 /= norm1
    r2 /= norm2 
    r3 = np.cross(r1,r2)
    t /= (norm1+norm2)/2.0
    

    """ Calculating R """
    R = np.c_[r1, np.c_[r2, r3]]
    
    #""" 剛体変換行列[Rt]生成 """
    #Rt = np.c_[R,t]
    

    """ Calculating location """
    return Bo.LocRotData(R, t)


"""============================================================================
============================================================================"""
if __name__ == "__main__":
    """" Call main() """
    Bcamera.main()
    
    
    
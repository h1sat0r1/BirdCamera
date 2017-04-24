# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
　BirdCamera-main Python version of master thesis
　　Created on Sun Jun 19 11:15:39 2016
　　@author: h1sat0r1
----------------------------------------------------------------------------"""

""" Import """
import numpy as np
import sys
import cv2
from matplotlib import pyplot as plt


""" Import Original Libs """
import BirdCamera_matching as Bm
import BirdCamera_others   as Bo
import BirdCamera_vtv      as Bv
import BirdCamera_localize as Bl


""" Camera parameter """
K = np.array([[2612.810,        0, 1214.722],\
              [       0, 2613.870, 1546.102],\
              [       0,        0,        1]])


"""============================================================================
    main関数
============================================================================"""
def main():

    """ Command line argments """
    #argv = sys.argv
    #argc = len(argv)


    """ Loading mobile camera images """
    #camimg = cv2.imread("input\\20131207_155745.jpg")    
    camimg = cv2.imread("input\\20131207_155757.jpg")
    #img_cam  = cv2.imread(argv[1])


    """ Loading GPS and accerelometer data """
    #sensordata = Bo.SensorData("input\\20131207_155745.txt")
    sensordata = Bo.SensorData("input\\20131207_155757.txt")
    #sensordata = Bo.SensorData(argv[2])
    print(sensordata)


    """ Loading aerial-view image """
    print("READ AERIAL-VIEW IMAGE...")
    aeroimg = cv2.imread("input\\IMG_2972_01.0.png")
    #img_aero = cv2.imread(argv[3])


    """ Generating virtual top-view image and getting homography matrix """
    print("GENARATE VIRTUAL TOP-VIEW IMAGE...")
    vtvimg, proj1 = Bv.makeVTV(camimg, sensordata)


    """ Finding corresponding pts """
    print("KEYPOINT MATCHING...")
    mtcimg, proj2 = Bm.kpMatch(vtvimg, aeroimg)


    """ Find homography between aerial-img and vtv img """
    proj21 = proj2.dot(proj1)
    dst = cv2.warpPerspective(camimg, proj21, (aeroimg.shape[1],aeroimg.shape[0]))


    """ Mobile camera localization """
    print("CAMERA LOCALIZATION...")
    locrot = Bl.localize(proj21, K)


    """ Display result """
    """ *Includes bugs. """
    print(locrot)    
    

    """ Display image """
    Bo.dispImg2(aeroimg, num=0, title="Aerial Image")
    Bo.dispImg2(vtvimg, title="Virtual Top-View Image", num=1)
    Bo.dispImg2(mtcimg, title="Kp", num=2)
    Bo.dispImg2(dst, title="Dst", num=3)
    plt.show()
    

    """ Save """
    cv2.imwrite("vtv.jpg",vtvimg)
    cv2.imwrite("dst.jpg",dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


    print("\n" + \
          "---------\n" + \
          "   END   \n" + \
          "---------\n")


"""============================================================================
============================================================================"""
if __name__ == "__main__":
    main()

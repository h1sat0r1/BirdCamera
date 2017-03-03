# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
　BirdCameraのmainプログラム 修論プログラムのPython書き直し版
　　Created on Sun Jun 19 11:15:39 2016
　　@author: Hisa
----------------------------------------------------------------------------"""

""" Import """
import numpy as np
import sys
import cv2
from matplotlib import pyplot as plt


""" Import Original Libs """
import BirdCamera_matching as Bm
import BirdCamera_others as Bo
import BirdCamera_vtv as Bv
import BirdCamera_localize as Bl


""" Const Number """
K = np.array([[2612.810,        0, 1214.722],\
              [       0, 2613.870, 1546.102],\
              [       0,        0,        1]])


"""============================================================================
    main関数
============================================================================"""
def main():

    """ コマンドライン引数 """
    argv = sys.argv
    argc = len(argv)


    """ モバイルカメラ画像の読込 """
    #camimg = cv2.imread("input\\20131207_155745.jpg")    
    camimg = cv2.imread("input\\20131207_155757.jpg")
    #img_cam  = cv2.imread(argv[1])


    """ 加速度センサ・GPS計測値の読込 """
    #sensordata = Bo.SensorData("input\\20131207_155745.txt")
    sensordata = Bo.SensorData("input\\20131207_155757.txt")
    #sensordata = Bo.SensorData(argv[2])
    print(sensordata)


    """ 空撮画像の読込 """
    print("READ AERIAL-VIEW IMAGE...")
    aeroimg = cv2.imread("input\\IMG_2972_01.0.png")
    #img_aero = cv2.imread(argv[3])


    """ 仮想俯瞰画像生成＆射影変換行列取得 """
    print("GENARATE VIRTUAL TOP-VIEW IMAGE...")
    vtvimg, proj1 = Bv.makeVTV(camimg, sensordata)


    """ 仮想俯瞰画像と空撮画像のマッチング """
    print("KEYPOINT MARCHING...")
    mtcimg, proj2 = Bm.kpMatch(vtvimg, aeroimg)


    """ モバイルカメラ画像<->空撮画像の射影変換行列 """
    proj21 = proj2.dot(proj1)
    dst = cv2.warpPerspective(camimg, proj21, (aeroimg.shape[1],aeroimg.shape[0]))


    """ マッチングの結果の射影変換行列からカメラのローカライズ """
    print("CAMERA LOCALIZATION...")
    locrat = Bl.localize(proj21, K)


    """ 表示 """
    print(locrat)    
    

    """ 表示 """
    Bo.dispImg2(aeroimg, num=0, title="Aerial Image")
    Bo.dispImg2(vtvimg, title="Virtual Top-View Image", num=1)
    Bo.dispImg2(mtcimg, title="Kp", num=2)
    Bo.dispImg2(dst, title="Dst", num=3)
    plt.show()
    

    """ 保存 """
    cv2.imwrite("vtv.jpg",vtvimg)
    cv2.imwrite("dst.jpg",dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


    """ おわり """
    print("\n" + \
          "---------\n" + \
          "   END   \n" + \
          "---------\n")


"""============================================================================
============================================================================"""
if __name__ == "__main__":
    """" main()関数呼び出し """
    main()

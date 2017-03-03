# -*- coding: utf-8 -*-
"""--------------------------------
BirdCameraの仮想空撮画像生成処理部
Virtual Top-View image

Created on Sat Jul  2 22:42:59 2016
@author: Hisa
--------------------------------"""


""" Import """
import BirdCamera as Bcamera
import numpy as np
import cv2


""" Constant Numbers """
CAM_HEIGHT  = 150.0  # カメラの地上からの高さ[cm]
RES_AERIAL  = 1.0	# 空撮画像の空間解像度[cm/pix]
PHI1		= np.arctan2(1224.0, 2610.066) # 垂直画角の1/2
PHI2		= np.arctan2(1632.0, 2612.083) # 水平画角の1/2


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
	"""	仮想俯瞰画像の生成関数 """
	
	
	""" 画像サイズの取得 """
	height,width,color = _camimg.shape

	
	""" ロール角の補正 """
	center = (width/2.0, height/2.0)
	rMat = cv2.getRotationMatrix2D(center, -_sensor.roll*180.0/np.pi, 1.0)
	rotImg = cv2.warpAffine(_camimg, rMat, (width,height))

	
	""" AREAの決定 """
	area			= decideArea(RES_AERIAL, height, CAM_HEIGHT, _sensor)


	""" 諸々の定数 """
	psi			 = np.pi/2.0 - _sensor.pitch
	tan_psi		 = np.tan(psi)
	tan_psimphi1	= np.tan(psi - PHI1)
	tan_phi2		= np.tan(PHI2)

	
	""" 変換元の座標値リスト """
	pts1 = np.float32([[0, height/2.0],\
					   [0, height],\
					   [width-1, height],\
					   [width-1, height/2.0]])
	

	""" 変換先の座標値 """
	p0 = width/2.0 - (1.0/area) * height * tan_phi2 / (np.sin(psi) - tan_psimphi1 * np.cos(psi))
	p1 = width/2.0 - (1.0/area) * height * np.cos(PHI1) * tan_phi2 / (tan_psi * np.cos(psi - PHI1) - np.sin(psi - PHI1))
	p2 = width/2.0 + (1.0/area) * height * np.cos(PHI1) * tan_phi2 / (tan_psi * np.cos(psi - PHI1) - np.sin(psi - PHI1))
	p3 = width/2.0 + (1.0/area) * height * tan_phi2 / (np.sin(psi) - tan_psimphi1 * np.cos(psi))
	pts2 = np.float32([[p0, (1.0-1.0/area)*height],\
					   [p1, height-1],\
					   [p2, height-1],\
					   [p3, (1.0-1.0/area)*height]])
					   

	""" 変換前後の4座標値から射影変換行列を算出 """
	pMat = cv2.getPerspectiveTransform(pts1,pts2)
	

	""" アフィン変換行列を3x3の形にする """	
	rMat3x3 = np.vstack((rMat, np.array([0,0,1])))
   

	""" 仮想俯瞰画像生成 """	
	vtvimg = cv2.warpPerspective(rotImg, pMat, (width,height))
	

	""" 仮想俯瞰画像と射影変換行列を返り値にする """
	return [vtvimg, pMat.dot(rMat3x3)]
	

 
"""============================================================================
============================================================================"""
if __name__ == "__main__":
    """" main()関数呼び出し """
    Bcamera.main()
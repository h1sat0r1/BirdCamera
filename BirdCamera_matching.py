# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
 BirdCameraの特徴点マッチング処理部分
　 Created on Sun Jun 19 18:41:30 2016
　 @author: Hisa
----------------------------------------------------------------------------"""

""" Import """
import BirdCamera as Bcamera
import numpy as np
import cv2
import sys


""" Const Numbers """
NN_DIST_RATIO   = 0.65
MIN_MATCH_COUNT = 8
RANSAC_THRESH   = 0.50
DRAW_PARAMS	 = dict(matchColor=(0,255,255),singlePointColor=(255,0,0),flags=0)


"""============================================================================
pickGoodMatches 関数
============================================================================"""
def pickGoodMatches(_matches):
	
	good = []
	for m,n in _matches:
		if m.distance < NN_DIST_RATIO*n.distance:
			good.append(m)
	return good


"""============================================================================
	drawMatch 関数
============================================================================"""
def drawMatch(_img0, _kps0, _img1, _kps1):
	"""
	簡易的にマッチング結果を出力できる。OpenCV標準のが使いにくいので自作した。
	_kps0[n]と_kps1[n]が対応点であるようなkeypoint配列
	"""

	""" 各画像のサイズ """
	h0, w0 = _img0.shape[:2]
	h1, w1 = _img1.shape[:2]


	""" Resizeするには配列コピーが必要なので """
	i0 = _img0.copy()
	i1 = _img1.copy()


	""" 縦サイズをどちらに合わせるか """
	if(h0 < h1):
		i0.resize((h1, w0))
	elif(h0 > h1):
		i1.resize((h0, w1))


	""" 横に連結 """ #結局縦連結はメモリ配置の関係で難しいという結論。横だけでやる。
	merge = cv2.hconcat([i0, i1])
	
	
	""" lineの描画 """


	""" 表示 """
	cv2.imshow("merge", merge)


	""" 仮にmergeを出力 """
	return merge


"""============================================================================
Keypoint Match 関数
============================================================================"""
def kpMatch(_img0, _img1):
	""" 特徴点マッチング・ホモグラフィ行列算出処理を行う関数 """

	""" まずグレースケール変換 """
	gry0 = cv2.cvtColor(_img0, cv2.COLOR_BGR2GRAY)
	gry1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2GRAY)
	
	
	""" SIFTとかSURFとか """
	"""-------------------------------------------------------
	※ SIFT,SURFはopencv_contrib内にあり、特許が取られている関係から別途コンパイルする必要がある。
	そういった作業が必要ない手法としてはORBやFREAKなどを使う。
	※ SIFT以外ならAKAZEが意外とマッチング結果がいいが、スケール変化には弱い印象。。。
	-------------------------------------------------------"""
	""" Detector """
	detect   = cv2.xfeatures2d.SIFT_create()
	#detect   = cv2.xfeatures2d.SURF_create()
	#detect   = cv2.ORB_create()
	#detect   = cv2.AgastFeatureDetector_create()
	#detect   = cv2.AKAZE_create()
	
 
	""" Descriptor """
	descript = cv2.xfeatures2d.SIFT_create()
	#descript = cv2.xfeatures2d.SURF_create()
	#descript = cv2.xfeatures2d.DAISY_create()
	#descript = cv2.ORB_create()
	#descript = cv2.BRISK_create()
	#descript = cv2.xfeatures2d.FREAK_create()
	#descript = cv2.AKAZE_create()
	
	
	""" Detection """
	kp0 = detect.detect(gry0)
	kp1 = detect.detect(gry1)
	
	
	""" Description """
	kp0, dsc0 = descript.compute(gry0, kp0)
	kp1, dsc1 = descript.compute(gry1, kp1)
	
	
	""" Matching """
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(dsc0, dsc1, k=2)
	good = pickGoodMatches(matches)
	
	
	"""" Compute Homography"""
	if(len(good) < MIN_MATCH_COUNT):
		""" 対応点数が必要最少数に達していない場合 """
		print("[ERROR] Not enough matches are found...\n")
		exit(-1)
	else:
		""" 対応点数が充分な場合 """
		srcPts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
		dstPts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

		""" Homography行列の算出 """
		proj2, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, RANSAC_THRESH)
	
	
	""" Display Matching Result """
	#cv2.imshow("Match", img2)
	#img2 = drawMatch(_img0, srcPts, _img1, dstPts)
	img2 = cv2.drawMatchesKnn(_img0, kp0, _img1, kp1,[good], None, **DRAW_PARAMS)
	
	
	""" Display Detection&Description Results """
	kpimg0 = cv2.drawKeypoints(_img0, kp0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	kpimg1 = cv2.drawKeypoints(_img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	
	""" imwrite """
	cv2.imwrite("Kp0.jpg", kpimg0)
	cv2.imwrite("Kp1.jpg", kpimg1)
	cv2.imwrite("Kps.jpg", img2)
	
	return [img2, proj2]
 
 
"""============================================================================
============================================================================"""
if __name__ == "__main__":
    """" main()関数呼び出し """
    Bcamera.main()
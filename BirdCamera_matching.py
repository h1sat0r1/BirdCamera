# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
 BirdCamera-matching
　 Created on Sun Jun 19 18:41:30 2016
　 @author: h1sat0r1
----------------------------------------------------------------------------"""

""" Import """
import BirdCamera as Bcamera
import numpy as np
import cv2
import sys


""" Const Numbers """
NN_DIST_RATIO      = 0.65
MIN_MATCH_COUNT    = 8
THRESH_RANSAC      = 0.50
PARAMS_DRAW        = dict(matchColor=(0,255,255),singlePointColor=(255,0,0),flags=0)
NUM_HIST_ANGLE     = 360
NUM_HIST_OCTAVE    = 32
THRESH_HIST_ANGLE  = 30
THREXH_HIST_OCTAVE = 1

"""============================================================================
    createHist()
============================================================================"""
def createHist(_matches):

    hist_angle  = [0] * NUM_HIST_ANGLE
    hist_octave = [0] * (NUM_HIST_OCTAVE * 2 + 1)
    
    
    for m,n in _matches:

        """ Angle """
        gap_angle  = int(m.angle  - n.angle + 0.5)
        while (gap_angle < 0):
            gap_angle += NUM_HIST_ANGLE
        hist_angle[gap_angle] += 1


        """ Octave """        
        gap_octave = m.octave - n.octave
        if ((gap_octave < -NUM_HIST_OCTAVE) or (NUM_HIST_OCTAVE < gap_octave)):
            continue
        hist_octave[gap_octave + NUM_HIST_OCTAVE] += 1


    return [hist_angle, hist_octave]


"""============================================================================
    pickGoodMatches()
============================================================================"""
def pickGoodMatches(_matches):
    
    hist_angle, hist_octave = createHist(_matches)

    num_max_hist_angle  = max(hist_angle)
    num_max_hist_octave = max(hist_octave)
    id_max_hist_angle   = hist_angle.index(num_max_hist_angle) 
    id_max_hist_octave  = hist_angle.index(num_max_hist_octave) 
    
        
    
    good = []
    for m,n in _matches:
        if m.distance < NN_DIST_RATIO*n.distance:
            good.append(m)
            
    return good


"""============================================================================
    drawMatch()
============================================================================"""
def drawMatch(_img0, _kps0, _img1, _kps1):
    
    """ Size of each image """
    h0, w0 = _img0.shape[:2]
    h1, w1 = _img1.shape[:2]


    """ Copying before resize """
    i0 = _img0.copy()
    i1 = _img1.copy()


    """ Determining height size """
    if(h0 < h1):
        i0.resize((h1, w0))
    elif(h0 > h1):
        i1.resize((h0, w1))


    """ Combining """ #結局縦連結はメモリ配置の関係で難しいという結論。横だけでやる。
    merge = cv2.hconcat([i0, i1])
    
    
    """ drawing line """


    """ Displaying """
    cv2.imshow("merge", merge)


    #return 


"""============================================================================
    Keypoint Match ()
============================================================================"""
def kpMatch(_img0, _img1):

    """ Glayscale """
    gry0 = cv2.cvtColor(_img0, cv2.COLOR_BGR2GRAY)
    gry1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2GRAY)
    
    
    """ SIFT, SURF, others... """
    """-------------------------------------------------------
    ※ SIFT,SURFはopencv_contrib内にあり、特許が取られている関係から別途コンパイルする必要がある。
    そういった作業が必要ない手法としてはORBやFREAKなどを使う。
    ※ SIFT以外ならAKAZEが意外とマッチング結果がいいが、スケール変化には弱い印象・・・
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
        """ In case of few matches """
        print("[ERROR] Not enough matches are found...\n")
        exit(-1)

    else:
        """ Enough number of matches """
        srcPts = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstPts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        """ Calculating homography """
        proj2, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, THRESH_RANSAC)
    
    
    """ Draw Matching Result """
    #img2 = drawMatch(_img0, srcPts, _img1, dstPts)
    img2 = cv2.drawMatchesKnn(_img0, kp0, _img1, kp1,[good], None, **PARAMS_DRAW)
    
    
    """ Draw Detection&Description Results """
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
    """" Call main() """
    Bcamera.main()
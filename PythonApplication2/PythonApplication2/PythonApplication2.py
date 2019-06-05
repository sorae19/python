#https://code-graffiti.com/opencv-lucas-kanade-optical-flow-in-python/
import numpy as np
import cv2

cap = cv2.VideoCapture('aerial.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (100,100),               #winSize引数、小さくするとノイズに敏感になり、大きな動きを見逃す可能性があります。
                  maxLevel = 2,   #maxLevelは画像のピラミッドのことで、ここが0の場合、ピラミッドを使用しないことを意味します。ピラミッドを使用すると、画像のさまざまな解像度でオプティカルフローを見つけることができます。
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params) #cv2.goodFeaturesToTrack()で物体のコーナーを検出（Shi-Tomasiコーナー検出）しています。

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #cv2.calcOpticalFlowPyrLK()を使ってグレースケールのフレーム上のオプティカルフローを計算します。この関数の引数に、直前のイメージ、直後のイメージ、直前に検出したポイント、直後のポイント（ここではNone）、最初の方で準備した辞書型のパラメータを順に渡して計算します。

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points       
    good_new = p1[st==1]  #返された新しいポイントと直前のポイントの配列を使います。
                                            #対応する特徴のオプティカルフローが見つかった場合はベクトルの各要素は1に設定され、それ以外の場合は0に設定されます。これをそれぞれgood_new、good_oldにします。
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
import cv2 
import numpy as np 

img = cv2.imread('/home/rui/remotePan/MLS/data/cat_affine.png')
# img = cv2.imread('/home/rui/remotePan/MLS/data/cat_sim.png')
# img = cv2.imread('/home/rui/remotePan/MLS/data/cat_arigid.png')
pad = np.zeros((50,img.shape[1],3),dtype=np.uint8)
imgPad = np.vstack([pad,img])
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(imgPad,'affine_mls',(290,30),font,0.8,(0,0,255),2)
# cv2.putText(imgPad,'similarity_mls',(290,30),font,0.8,(0,0,255),2)
# cv2.putText(imgPad,'rigid_mls',(290,30),font,0.8,(0,0,255),2)

cv2.imshow('img',imgPad)
cv2.imwrite('/home/rui/remotePan/MLS/data/cat_affine_pad.png',imgPad)
cv2.waitKey(0)
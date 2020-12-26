import cv2 
import numpy as np
def drawGrid(img,gridSize = 3):
    for i in range(0,img.shape[0],gridSize):
        cv2.line(img,(0,i),(img.shape[1]-1,i),(0,0,255),1)
    for j in range(0,img.shape[1],gridSize):
        cv2.line(img,(j,0),(j,img.shape[0]-1),(0,0,255),1)
    
    cv2.imshow('imgGrid',img)
    cv2.waitKey(0)
def drawGridPlus(img,gridSize = 3,gridArray = None,noShow = False):
    if gridArray is None:
        gridArray = []
        for i in range(0,img.shape[0]+gridSize,gridSize):
            tmpList = []
            for j in range(0,img.shape[1]+gridSize,gridSize):
                tmpList.append([j,i])
            gridArray.append(tmpList)
        gridArray = np.array(gridArray)
    # print('gridArray',gridArray.shape)
    
    for i in range(0,gridArray.shape[0]):
        for j in range(0,gridArray.shape[1]):
            x0,y0 = int(gridArray[i,j,0]),int(gridArray[i,j,1])
            if i < gridArray.shape[0]-1:
                x1,y1 = int(gridArray[i+1,j,0]),int(gridArray[i+1,j,1])
                cv2.line(img,(x0,y0),(x1,y1),(0,0,255),1 )
            if j <gridArray.shape[1]-1:
                x2,y2 = int(gridArray[i,j+1,0]),int(gridArray[i,j+1,1])
                cv2.line(img,(x0,y0),(x2,y2),(0,0,255),1 )
            # if noShow is False:
            #     print('----------------')
            #     print(x0,y0)
            #     print(x1,y1)
            #     print(x2,y2)
            #     print('----------------')
            #     cv2.imshow('imgGrid',img)
            #     cv2.waitKey(0)
    if noShow is False:
        cv2.imshow('imgGrid',img)
        cv2.waitKey(20)
    return gridArray

def drawLines(img,lines,winName):
    for points in lines:
        cv2.line(img,(int(points[0,0]),int(points[0,1])),(int(points[1,0]),int(points[1,1])),(255,0,0),3)
    cv2.imshow(winName,img)
    cv2.waitKey(0)
    return img
import cv2 
import numpy as np 
import os 
import math 

from tool.drawTools import *
from tool.geo import *
from tool.grid_warping import *
def  doublebilinear_interp(x,y,v11, v12, v21,v22): 

    return (v11*(1-y) + v12*y) * (1-x) +(v21*(1-y) + v22*y) * x
class AF_LINE_MLS:
    def __init__(self,imgRaw,p_list=None,q_list = None,a = 1.0,activatePoint = None):
        self.imgRaw = imgRaw
        self.p_list = p_list
        self.q_list = q_list
        self.activatePoint = activatePoint
        self.eps = 1e-2
    def run(self):
        img = self.imgRaw.copy()
        newImg = np.zeros((img.shape[0],img.shape[1],2),dtype=float)
        n_line = self.q_list.shape[0]

        lineNorm = np.zeros((self.p_list.shape[0],),dtype=float)

        dimg = drawLines(img.copy(),self.p_list,winName= 'src')
        dtImg = drawLines(img.copy(),self.q_list,winName= 'tar')

        for k in range(n_line):
            lineNorm[k] = np.linalg.norm(self.p_list[k,0]-self.p_list[k,1])
        # print('lineNorm',lineNorm,lineNorm.shape)
        for i in range(0,img.shape[0]):
            print(i,img.shape[0])
            # input('check')
            for j in range(0,img.shape[1]):
                # print('j',j)
                v = np.array([j,i]).reshape((-1,2))
                # v = np.array([68,8]).reshape((-1,2))
                line_thigma = np.zeros((n_line,3),dtype=float)
                endPoint = False
                for k in range(n_line):
                    ai = self.p_list[k,[0]]
                    bi = self.p_list[k,[1]]
                    ci = self.q_list[k,[0]]
                    di = self.q_list[k,[1]]
                    # print(k,ai,bi,ci,di)

                    if np.linalg.norm(ai-v)<self.eps :
                        newImg[i,j] = np.array([ci[0,0]-j,ci[0,1]-i])
                        # newImg[i,j] = np.array([ci[0,1]-i,ci[0,0]-j])
                        
                        endPoint = True
                    elif np.linalg.norm(bi-v)<self.eps :
                        endPoint = True
                        newImg[i,j] = np.array([di[0,0]-j,di[0,1]-i])
                        # newImg[i,j] = np.array([di[0,1]-i,di[0,0]-j])
                    # input('check')
                    if endPoint :
                        # newImg[i,j] = np.array([0.0,0.0])
                        break
                    t = pointOnSegLineV1(ai,bi,v)
                    # print('t',t,ai,bi,v)
                    
                    if t>-self.eps : # v in the line
                        t = math.fabs(t)
                        thigma_00,thigma_01,thigma_11 = computerthigmaInLine(self.p_list[k],lineNorm[k],v)
                        line_thigma[k] = np.array([thigma_00,thigma_01,thigma_11]).reshape((-1))
                        # print('thigma_00,thigma_01,thigma_11 in line',thigma_00,thigma_01,thigma_11)
                    else:
                        thigma_00,thigma_01,thigma_11 = computerthigmaOutLine(self.p_list[k],lineNorm[k],v)
                        line_thigma[k] = np.array([thigma_00,thigma_01,thigma_11]).reshape((-1))
                        # print('thigma_00,thigma_01,thigma_11 out line',thigma_00,thigma_01,thigma_11)
                # print('line_thigma',line_thigma,line_thigma.shape) 
                # input('chekc')
                if endPoint :
                    continue
                p_star = np.zeros((1,2),dtype=float)
                q_star = np.zeros((1,2),dtype=float)
                weight_sum = 0.0
                w_mat = np.zeros((n_line,2,2))
                for k in range(n_line):
                    thigma_00,thigma_01,thigma_11 = line_thigma[k]
                    w_mat[k,0,0],w_mat[k,0,1],w_mat[k,1,0],w_mat[k,1,1] = thigma_00,thigma_01,thigma_01,thigma_11
                    # print('thigma_00,thigma_01,thigma_11',thigma_00,thigma_01,thigma_11)
                    # print('w_mat',w_mat[k])
                    p_tmp = self.p_list[k,[0]]*(thigma_00 + thigma_01) + self.p_list[k,[1]] * (thigma_01 + thigma_11)
                    weight_sum += thigma_00 + thigma_01 + thigma_01 + thigma_11
                    q_tmp = self.q_list[k,[0]]*(thigma_00 + thigma_01) + self.q_list[k,[1]] * (thigma_01 + thigma_11)
                    p_star += p_tmp
                    q_star += q_tmp
                    # input('check')
                p_star /= weight_sum
                q_star /= weight_sum
                _p_hat = np.zeros((n_line,2,2),dtype=float)
                _q_hat = np.zeros((n_line,2,2),dtype=float)
                AMat = np.zeros((2,2,),dtype=float)
                for k in range(n_line):
                    _p_hat[k] = self.p_list[k] - p_star
                    _q_hat[k] = self.q_list[k] - q_star
                    mat_hat_ab = np.mat(_p_hat[k])
                    mat_hat_ab_t = mat_hat_ab.T 
                    tmp_mat = mat_hat_ab_t.dot(w_mat[k])
                    tmp_mat = tmp_mat.dot(mat_hat_ab)
                    AMat += tmp_mat
                AMat_inv = np.mat(AMat).I 
                v_p_star = v - p_star
                out = np.zeros((1,2),dtype=float)
                for k in range(n_line):
                    mat_hat_cd = np.mat(_q_hat[k])

                    mat_hat_ab = np.mat(_p_hat[k])
                    mat_hat_ab_t = mat_hat_ab.T 

                    Aj = v_p_star.dot(AMat_inv)
                    Aj = Aj.dot(mat_hat_ab_t)
                    Aj = Aj.dot(w_mat[k])
                    out += Aj.dot(mat_hat_cd)
                out += q_star

                if out[0,0]>0 and out[0,0]<img.shape[1]-1 and out[0,1]>0 and out[0,1]<img.shape[0]-1:
                    # x = int(out[0,0])
                    # y = int(out[0,1])
                    # newImg[i,j] = np.array([out[0,0]-j,out[0,1]-i])
                    newImg[i,j] = np.array([out[0,0]-j],out[0,1]-i)
                    # newImg[i,j] = np.array([out[0,1]-i,out[0,0]-j])

        # timg = wrapImg(img,newImg)
        # timg = drawLines(timg.copy(),self.q_list,winName='deform')
        # cv2.imshow('timg',timg)
        # print('dimg,dtImg,timg',dimg.shape,dtImg.shape,timg.shape)
        # cv2.waitKey(20)
        # imgCompose = np.hstack([dimg,dtImg,timg])
        # cv2.imshow('imgCompose',imgCompose)
        # cv2.imwrite('./data/affineLine_small.png',imgCompose)
        # cv2.waitKey(0)
        self.getDeformImgPoints(newImg,img,dimg,dtImg)
    def getDeformImgPoints(self,dept,img,dimg,dtImg,gridSize = 1):
        timg = img.copy()
        timg[:] = 0
        nleft ,ntop,nbottom,nright = 0,0,0,0
        lt,rt,lb,rb = np.array([0,0]),np.array([0,0]),np.array([0,0]),np.array([0,0])
        piex = None
        for i in range(0,img.shape[0],gridSize):   #   y
            print('deform',i)
            for j in range(0,img.shape[1],gridSize):   #  x
                nleft=j
                ntop=i
                nright=j+gridSize
                nbottom=i+gridSize
                nright = min(nright,img.shape[1]-1)
                nbottom = min(nbottom,img.shape[0]-1)

                lefttop_offset = dept[ntop,nleft]
                leftbottom_offset = dept[nbottom,nleft]
                righttop_offset = dept[ntop,nright]
                rightbottom_offset = dept[nbottom,nright]
                for dj in range(nleft,nright):
                    for di in range(ntop,nbottom):
                        deltax = doublebilinear_interp( (dj-nleft)/(nright-nleft),(di-ntop)/(nbottom-ntop),\
                                                 lefttop_offset[0],righttop_offset[0],\
                                                leftbottom_offset[0],rightbottom_offset[0] )
                        deltay = doublebilinear_interp( (dj-nleft)/(nright-nleft),(di-ntop)/(nbottom-ntop),\
                                                 lefttop_offset[1],righttop_offset[1],\
                                                leftbottom_offset[1],rightbottom_offset[1] )
                        dx = dj - deltax  
                        dy = di - deltay
                        floor_x = max(0,math.floor(dx))
                        floor_x = min(img.shape[1]-1,floor_x)

                        floor_y = max(0,math.floor(dy))
                        floor_y = min(img.shape[0]-1,floor_y)

                        ceil_x = max(0,math.ceil(dx))
                        ceil_x = min(img.shape[1]-1,ceil_x)

                        ceil_y = max(0,math.ceil(dy))
                        ceil_y = min(img.shape[0]-1,ceil_y)

                        floor_pos = np.array([floor_x,floor_y])
                        ceil_pos = np.array([ceil_x,ceil_y])
                        for k in range(3):
                            timg[di,dj,k] = doublebilinear_interp(ceil_pos[0]-dx,ceil_pos[1]-dy,\
                                                    img[floor_y,floor_x,k],img[ceil_y,floor_x,k],\
                                                    img[floor_y,ceil_x,k],img[ceil_y,ceil_x,k] )
        
        imgCompose = np.hstack([dimg,dtImg,timg])
        cv2.imshow('imgCompose',imgCompose)
        cv2.imwrite('./data/zk_arigid.png',imgCompose)
        cv2.waitKey(0)

    def getDeformImg(self,dept,img,dimg,dtImg,gridSize = 1):
        timg = img.copy()
        timg[:] = 0
        nleft ,ntop,nbottom,nright = 0,0,0,0
        lt,rt,lb,rb = np.array([0,0]),np.array([0,0]),np.array([0,0]),np.array([0,0])
        piex = None
        for i in range(0,img.shape[0],gridSize):   #   y
            print('deform',i)
            for j in range(0,img.shape[1],gridSize):   #  x
                nleft=j
                ntop=i
                nright=j+gridSize
                nbottom=i+gridSize
                nright = min(nright,img.shape[1]-1)
                nbottom = min(nbottom,img.shape[0]-1)

                lefttop_offset = dept[ntop,nleft]
                leftbottom_offset = dept[nbottom,nleft]
                righttop_offset = dept[ntop,nright]
                rightbottom_offset = dept[nbottom,nright]
                for dj in range(nleft,nright):
                    for di in range(ntop,nbottom):
                        deltax = doublebilinear_interp( (dj-nleft)/(nright-nleft),(di-ntop)/(nbottom-ntop),\
                                                 lefttop_offset[0],righttop_offset[0],\
                                                leftbottom_offset[0],rightbottom_offset[0] )
                        deltay = doublebilinear_interp( (dj-nleft)/(nright-nleft),(di-ntop)/(nbottom-ntop),\
                                                 lefttop_offset[1],righttop_offset[1],\
                                                leftbottom_offset[1],rightbottom_offset[1] )
                        dx = dj - deltax  
                        dy = di - deltay
                        floor_x = max(0,math.floor(dx))
                        floor_x = min(img.shape[1]-1,floor_x)

                        floor_y = max(0,math.floor(dy))
                        floor_y = min(img.shape[0]-1,floor_y)

                        ceil_x = max(0,math.ceil(dx))
                        ceil_x = min(img.shape[1]-1,ceil_x)

                        ceil_y = max(0,math.ceil(dy))
                        ceil_y = min(img.shape[0]-1,ceil_y)

                        floor_pos = np.array([floor_x,floor_y])
                        ceil_pos = np.array([ceil_x,ceil_y])
                        for k in range(3):
                            timg[di,dj,k] = doublebilinear_interp(ceil_pos[0]-dx,ceil_pos[1]-dy,\
                                                    img[floor_y,floor_x,k],img[ceil_y,floor_x,k],\
                                                    img[floor_y,ceil_x,k],img[ceil_y,ceil_x,k] )
        
        timg = drawLines(timg.copy(),self.q_list,winName='deform')
        cv2.imshow('timg',timg)
        print('dimg,dtImg,timg',dimg.shape,dtImg.shape,timg.shape)
        cv2.waitKey(20)
        imgCompose = np.hstack([dimg,dtImg,timg])
        cv2.imshow('imgCompose',imgCompose)
        cv2.imwrite('./data/affineLine_small.png',imgCompose)
        cv2.waitKey(0)


# handle Img
# imgRaw = cv2.imread('/home/rui/remotePan/MLS/data/raw_line_0.png')
# # imgRaw  = cv2.resize(imgRaw,(imgRaw.shape[1]//2,imgRaw.shape[0]//2))

# imgCrop = imgRaw[165*2:451*2,205*2:846*2]
# cv2.imshow('imgRaw',imgRaw)
# cv2.imshow('imgCrop',imgCrop)
# img_0 = imgCrop[:,0:311*2]
# img_1 = imgCrop[:,329*2:-1*2]
# print('img',img_0.shape,img_1.shape)
# cv2.imshow('img_0',img_0)
# cv2.imshow('img_1',img_1)
# cv2.imwrite('./data/imgRaw_segLine.png',img_0)
# cv2.imwrite('./data/imgRaw_segLine_deform_paper.png',img_1)

img = cv2.imread('./data/imgRaw_segLine.png')
cv2.imshow('raw',img)
img_deform_affine_office = cv2.imread('./data/imgRaw_segLine_deform_paper.png')
scale = 0.2
img = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)))

# q_list = np.array([[2,61,48,116],[48,116,42,182],[42,182,95,245],[95,245,85,480],[85,480,0,487],[85,480,309,480],[309,479,332,152],[309,480,616,482],[616,482,618,33]])
p_list = np.array([[2,61,48,116],[48,116,42,182],[42,182,95,245],[95,245,85,480],[85,480,0,487],[85,480,309,480],[309,479,340,40],[309,480,616,482],[616,482,618,33]])
q_list = np.array([[2,61,48,116],[48,116,42,182],[42,182,95,245],[95,245,85,480],[85,480,0,487],[85,480,309,480],[309,479,250,160],[309,480,616,482],[616,482,618,33]])

p_list = p_list.reshape((p_list.shape[0],-1,2))*scale
q_list = q_list.reshape((p_list.shape[0],-1,2))*scale

activatePoint = np.array([340,40,250,160]).reshape((1,-1,2))*0.2
print('activatePoint',activatePoint)
# drawLines(img.copy(),activatePoint,winName='activate')
# print('p_list',p_list.shape)
# drawLines(img,p_list)
# drawLines(img_deform_affine_office,q_list)
# cv2.waitKey(0)

af_line_deform = AF_LINE_MLS(img,p_list,q_list,activatePoint)
af_line_deform.run()
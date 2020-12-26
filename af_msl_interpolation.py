import cv2 
import numpy as np
import math
def showImgPoints(img,points,winName = 'imgJoint'):
    for point in points:
        x = int(point[0])
        y = int(point[1])
        cv2.circle(img,(x,y),6,(255,0,255),3)
    cv2.imshow(winName,img)
    cv2.waitKey(0)
    return img.copy()

def  doublebilinear_interp(x,y,v11, v12, v21,v22): 

    return (v11*(1-y) + v12*y) * (1-x) +(v21*(1-y) + v22*y) * x


class AF_MLS:
    def __init__(self,img=None,tImg = None,plist = None,qlist = None,a=1.0):
        self.img = img
        self.tImg = tImg
        self.a = a 
        self.plist = plist
        self.qlist = qlist
        print('plist',self.plist)
        print('qlist',self.qlist)
        self.n = len(self.plist)
    def run(self):
        img = self.img
        tImg = self.tImg
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        dimg = showImgPoints(img.copy(),self.plist,winName= 'src')
        dtImg = showImgPoints(tImg.copy(),self.qlist,winName= 'tar')
        newImg = np.zeros((img.shape[0],img.shape[1],2),dtype=float)
        for i in range(img.shape[0]):
            print(i,img.shape[0])
            for j in range(img.shape[1]):
                v = np.array([j,i])
                # v = self.plist[-1]-0.5 #
                # v = np.array([img.shape[1]//2,img.shape[0]//2])
                w = np.zeros((self.n,),dtype=float)
                nearControl = False
                q_star = np.zeros((1,2),dtype=float)
                p_star = np.zeros((1,2),dtype=float)
                weight_sum = 0.0
                for k in range(self.n):
                    norm = (self.plist[k][0] - v[0] ) **2+ (self.plist[k][1] - v[1] ) **2
                    if norm < 1e-6:
                        newImg[i,j] = np.array([0,0])
                        nearControl = True
                        break
                    else:
                        w[k] = 1.0 / norm
                        weight_sum +=w[k]
                        p_star += w[k] * self.plist[k]
                        q_star += w[k] * self.qlist[k]
                # print('weight',w)
                # input('check')
                if nearControl is False:
                    p_star /= weight_sum 
                    q_star /= weight_sum 
                else:
                    continue
                
                # print('p_star',p_star)  
                # print('q_star',q_star)  
                # input('check')
                _p_hat = np.zeros((self.n,1,2),dtype=float)
                _q_hat = np.zeros((self.n,1,2),dtype=float)
                for k in range(self.n):
                    _p_hat[k,0] = self.plist[k] -  p_star 
                    _q_hat[k,0] = self.qlist[k] -  q_star 
                
                # left_term_A = (v - p_star)
                # print('left_term_A',left_term_A,left_term_A.shape)
                # input('check')
                # print('_p_hat',_p_hat)  
                # print('_q_hat',_q_hat)  
                # input('check')
                out = np.zeros((1,2),dtype=float)
                for kj in range(self.n):
                    _q_j = _q_hat[kj]
                    _p_j = _p_hat[kj] 
                    _p_j_t = np.transpose(_p_j,(1,0))
                    mat_sum = np.zeros((2,2),dtype=float)
                    for ki in range(self.n):  
                        _p_i =  _p_hat[ki] 
                        _p_i_t = np.transpose(_p_i,(1,0))*w[ki]
                        # print('_p_i',_p_i,_p_i_t)
                        mat = _p_i_t.dot(_p_i)
                        # print('mat',mat,w[ki])
                        # mat *= w[ki]
                        # print('mat',mat)
                        mat_sum += mat 
                        # input('check ')
                    
                    mat_inv = np.mat(mat_sum).I
                    # print('mat_sum',mat_sum,mat_inv,mat_inv.dot(mat_sum))
                    
                    left_term = v - p_star
                    # print('mat_inv',mat_inv,left_term)
                    A = (v - p_star).dot(mat_inv)
                    # print('A0',A,_p_j_t)
                    Aj = A.dot(_p_j_t)*w[kj]
                    # print('Aj',Aj,_q_j)
                    
                    out_tmp= Aj*_q_j
                    
                    out += out_tmp
                    # print('out_tmp',out_tmp,out)
                    # input('check')
                # print('---out',out,q_star)
                out += q_star
                
                if out[0,0]>0 and out[0,0]<img.shape[1]-1 and out[0,1]>0 and out[0,1]<img.shape[0]-1:
                    # x = int(out[0,0])
                    # y = int(out[0,1])
                    newImg[i,j] = np.array([out[0,0]-j,out[0,1]-i])
                    # newImg[y,x] = img[i,j]
        self.getDeformImg(newImg,img,dimg,dtImg)
    def getDeformImg(self,dept,img,dimg,dtImg,gridSize = 3):
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
        cv2.imwrite('./data/cat_affine.png',imgCompose)
        cv2.waitKey()

        

img = cv2.imread('./data/1.png')
img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
# cv2.imshow('img',img)
imgCrop = img[176:413,133:693]
# cv2.imshow('imgCrop',imgCrop)
img_0 = imgCrop[:,0:260]
img_1 = imgCrop[:,300:560]
# img_0 = cv2.resize(img_0,(img_0.shape[1]*2,img_0.shape[0]*2))
# img_1 = cv2.resize(img_1,(img_1.shape[1]*2,img_1.shape[0]*2))
# cv2.imshow('img_0',img_0)
# cv2.imshow('img_1',img_1)
# print('img_0',img_0.shape,img_1.shape)

# cv2.waitKey(0)
plist = np.array([[127,316],[253,300],[212,412],[293,413],[366,356]]) *0.5
qlist = np.array([[151,309],[258,300],[235,410],[283,414],[346,352]]) *0.5
# qlist = np.array([[150,49],[409,21],[160,201],[354,213],[256,253],[189,282],[314,293]])


# plist = np.array([[144,17],[149,20]]) # [362,23],[160,201],[354,213],[256,253],[189,282],[314,293],[221,311],[275,311]
# qlist = np.array([[150,49],[409,21],[160,201],[354,213],[256,253],[189,282],[314,293]])
# qlist = np.array([[145,18],[363,21],[160,201],[354,213],[256,253],[189,282],[314,293]])



# afMls = AF_MLS(img=img_0,tImg = img_1,plist = plist,qlist = qlist.copy())
# afMls.run()

plist = np.array([[144,17],[362,23],[160,201],[354,213],[256,253],[189,282],[314,293]])*0.5
#  # [362,23],[160,201],[354,213],[256,253],[189,282],[314,293],[221,311],[275,311]
qlist = np.array([[150,49],[398,33],[160,201],[354,213],[256,253],[189,282],[314,293]])*0.5
# # qlist = np.array([[145,18],[363,21],[160,201],[354,213],[256,253],[189,282],[314,293]])
# print('plist',plist.shape)

img = cv2.imread('./data/cat.jpg')
cv2.imshow('cat',img)

img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
afMls = AF_MLS(img=img,tImg = img,plist = plist,qlist = qlist.copy())
afMls.run()
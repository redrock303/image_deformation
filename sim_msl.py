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
class SIM_MLS:
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
        dimg = showImgPoints(img.copy(),self.plist,winName= 'src',color)
        dtImg = showImgPoints(dimg.copy(),self.qlist,winName= 'tar')
        newImg = img.copy()
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
                        newImg[int(self.qlist[k][1]),int(self.qlist[k][0])] = img[i,j]
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
                
                miu_s = 0.0
                left_term = v - p_star
                for k in range(self.n):
                    _p_hat_k = _p_hat[k]
                    _p_hat_k_t = np.transpose(_p_hat_k,(1,0))
                    miu_s += w[k] * _p_hat_k.dot(_p_hat_k_t)
                out = np.zeros((1,2),dtype=float)
                for ki in range(self.n):
                    _p_hat_ki = _p_hat[ki]
                    _neg_p_hat_ki_t = np.array([[_p_hat_ki[0,1],-1.0*_p_hat_ki[0,0]]])
                    left_mat = np.concatenate([_p_hat_ki,_neg_p_hat_ki_t])
                    # print('_p_hat_ki',_p_hat_ki,_neg_p_hat_ki_t,left_mat)
                    
                    neg_left_term_t = np.array([[left_term[0,1],-1.0*left_term[0,0]]])
                    right_mat = np.concatenate([left_term,neg_left_term_t])
                    right_mat_t = np.mat(right_mat).T
                    # print('right_mat',left_term,neg_left_term_t,right_mat,right_mat_t)
                    # input('check')
                    Ai = w[ki] * left_mat.dot(right_mat_t)
                    out_tmp = _q_hat[ki].dot(Ai)/miu_s
                    out += out_tmp
                out += q_star
                # print('out',out)
                # input('check')
                if out[0,0]>0 and out[0,0]<img.shape[1]-1 and out[0,1]>0 and out[0,1]<img.shape[0]-1:
                    x = int(out[0,0])
                    y = int(out[0,1])
                    newImg[y,x] = img[i,j]
        
        imgCompose = np.hstack([dimg,dtImg,newImg])
        cv2.imshow('imgCompose',imgCompose)
        cv2.imwrite('./data/cat_sim.png',imgCompose)
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



# afMls = SIM_MLS(img=img_0,tImg = img_1,plist = plist,qlist = qlist.copy())
# afMls.run()

# plist = np.array([[144,17],[362,23],[160,201],[354,213],[256,253],[189,282],[314,293]])*0.5
#  # [362,23],[160,201],[354,213],[256,253],[189,282],[314,293],[221,311],[275,311]
# qlist = np.array([[150,49],[409,21],[160,201],[354,213],[256,253],[189,282],[314,293]])*0.5
# # qlist = np.array([[145,18],[363,21],[160,201],[354,213],[256,253],[189,282],[314,293]])
# print('plist',plist.shape)
# img = cv2.imread('./data/cat.jpg')
# img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
# afMls = SIM_MLS(img=img,tImg = img,plist = plist,qlist = qlist.copy())
# afMls.run()


img = cv2.imread('/home/rui/remotePan/MLS/data/zk.png')
cv2.imshow('img',img)
cv2.waitKey(0)

# plist = np.array([[59,186],[189,186],[105,194],[145,193],[86,218],[163,221]]) #*0.5
# qlist = np.array([[71,181],[182,180],[116,193],[138,195],[93,212],[160,215]]) #*0.5

plist = np.array([[59,186],[189,186],[86,218],[163,221],[59,144],[194,139],[70,85],[176,85],[127,69]]) #*0.5
qlist = np.array([[71,181],[182,180],[93,212],[160,215],[63,135],[189,136],[76,88],[170,87],[127,79]]) #*0.5
afMls = SIM_MLS(img=img,tImg = img,plist = plist,qlist = qlist.copy())
afMls.run()
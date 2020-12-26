import numpy as np 
import math
def pointOnSegLine(a,b,v,eps = 1e-2):
    vec_ab = a - b
    
    if math.fabs(vec_ab[0,0])<eps and math.fabs(vec_ab[0,1])>eps:
        if v[0,0]<=max(a[0,0],b[0,0]) and v[0,0]>=min(a[0,0],b[0,0]):
            t = 1.0 
        else:
            return -1
        t = (v[0,1] - b[0,1])/(a[0,1] - b[0,1])
        # print('h')
        if t>=0 and t<=1 :
            return t 
        else:
            return -1
    elif math.fabs(vec_ab[0,1])<eps and math.fabs(vec_ab[0,0])>eps:
        if v[0,1]<=max(a[0,1],b[0,1]) and v[0,1]>=min(a[0,1],b[0,1]):
            t = 1.0 
        else:
            return -1
        t = (v[0,0] - b[0,0])/(a[0,0] - b[0,0])
        # print('v')
        if t>=0 and t<=1:
            return t 
        else:
            return -1
    else:
        # print('x')
        t0 = (v[0,1] - b[0,1])/(a[0,1] - b[0,1])
        t1 = (v[0,0] - b[0,0])/(a[0,0] - b[0,0])
        if math.fabs(math.fabs(t0) - math.fabs(t1))<eps:
            return (t0+t1)*0.5
        else:
            return -1

def pointOnSegLineV1(a,b,v,eps = 1e-6):
    vec_ab = a - b
    
    if math.fabs(vec_ab[0,0])<eps and math.fabs(vec_ab[0,1])>eps:
        if math.fabs(v[0,0] - a[0,0])<eps:
            return 1.0
        else:
            return -1
        
    elif math.fabs(vec_ab[0,1])<eps and math.fabs(vec_ab[0,0])>eps:
        if math.fabs(v[0,1] - a[0,1])<eps:
            return 1.0
        else:
            return -1
    else:
        # print('x')
        t0 = (v[0,1] - b[0,1])/(a[0,1] - b[0,1])
        t1 = (v[0,0] - b[0,0])/(a[0,0] - b[0,0])
        if math.fabs(math.fabs(t0) - math.fabs(t1))<eps:
            return (t0+t1)*0.5
        else:
            return -1

# self.p_list[k],lineNorm[k],v
def computerthigmaInLine(line_points,lineNorm,vertex):
    norm_5 = math.pow(lineNorm,5.0)
    a = line_points[[0]]
    b = line_points[[1]]
    v = vertex.copy()

    vb = v - b 
    ba_t = np.transpose(b-a,(1,0))
    av = a -v 

    dot_0 = vb.dot(ba_t)
    dot_1 = av.dot(ba_t)
    thigma_00 = norm_5 /      (3.0 *       dot_0     *     math.pow(dot_1,3.0))
    thigma_01 = -1.0*norm_5 / (6.0 * math.pow(dot_0,2.0) * math.pow(dot_1,2.0))
    thigma_11 = norm_5 / (3.0 * math.pow(dot_0,3.0) * dot_1)
    return thigma_00,thigma_01,thigma_11
def vec_neg(vec):
    return np.array([[-vec[0,1],vec[0,0]]])
def computerthigmaOutLine(line_points,lineNorm,vertex):
    a = line_points[[0]]
    b = line_points[[1]]
    v = vertex.copy()

    av = a -v 
    av_t = np.transpose(av,(1,0))

    av_neg = vec_neg(av)

    ab = a -b 
    ab_t = np.transpose(ab,(1,0))

    vb = v -b 
    vb_t = np.transpose(vb,(1,0))

    bv = b - v
    bv_neg = vec_neg(bv)

    ba = b - a 
    ba_t = np.transpose(ba,(1,0))

    delta = av_neg.dot(ab_t)
    bata_00 = av.dot(av_t)
    bata_01 = av.dot(vb_t)
    beta_11 = vb.dot(vb_t)
    # print(a,b,v)
    # print('ba_t)/(bv_neg.dot(ba_t)',bv_neg.dot(ba_t))
    # print(ba_t,bv_neg)
    # input('check')
    theta_left_in_up = (bv.dot(ba_t))/(bv_neg.dot(ba_t))
    theta_left = math.atan(theta_left_in_up)

    theta_right_in_up = (av.dot(ab_t))/(av_neg.dot(ab_t))
    theta_right = math.atan(theta_right_in_up)

    theta = theta_left - theta_right
    left_term = lineNorm / (2.0 * delta**2) 
    thigma_00 = left_term * (bata_01/bata_00 - (beta_11 * theta)/delta)
    thigma_01 = left_term * (1.0 - (bata_01 * theta)/delta)
    thigma_11 = left_term * (bata_01/beta_11 - (bata_00 * theta)/delta)
    return thigma_00,thigma_01,thigma_11
# a = np.array([[0,1]])
# b = np.array([[1,1]])

# v = np.array([[1.0,1]])
# t = pointOnSegLine(a,b,v)
# print('t',t)

# angle = math.sqrt(3)
# print(math.atan(angle)*180.0/3.14)
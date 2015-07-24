import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.morphology import binary_dilation
from mayavi import mlab
from copy import deepcopy
import cPickle

class model():
    def __init__(self,R,support):
        self.R = R
        self.size = 2*R+1
        self.shape = (self.size,self.size,self.size)
        self.array = np.random.rand(*self.shape)
        assert (self.array.shape == support.shape)
        filter = np.ones((3,3,3)) / 27
        self.array[~support] = 0
        self.array = convolve(self.array,filter)

    @mlab.show
    def show(self):
        max_ = np.max(self.array)
        mlab.pipeline.volume(mlab.pipeline.scalar_field(self.array), vmin=0.2*max_, vmax=0.8*max_)
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(self.array),
                            plane_orientation='x_axes',
                            slice_index=8)
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(self.array),
                            plane_orientation='y_axes',
                            slice_index=8)
        #mlab.contour3d(self.array, contours=8, opacity=0.3, transparent=True)
        mlab.outline()

    def rotate(self,rot_mat):
        displace = np.array([self.R,self.R,self.R])
        offset = -np.dot(rot_mat,displace) + displace
        self.array = affine_transform(self.array,rot_mat,offset)

    def clear(self):
        self.array = np.zeros(self.shape)

def rotate(m,mat):
    mm = deepcopy(m)
    mm.rotate(mat)
    return mm

def support2(R):
    size = 2 * R + 1
    s = np.zeros((size,size,size)).astype('bool')
    s[R, R, R-2:R+3] = True
    s[R, R-2:R+3, R] = True
    s[R-2:R+3, R, R] = True
    s = binary_dilation(s,iterations=5)
    return s

def support1(R):
    size = 2*R+1
    s = np.zeros((size,size,size)).astype('bool')
    s[R, R, R-5:R+5] = True
    s[R, R-4:R+4,R] = True
    s[R-4:R+4, R, R] = True
    s = binary_dilation(s,iterations=2)
    return s

def randomRotationMatrices(n):
    """
    Return a list of random rotation matrices
    """
    from random import sample
    (r, _) = rotationSamples(2000,1)
    return sample(r,n)

def makeCrossMatrix(vec):
    m = np.zeros((3,3))
    m[0,1], m[0,2] = -vec[2], vec[1]
    m[1,0], m[1,2] = vec[2], -vec[0]
    m[2,0], m[2,1] = -vec[1], vec[0]
    return m

def fibonacciSphere(n):
    """
    Generate n points on Fibonacci Sphere
    :return: a list of points
    """
    points = np.zeros((n,3))
    offset = 2./n
    increment = np.pi * (3. - np.sqrt(5.));
    for i in range(n):
        y = ((i * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - y * y)

        phi = ((i + 1) % n) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points[i] = np.array((x,y,z))
    return points

def rotationSamples(m,n):
    """
    Generate rotation samples according to a Fibonacci Sphere of m points
    Rotation around a certain axis is discretized into n parts
    :return: m*n samples of 3d rotation space
    """
    points = fibonacciSphere(m)
    rot_mats = []
    inv_rot_mats = []
    p0 = points[0,:]
    sin_value = np.sin(np.linspace(0, 2*np.pi, n))
    cos_value = np.cos(np.linspace(0, 2*np.pi, n)) # store the values to reuse
    for p1 in points:
        v = np.cross(p0,p1)
        s = np.linalg.norm(v) # sin of vectors
        c = np.dot(p0,p1) # cos of vectors
        v = makeCrossMatrix(v)
        vv = (1-c)/(s*s) * np.dot(v,v)
        if np.any(np.isnan(vv)):
            continue
        rot_mat = np.eye(3) + v + vv
        inv_rot_mat = np.eye(3) - v + vv
        for j in range(n):
            k = makeCrossMatrix(p1) # to rotate around vector p1 
            kk = np.dot(k,k)
            s, c = sin_value[j], cos_value[j]
            rot_axis = np.eye(3) + s*k + (1-c)*kk
            inv_rot_axis = np.eye(3) - s*k + (1-c)*kk
            rot_mats.append(np.dot(rot_mat, rot_axis))
            inv_rot_mats.append(np.dot(inv_rot_axis, inv_rot_mat))

    return rot_mats, inv_rot_mats


def makeData():
    import cPickle
    # generate random 2d data
    R = 8
    size = 2*R+1
    s = support1(R)
    m = model(R,s)
    m.show()
    M_DATA = 1000
    M_PIX = size**2
    exp_data = np.zeros((M_PIX,M_DATA))
    f = open('rotation_10000','rb')
    random_mat = cPickle.load(f)[0]

    for i in range(M_DATA):
        exp_data[:,i] = rotate(m,random_mat[i]).array[R,:,:].flatten()

    np.save('data'+str(M_DATA)+'_'+str(R),exp_data)
    f.close()

def randomMatrices(n):
    '''
    save and return a tuple of two lists of random rotation matrices and their inverse
    '''
    from numpy.linalg import qr
    import cPickle
    rot = []
    inv = []
    for i in range(n):
        q, r = qr(np.random.randn(3,3))
        d = np.diagonal(r)
        d = d/np.abs(d)
        q = np.multiply(q,d)
        if np.linalg.det(q) < 0:
            q = np.fliplr(q)
        try:
            iq = np.linalg.inv(q)
        except: # in case of q is singular
            i -=1
            continue
        rot.append(q)
        inv.append(iq)
    t = (rot,inv)
    with open('_'.join(['rotation', str(n)]), 'wb') as f:
            cPickle.dump(t, f)
    return t
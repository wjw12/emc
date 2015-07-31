"""
Some utility functions and test functions
Generating rotation matrices, data, model, mask, etc.
"""
import numpy as np

__author__ = 'Jiewen_Wang'

def fibonacciSphere(n):
    """
    Generate n points on Fibonacci Sphere
    Return a 2D array of points
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

def randomMatrices(n):
    """
    Save and return two lists of random rotation matrices and their inverse
    """
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
            q = np.fliplr(q) # make sure det > 0
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
    
def safe_log(x):
    """
    Numpy log function without nan or inf
    Return 0 if x <= 0
    """
    x = np.log(x)
    x[np.isnan(x) | np.isinf(x)] = 0
    return x

def makeCrossMatrix(vec):
    """
    Change a vector into cross product matrix
    """
    m = np.zeros((3,3))
    m[0,1], m[0,2] = -vec[2], vec[1]
    m[1,0], m[1,2] = vec[2], -vec[0]
    m[2,0], m[2,1] = -vec[1], vec[0]
    return m

def showProgress(total,current,char='#',length=75):
    """
    Show a progress bar in command line. Works only in Windows
    """
    import sys
    progress = int(current / total * 100)
    n = int(current / total * length)
    s = str(progress) + "% " + char * n
    f = sys.stdout
    f.write(s)
    f.flush()
    f.write('\r')
    if total == current:
        f.write('\n')

def sphereMask(R):
    """
    A sphere mask of radius R
    Return an boolean array
    """
    size = 2*R+1
    m = np.zeros((size,size,size)).astype('bool')
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if (i-R)**2 + (j-R)**2 + (k-R)**2 <= R*R:
                    m[i,j,k] = True
    return m

def mask2(R):
    size = 2 * R + 1
    s = np.zeros((size,size,size)).astype('bool')
    s[R, R, R-2:R+3] = True
    s[R, R-2:R+3, R] = True
    s[R-2:R+3, R, R] = True
    s = binary_dilation(s,iterations=15)
    return s

def mask1(R):
    size = 2*R+1
    s = np.zeros((size,size,size)).astype('bool')
    s[R, R, R-5:R+5] = True
    s[R, R-4:R+4,R] = True
    s[R-4:R+4, R, R] = True
    s = binary_dilation(s,iterations=2)
    return s

def makeData():
    # generate random 2d data
    R = 8
    size = 2*R+1
    s = mask1(R)
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

def makeFourierData():
    # generate random 2d data
    R = 8
    size = 17
    frame_size = 35
    s = mask1(R)
    m = model(R,s)
    m.show()
    M_DATA = 1000
    M_PIX = frame_size**2
    exp_data = np.zeros((M_PIX,M_DATA))
    f = open('rotation_10000','rb')
    random_mat = cPickle.load(f)[0]
    frame = np.zeros((frame_size,frame_size,frame_size))
    frame[9:26,9:26,9:26] = m.array
    frame = np.fft.fftshift(np.fft.fftn(frame)) # frame in the fourier space
    frame = np.abs(frame) ** 2 # 3d diffraction image
    for i in range(M_DATA):
        rot_mat = random_mat[i]
        displace = np.array([17,17,17])
        offset = -np.dot(rot_mat,displace) + displace
        img = affine_transform(frame,rot_mat,offset)
        exp_data[:,i] = img[17,:,:].flatten()

    np.save('fdata'+str(M_DATA)+'_'+str(R),exp_data)
    f.close()

def saveFourier():
    R = 8
    size = 17
    frame_size = 35
    s = mask1(R)
    m = model(R,s)
    frame = np.zeros((35,35,35))
    frame[9:26,9:26,9:26] = m.array
    frame = np.fft.fftshift(np.fft.fftn(frame)) # frame in the fourier space
    frame = np.abs(frame) ** 2 # 3d diffraction image
    np.save('fmodel',frame)
    plt.imshow(frame[10,:,:])
    plt.show()
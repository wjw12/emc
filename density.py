from particle import *

@mlab.show
def show(a):
    mlab.pipeline.volume(mlab.pipeline.scalar_field(a))

@mlab.show
def drawp(points):
    mlab.points3d(points[:,0],points[:,1],points[:,2])

def test0():
    # rotate a point
    a = 17
    r = 8
    p = np.ones((a,a))
    m = np.zeros((a,a,a))
    m[r,:,:] = p
    new_m = np.zeros((a,a,a))
    points = fibonacciSphere(500)
    rot = []
    p0 = points[0]
    eye = np.eye(3)
    for p1 in points[1:]:
        v = np.cross(p0,p1)
        s = np.linalg.norm(v) # sin of vectors
        c = np.dot(p0,p1) # cos of vectors
        v = makeCrossMatrix(v)
        vv = (1-c)/(s*s) * np.dot(v,v)
        rot_mat = np.eye(3) + v + vv
        rot.append(rot_mat)
    ro_points = np.zeros((500,3))
    ro_points[0] = p0
    for i in range(1,500):
        p_ = np.dot(rot[i-1], p0)
        ro_points[i] = p_
    drawp(ro_points)

def test():
    import matplotlib.pyplot as plt
    # rotate a plane, without rotating around an axis
    a = 17
    r = 8
    p = np.ones((a,a))
    m = np.zeros((a,a,a))
    m[:,r,:] = p
    new_m = np.zeros((a,a,a))
    points = fibonacciSphere(500)
    rot = []
    eye = np.eye(3)
    p0 = points[0]
    for p1 in points[1:]:
        v = np.cross(p0,p1)
        s = np.linalg.norm(v) # sin of vectors
        c = np.dot(p0,p1) # cos of vectors
        v = makeCrossMatrix(v)
        vv = (1-c)/(s*s) * np.dot(v,v)
        rot_mat = np.eye(3) + v + vv
        rot.append(rot_mat)

    show(m)
    # compress all planes of random directions by simply adding them together
    c = 0
    for i in rot:
        displace = np.array([r,r,r])
        offset = -np.dot(i,displace) + displace
        mm = affine_transform(m,i,offset)
        new_m += mm
        c += 1
    
    show(new_m)

    y = new_m[r,r,:]
    x = [i for i in range(len(y))]
    plt.plot(x,y)
    plt.show()

def test__():
    a = 17
    r = 8
    p = np.ones((a,a))
    m = np.zeros((a,a,a))
    m[:,r,:] = p
    new_m = np.zeros((a,a,a))
    points = fibonacciSphere(200)
    rot = []
    ang = 50
    sin_value = np.sin(np.linspace(0, 2*np.pi, ang))
    cos_value = np.cos(np.linspace(0, 2*np.pi, ang)) # store the values to reuse
    eye = np.eye(3)
    for p1 in points:
        k = makeCrossMatrix(p1)
        kk = np.dot(k,k)
        for i in range(ang):
            rot.append(eye + k*sin_value[i] + kk*(1-cos_value[i]))

    # compress all planes of random directions by simply adding them together
    for i in rot:
        displace = np.array([r,r,r])
        offset = -np.dot(i,displace) + displace
        mm = affine_transform(m,i,offset,order=5)
        new_m += mm
    show(m)
    show(new_m)

def test2(n):
    from numpy.linalg import qr
    a = 17
    R = 8
    p = np.ones((a,a))
    m = np.zeros((a,a,a))
    m[:,R,:] = p
    new_m = np.zeros((a,a,a))
    for i in range(n):
        q, r = qr(np.random.randn(3,3))
        d = np.diagonal(r)
        d = d/np.abs(d)
        q = np.multiply(q,d)
        displace = np.array([R,R,R])
        offset = -np.dot(q,displace) + displace
        mm = affine_transform(m,q,offset,order=5)
        new_m += mm
    show(new_m)
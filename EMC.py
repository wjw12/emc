import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.interpolation import rotate
from scipy.sparse import csc_matrix,lil_matrix
from scipy.ndimage import convolve

class EMC2D():
    def __init__(self,model="2.bmp",cxifile='115.cxi',samples=1000):
        self.M_ROT = 250
        self.M_DATA = samples
        self.NORM = 100000 # normalize factor
        self.model,self.WIDTH = self.loadBmp(model)
        self.M_PIX = self.WIDTH * self.WIDTH
        #self.getDataFromCxi(cxifile,samples)
        self.exp_data = load_sparse_csc('exp_data_025.npz')[:,:samples].astype('float64')
        self.generateRotationSpace()
        # normalize model
        self.model = self.normalizeImgArray(self.model)
        # normalize experimental data
        self.exp_data /= self.NORM

        self.prev_model = 0

    def getDataFromCxi(self,cxi,n): # get n frames from cxi file
        print ('Getting data from cxi file...')
        import h5py
        f = h5py.File(cxi)
        data = f['entry_1']['instrument_1']['detector_1']['data']
        assert (data.shape[1] == self.WIDTH)
        self.exp_data = lil_matrix((self.M_PIX,n))
        x = 150
        for i in range(x):
            showProgress(x,i)
            self.exp_data[:, i*n//x:(i+1)*n//x] = lil_matrix(data[i*n//x:(i+1)*n//x,:,:].reshape((self.M_PIX,n//x)).astype('float64'))
        self.exp_data = self.exp_data.tocsc()
        save_sparse_csc('exp_data',self.exp_data)
        print ('Data done.')
        
    def loadBmp(self,filename):
        import struct as st
        f = open(filename,'rb')
        data = bytearray(f.read())
        offsetToArray = st.unpack_from('I',data,10)[0]
        width = st.unpack_from('I',data,18)[0]
        height = st.unpack_from('I',data,22)[0]
        assert(width == height),"image must have same width and height"
        bitsPerPixel = st.unpack_from('H',data,28)[0]
        bytesPerPixel = bitsPerPixel // 8
        bytesPerRow = bytesPerPixel * width

        # store the pixel array (in greyscale)
        pixels = []
        for row in range(height):
            for col in range(width):
                offset = offsetToArray + row * bytesPerRow + col * bytesPerPixel
                b = st.unpack_from('B',data,offset)[0]
                g = st.unpack_from('B',data,offset+1)[0]
                r = st.unpack_from('B',data,offset+2)[0]
                pixels.append((b+g+r)//3)

        pixels = np.array(pixels).reshape((width,width))
        return pixels, width

    def rotateImg(self,image,angle):
        return self.normalizeImgArray( rotate(self.unnormalizeImgArray(image),angle,reshape=False) )

    def generateRotationSpace(self):
        self.rotation = np.linspace(0,360,self.M_ROT)

    def normalizeImgArray(self,img):
        return np.float64(img) / self.NORM

    def unnormalizeImgArray(self,img):
        return np.float64(img * self.NORM)

    def expand(self):
        result = np.zeros((self.M_PIX,self.M_ROT))
        for j in range(self.M_ROT):
            rot_img = self.rotateImg(self.model,self.rotation[j]).reshape(self.M_PIX,1)
            result[:,j] = rot_img[:,0]
        self.intensities = result

    def cond_prob(self):
        log_W = np.log(self.intensities.T)
        log_W[np.isinf(log_W) | np.isnan(log_W)] = -100
        W = np.exp(log_W)
        A = csc_matrix(log_W) * self.exp_data
        A = A.toarray()
        prob = A - np.tile(np.sum(W,1), (self.M_DATA,1)).T # log probability
        prob -= np.max(prob)
        prob = np.exp(prob)
        S = np.sum(prob, 0)
        prob = prob / np.tile(S, (self.M_ROT,1))
        return prob


    def EM(self,row_n=200,col_n=2):
        P = self.cond_prob()
        
        for i in range(self.M_DATA):
            ind = np.argpartition(P[:,i], self.M_ROT-col_n)
            ind1 = ind[:self.M_ROT-col_n]
            ind2 = ind[-col_n:]
            P[:,i][ind1] = 1e-80
        for i in range(self.M_ROT):
            ind = np.argpartition(P[i,:], self.M_DATA-row_n)
            ind1 = ind[:self.M_DATA-row_n]
            ind2 = ind[-row_n:]
            P[i,:][ind1] = 1e-80
        # test P
        
        #c = P != 1e-80
        #a = np.where(np.sum(c,axis=0) == col_n)
        #i = a[0][0]
        '''
        ind = np.argpartition(P[0,:], self.M_DATA-row_n)
        ind1 = ind[:self.M_DATA-row_n]
        ind2 = ind[-row_n:]
        p = np.sort(P[0,:][ind2])
        x = [i for i in range(len(p))]
        plt.plot(x,p)
        plt.show()
        '''
        w = np.max(P,1)
        maxw = np.max(w)
        minw = np.min(w)
        delta = 1e-50
        if maxw - minw < delta:
            self.weight = np.ones(w.shape)
        else:
            self.weight = (w - np.min(w)) / (np.max(w) - np.min(w)) # 1*M_ROT array, weight for compression
        # j-th element represents weight for j-th intensity

        new_inten = self.exp_data*csc_matrix(P.T)
        new_inten = new_inten.toarray()
        weight = np.tile( np.max(P,1), (self.M_PIX, 1))
        new_inten *= weight

        S = np.sum(P,1)
        new_inten /= np.tile(S, (self.M_PIX,1))
        self.intensities = new_inten

    def compress(self):
        self.prev_model = self.model
        new_inten = self.intensities * np.tile(self.weight, (self.M_PIX,1))
        r_img = np.reshape(new_inten,(self.WIDTH,self.WIDTH,self.M_ROT))
        model = np.zeros((self.WIDTH,self.WIDTH))
        for j in range(self.M_ROT):
            re_img = self.rotateImg(r_img[:,:,j], -self.rotation[j])
            model += re_img
        model = model / self.M_ROT
        self.model = model


    def run(self,iterations):
        log_file = open('diff.log','w')
        for it in range(iterations):
            print ("Iteration ",it+1)
            self.expand()
            self.EM()
            self.compress()
            # bluring the model
            k = np.ones((3,3)) / 9
            self.model = convolve(self.model,k)
            self.save_model(it+1)
            # calculate the difference
            diff = np.mean(np.abs(self.model - self.prev_model))
            print (diff, file=log_file)
            if it==iterations-1:
                self.show(1,1)
        print ('Done.')

    def runModel(self,model_file,iterations,curr_it=1):
        self.model = np.load(model_file)
        log_file = open('diff.log','a')
        for it in range(iterations):
            print("Iteration ",it+curr_it)
            self.expand()
            self.EM()
            self.compress()
            # bluring the model
            k = np.ones((3,3)) / 9
            self.model = convolve(self.model,k)
            self.save_model(it+curr_it)
            # calculate the difference
            diff = np.mean(np.abs(self.model - self.prev_model))
            print (diff, file=log_file)
            if it==iterations-1:
                self.show(1,1)
        print ('Done.')
    
    def show(self,total,subplot):
        plt.subplot(1,total,subplot)
        model = self.unnormalizeImgArray(self.model)
        img_plot = plt.imshow(np.abs(model), cmap=cm.Greys_r)
        img_plot.set_clim(0.0, np.max(model))
        if subplot == total:
            plt.show()

    def save_model(self,n):
        np.save('model_' + str(n),self.model)


def showProgress(total,current,char='#',length=75):
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

def save_sparse_csc(filename,array):
    np.savez(filename,data = array.data, indices = array.indices, indptr = array.indptr, shape = array.shape)

def load_sparse_csc(filename):
    loader = np.load(filename)
    return csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

def convertData(dat_file,detector_file):
    dat = open(dat_file)
    detector = open(detector_file)
    i = iter(dat)
    frames = int(i.readline())
    next(i)
    next(i) # skip 2 lines
    print ('Total frames:', frames)
    ii = iter(detector)
    pixels = int(ii.readline())
    print ('Pixels in the detector:',pixels)
    det_dict = {} # store the coordinates in a dictionary
    for j in range(pixels):
        coo = tuple(ii.readline().split('\t'))
        coo = (90 - int(coo[1])) * 181 + (int(coo[0]) + 90)
        det_dict[j] = int(coo)
    exp_data = lil_matrix((181*181,frames),dtype='uint8')
    for frame in range(frames):
        locations = [int(n) for n in i.readline().split(' ') if n != '\n']
        for n in locations:
            exp_data[det_dict[n],frame] = 1
        try:
            next(i)
            next(i)
            next(i)
            next(i) # skip 4 lines
        except:
            break
        
    exp_data = exp_data.tocsc()
    save_sparse_csc('exp_data',exp_data)

    # test plotting
    img = exp_data.sum(axis=1).reshape((181,181))
    img_plot = plt.imshow(img, cmap = cm.Greys_r)
    img_plot.set_clim(0,np.max(img))
    plt.show()

if __name__ == '__main__':
    emc1 = EMC2D(samples=450000)
    #emc1.run(50)
    emc1.runModel('model_50.npy',50,51)
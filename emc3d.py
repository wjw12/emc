from __future__ import division, print_function
from particle import *
import matplotlib.pyplot as plt

class emc3d():
    def __init__(self,datafile):
        self.exp_data = np.load(datafile)
        self.M_PIX = self.exp_data.shape[0]
        self.M_DATA = self.exp_data.shape[1]
        self.size = int(np.sqrt(self.M_PIX))
        self.R = int((self.size - 1) / 2)
        self.NORM = 1000000 # normalize factor

        # generate rotation space
        with open('rotation_10000') as f:
            self.rotation, self.inv_rotation = cPickle.load(f)
            self.rotation = self.rotation[1000:3000]
            self.inv_rotation = self.inv_rotation[1000:3000]
            self.M_ROT = len(self.rotation)

        # create initial model
        s = support2(self.R)
        self.model = model(self.R,s)
        # filtering
        k = np.ones((3,3,3)) / 27
        for i in range(4):
            self.model.array = convolve(self.model.array, k)
        self.model.show()

        # normalize
        self.model.array /= self.NORM
        self.exp_data /= self.NORM

        # density weight for compression
        # generate once and for all
        '''
        plane = np.ones((self.size,self.size))
        m = np.zeros((self.size,self.size,self.size))
        m[self.R, :, :] = plane
        self.density = np.zeros((self.size,self.size,self.size))
        for rot in self.inv_rotation:
            displace = np.array([self.R,self.R,self.R])
            offset = -np.dot(rot,displace) + displace
            mm = affine_transform(m,rot,offset,order=4)
            self.density += mm
        np.save('_'.join(['density',str(self.M_ROT)]),self.density)
        '''
        self.density = np.load('_'.join(['density',str(self.M_ROT)]) + '.npy')

        # a sphere mask to filter out margin
        self.mask = np.load('mask.npy')

    def expand(self):
        result = np.zeros((self.M_PIX,self.M_ROT))
        displace = np.array([self.R,self.R,self.R])
        for j in range(self.M_ROT):
            #showProgress(self.M_ROT, j)
            off = -np.dot(self.rotation[j], displace) + displace
            m = affine_transform(self.model.array, self.rotation[j], off, order=4)
            result[:,j] = m[self.R,:,:].flatten()
        self.reference = result


    def compress(self):
        self.model.clear()
        for i in range(self.M_ROT):
            #showProgress(self.M_ROT, i)
            piece = self.reference[:,i].reshape((self.size,self.size))
            piece_3d = np.zeros((self.size, self.size, self.size))
            piece_3d[self.R, :,:] = piece
            # rotate the piece
            displace = np.array([self.R,self.R,self.R])
            offset = -np.dot(self.inv_rotation[i],displace) + displace
            piece_3d = affine_transform(piece_3d, self.inv_rotation[i], offset, order=4)
            self.model.array += piece_3d
        self.model.array = self.model.array / self.density
        self.model.array[np.isnan(self.model.array)] = 0
        self.model.array[~self.mask] = 0
        self.model.array[np.where(self.model.array < 0.001*np.max(self.model.array))] = 0
         
    def cond_prob(self):
        log_W = np.log(self.reference.T)
        log_W[np.isinf(log_W) | np.isnan(log_W)] = -100
        W = np.exp(log_W)
        A = np.dot(log_W, self.exp_data)
        prob = A - np.tile(np.sum(W,1), (self.M_DATA,1)).T # log probability
        prob -= np.max(prob)
        prob = np.exp(prob)
        S = np.sum(prob, 0)
        prob = prob / np.tile(S, (self.M_ROT,1))
        return prob


    def EM(self,row_n=2,col_n=1700): # row_n selects data for a certain rotation
        P = self.cond_prob()
        # normalize P
        P = (P - np.min(P)) / (np.max(P) - np.min(P))

        for i in range(self.M_DATA):
            ind1 = np.argpartition(P[:,i], self.M_ROT-col_n)[:self.M_ROT-col_n]
            P[:,i][ind1] = 1e-80
        
        for i in range(self.M_ROT):
            ind = np.argpartition(P[i,:], self.M_DATA-row_n)
            ind1 = ind[:self.M_DATA-row_n]
            P[i,:][ind1] = 1e-80

        new_refer = np.dot(self.exp_data,P.T)
        S = np.sum(P,1)
        new_refer /= np.tile(S, (self.M_PIX,1))
        self.reference = new_refer

    def run(self,iterations):
        for it in range(iterations):
            print ("Iteration ",it+1)
            self.expand()
            self.EM()
            self.compress()
            self.saveModel(it+1)
            if it==iterations-1:
                self.model.show()
        print ('Done.')

    def runModel(self,start,iterations):
        '''
        Run from previous model, given the starting iteration
        '''
        self.loadModel('_'.join(['model',str(start-1), str(self.R)]))
        for it in range(start, start + iterations):
            print ("Iteration ",it)
            self.expand()
            self.EM()
            self.compress()
            self.saveModel(it)
            self.model.show()
        print ('Done.')

    def saveModel(self, n):
        with open('_'.join(['model',str(n), str(self.model.R)]), 'wb') as f:
            cPickle.dump(self.model, f)

    def loadModel(self, filename):
        with open(filename, 'rb') as f:
            self.model = cPickle.load(f)

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

@mlab.show
def show(a):
    mlab.pipeline.volume(mlab.pipeline.scalar_field(a))

if __name__ == '__main__':
    emc = emc3d('data1000_8.npy')
    emc.run(8)
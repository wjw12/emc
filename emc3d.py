from __future__ import division, print_function
from particle import *
from utilities import *
"""
An implementation of EMC algorithm. Can be used for reconstructing 3D structure from simulated data

Reference: Ne-Te Duane Loh and Veit Elser, 
           "A reconstruction algorithm for single-particle diffraction imaging experiments",
           Physical Review E, 2009

Author: Jiewen Wang
        2015.7
        at Beijing Computational Science Research Center
"""

__author__ = 'Jiewen_Wang'

class emc3d():
    """
    A class to wrap up and run EMC algorithm
    """
    def __init__(self,datafile):
        self.exp_data = np.load(datafile)

        # pixels in a 2D diffraction image
        self.M_PIX = self.exp_data.shape[0]

        # frames of data
        self.M_DATA = self.exp_data.shape[1]

        # side length of the image
        self.size = int(np.sqrt(self.M_PIX))

        # radius
        self.R = int((self.size - 1) / 2)

        # generate rotation space
        with open('rotation_10000') as f:
            self.rotation, self.inv_rotation = cPickle.load(f)
            self.rotation = self.rotation[1000:3000]
            self.inv_rotation = self.inv_rotation[1000:3000]
            self.M_ROT = len(self.rotation) # number of rotation samples

        # a sphere mask to filter out margin
        try:
            self.mask = np.load('mask_' + str(self.R) + '.npy')
        except:
            self.mask = sphereMask(self.R)

        # create initial model
        self.model = model(self.R,self.mask)

        # filtering
        k = np.ones((3,3,3)) / 27
        for i in range(4):
            self.model.array = convolve(self.model.array, k)
        self.model.array = np.fft.fftshift(np.fft.fftn(self.model.array))
        self.model.array = np.abs(self.model.array) ** 2
        #self.model.show()

        # a scale factor to avoid overflow or underflow when calculating probability
        # should be carefully picked
        self.SCALE = 10000
        self.model.array /= self.SCALE
        self.exp_data /= self.SCALE

        # density weight for compression
        # generate once and for all
        try:
            self.density = np.load('_'.join(['density',str(self.M_ROT),str(self.R)]) + '.npy')
        except:
            plane = np.ones((self.size,self.size))
            m = np.zeros((self.size,self.size,self.size))
            m[self.R, :, :] = plane
            self.density = np.zeros((self.size,self.size,self.size))
            for rot in self.inv_rotation:
                displace = np.array([self.R,self.R,self.R])
                offset = -np.dot(rot,displace) + displace
                mm = affine_transform(m,rot,offset,order=4)
                self.density += mm
            np.save('_'.join(['density',str(self.M_ROT),str(self.R)]),self.density)
        
    def expand(self):
        # result is 2D array
        # each column represent a reference image (flattened to 1D)
        result = np.zeros((self.M_PIX,self.M_ROT))
        displace = np.array([self.R,self.R,self.R])
        for j in range(self.M_ROT):
            #showProgress(self.M_ROT, j)
            # rotate the model and cut a slice
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
         
    def cond_prob(self):
        """
        Conditional probability based on Poisson statistics
        Vectorized version
        Return a probability matrix of shape (M_ROT, M_DATA)
        """
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

    def maximize(self,row_n=3,col_n=450): 
        """
        Something like expectation-maximazation
        Update the reference using experimental data with the knowledge of probability
        but selects several most likely items
        row_n selects the most likely data for a certain reference orientation
        col_n determines how many reference orientations a image frame can be mapped into

        can be estimated using the code below
        though empirical adjustments are usually needed
        
        import matplotlib.pyplot as plt
        ind = np.argpartition(P[:,0], self.M_ROT-col_n)
        ind2 = ind[-col_n:]
        p = np.sort(P[:,0][ind2])
        x = [i for i in range(len(p))]
        plt.plot(x,p)
        plt.show()
        ind = np.argpartition(P[0,:], self.M_DATA-row_n)
        ind2 = ind[-row_n:]
        p = np.sort(P[0,:][ind2])
        x = [i for i in range(len(p))]
        plt.plot(x,p)
        plt.show()
        """
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
            self.maximize()
            self.compress()
            self.saveModel(it+1)
            if it==iterations-1:
                self.model.showLog()
        print ('Done.')

    def runModel(self,start,iterations):
        '''
        Run from previous model, given the starting iteration
        '''
        self.loadModel('_'.join(['model',str(start-1), str(self.R)]))
        for it in range(start, start + iterations):
            print ("Iteration ",it)
            self.expand()
            self.maximize()
            self.compress()
            self.saveModel(it+1)
        print ('Done.')

    def saveModel(self, n):
        with open('_'.join(['model',str(n), str(self.model.R)]), 'wb') as f:
            cPickle.dump(self.model, f)

    def loadModel(self, filename):
        with open(filename, 'rb') as f:
            self.model = cPickle.load(f)


if __name__ == '__main__':
    emc = emc3d('fdata1000_17.npy')
    emc.run(12)
from __future__ import division, print_function
from copy import deepcopy
from particle import *
import matplotlib.pyplot as plt
import cPickle

class emc3d():
    def __init__(self,datafile):
        self.exp_data = np.load(datafile)
        self.M_PIX = self.exp_data.shape[0]
        self.M_DATA = self.exp_data.shape[1]
        self.size = int(np.sqrt(self.M_PIX))
        self.R = int((self.size - 1) / 2)
        self.NORM = 1000000 # normalize factor

        # generate rotation space
        self.rotation, self.inv_rotation = rotationSamples(1000,1)
        self.M_ROT = len(self.rotation)
        print (self.M_ROT)

        # create random model
        s = support1(self.R)
        self.model = model(self.R,s)
        #self.model.show()

        # normalize
        self.model.array /= self.NORM
        self.exp_data /= self.NORM

        # density weight for compression
        r = np.arange(0,self.size)
        x, y, z = np.meshgrid(r,r,r)
        R = self.R
        self.density = 1./np.sqrt(((x-R)**2 + (y-R)**2 + (z-R)**2 + 1))
        self.density /= np.mean(self.density)

    def expand(self):
        result = np.zeros((self.M_PIX,self.M_ROT))
        displace = np.array([self.R,self.R,self.R])
        for j in range(self.M_ROT):
            #showProgress(self.M_ROT, j)
            off = -np.dot(self.rotation[j], displace) + displace
            m = affine_transform(self.model.array, self.rotation[j], off, order=5)
            result[:,j] = m[self.R,:,:].flatten()
        self.reference = result


    def compress(self):
        self.model.clear()
        #self.reference = self.reference * np.tile(self.weight, (self.M_PIX,1))
        for i in range(self.M_ROT):
            #showProgress(self.M_ROT, i)
            piece = self.reference[:,i].reshape((self.size,self.size))
            piece_3d = np.zeros((self.size, self.size, self.size))
            piece_3d[self.R, :,:] = piece
            # rotate the piece
            displace = np.array([self.R,self.R,self.R])
            offset = -np.dot(self.inv_rotation[i],displace) + displace
            piece_3d = affine_transform(piece_3d, self.inv_rotation[i], offset, order=5)
            self.model.array += piece_3d
        self.model.array = self.model.array / self.density
         
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


    def EM(self,row_n=100,col_n=700):
        P = self.cond_prob()
        # add contrast of P
        #P = (P - np.min(P)) / (np.max(P) - np.min(P))

        p = np.sum(P,1)
        x = [i for i in range(len(p))]
        plt.plot(x,p)
        plt.show()

        for i in range(self.M_ROT):
            ind = np.argpartition(P[i,:], self.M_DATA-row_n)
            ind1 = ind[:self.M_DATA-row_n]
            ind2 = ind[-row_n:]
            P[i,:][ind1] = 1e-80

        '''
        ind = np.argpartition(P[10,:], self.M_ROT-row_n)
        ind2 = ind[-row_n:]
        p = np.sort(P[10,:][ind2])
        x = [i for i in range(len(p))]
        plt.plot(x,p)
        plt.show()
        '''
        
        for i in range(self.M_DATA):
            ind1 = np.argpartition(P[:,i], self.M_ROT-col_n)[:self.M_ROT-col_n]
            #ind2 = np.argpartition(P[:,i], 980)[-20:]
            P[:,i][ind1] = 1e-80
            #P[:,i][ind2] = 1e-80
        
        w = np.max(P,1)
        maxw = np.max(w)
        minw = np.min(w)
        delta = 1e-50
        if maxw - minw < delta:
            self.weight = np.ones(w.shape)
        else:
            self.weight = (w - np.min(w)) / (np.max(w) - np.min(w)) # 1*M_ROT array, weight for compression
        # j-th element represents weight for j-th reference

        new_refer = np.dot(self.exp_data,P.T)
        #weight = np.tile( np.max(P,1), (self.M_PIX, 1))
        #new_refer *= weight

        S = np.sum(P,1)
        new_refer /= np.tile(S, (self.M_PIX,1))
        self.reference = new_refer

    def run(self,iterations):
        for it in range(iterations):
            print ("Iteration ",it+1)
            self.expand()
            self.EM()
            self.compress()
            # bluring the model
            k = np.ones((3,3,3)) / 27
            self.model.array = convolve(self.model.array,k)
            self.saveModel(it+1)
            if it+1 == iterations:
                self.model.array *= self.NORM
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

if __name__ == '__main__':
    emc = emc3d('data1000_8.npy')
    emc.run(8)
"""
Define a 3D model
Can be either a particle in real space
or Fourier transformd intensities in the Fourier space
"""

from utilities import *
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.morphology import binary_dilation
from mayavi import mlab
from copy import deepcopy
import cPickle

__author__ = 'Jiewen_Wang'

class model():
    """
    A particle model with radius R.
    Use a mask to filter out marginal areas of the grid
    """
    def __init__(self,R,mask):
        self.R = R
        self.size = 2*R+1
        self.shape = (self.size,self.size,self.size)
        self.array = np.random.rand(*self.shape)
        assert (self.array.shape == mask.shape)
        filter = np.ones((3,3,3)) / 27
        self.array[~mask] = 0
        self.array = convolve(self.array,filter)

    def show(self):
        """
        Display real space model using volume
        """
        max_ = np.max(self.array)
        mlab.pipeline.volume(mlab.pipeline.scalar_field(self.array), vmin=0.2*max_, vmax=0.9*max_)
        mlab.outline()
        mlab.show()

    def showContour(self):
        """
        Display real space model using contour
        """
        mlab.contour3d(self.array, contours=8, opacity=0.3, transparent=True)
        mlab.outline()
        mlab.show()

    def showLog(self):
        """
        Display log intensity. Should use this to display a model of Fourier space
        """
        l = safe_log(self.array)
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(l),
                            plane_orientation='x_axes',
                            slice_index=self.R)
        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(l),
                            plane_orientation='y_axes',
                            slice_index=self.R)
        mlab.outline()
        mlab.show()

    def rotate(self,rot_mat):
        """
        Rotate the 3D model
        """
        displace = np.array([self.R,self.R,self.R])
        offset = -np.dot(rot_mat,displace) + displace
        self.array = affine_transform(self.array,rot_mat,offset)

    def clear(self):
        self.array = np.zeros(self.shape)

def rotate(m,mat):
    mm = deepcopy(m)
    mm.rotate(mat)
    return mm

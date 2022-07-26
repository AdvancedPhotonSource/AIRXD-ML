# Import libraries
import numpy as np
import numpy.ma as ma
from cffi import FFI
import scipy.special as sc
if __name__ == "__main__":
    from _mask import lib
else:
    from ._mask import lib
import warnings
warnings.filterwarnings('ignore')

npcosd = lambda x: np.cos(x*np.pi/180.)
npsind = lambda x: np.sin(x*np.pi/180.)
nptand = lambda x: np.tan(x*np.pi/180.)
npatand = lambda x: 180.*np.arctan(x)/np.pi
npatan2d = lambda y,x: 180.*np.arctan2(y,x)/np.pi

class MASK:
    def __init__(self, controls, shape):
        self.shape = shape
        self.controls = controls
        self.TA = self.Make2ThetaAzimuthMap(self.controls,(0,shape[0]),(0,shape[1]))[0]

    def AutoSpotMask(self, image, esdmul=3.0, numchans=445):
        assert image.shape == self.shape, f"The image shape is different from the declared shape: {self.shape}"

        # Additional masking
        masks = {'Frames': None}
        frame = masks['Frames']
        tam = ma.make_mask_none(image.shape)

        LUtth = np.array(self.controls['IOtth'])
        dtth = (LUtth[1]-LUtth[0])/numchans
        TThs = np.linspace(LUtth[0], LUtth[1], numchans, False)
        band = np.array(image)

        ffi = FFI()
        m, n = tam.shape
        p = TThs.shape[0]
        ptam =  ffi.new('int['+str(m*n)+']', tam.ravel().tolist())
        pta = ffi.new('double['+str(m*n)+']', self.TA.ravel().tolist())
        pband = ffi.new('double['+str(m*n)+']', band.ravel().tolist())
        ptths = ffi.new('double['+str(p)+']', TThs.tolist())
        output = ffi.new('double['+str(m*n)+']')
        lib.mask(m, n, p, dtth, esdmul, ptam, pta, pband, ptths, output)
        mask = np.frombuffer(ffi.buffer(output, m*n*8), dtype=np.float64)
        mask.shape = (m, n)

        # Release memory
        ffi.release(ptam)
        ffi.release(pta)
        ffi.release(pband)
        ffi.release(ptths)
        ffi.release(output)

        return mask

    def peneCorr(self, tth, dep, dist):
        return dep*(1.-npcosd(tth))*dist**2/1000.

    def makeMat(self, Angle, Axis):
        '''Make rotation matrix from Angle and Axis
        :param float Angle: in degrees
        :param int Axis: 0 for rotation about x, 1 for about y, etc.
        '''
        cs = npcosd(Angle)
        ss = npsind(Angle)
        M = np.array(([1.,0.,0.],[0.,cs,-ss],[0.,ss,cs]),dtype=np.float32)
        return np.roll(np.roll(M,Axis,axis=0),Axis,axis=1)

    def Polarization(self, Pola, Tth, Azm=0.0):
        """   Calculate angle dependent x-ray polarization correction (not scaled correctly!)

        :param Pola: polarization coefficient e.g 1.0 fully polarized, 0.5 unpolarized
        :param Azm: azimuthal angle e.g. 0.0 in plane of polarization - can be numpy array
        :param Tth: 2-theta scattering angle - can be numpy array
          which (if either) of these is "right"?
        :return: (pola, dpdPola) - both 2-d arrays
          * pola = ((1-Pola)*npcosd(Azm)**2+Pola*npsind(Azm)**2)*npcosd(Tth)**2+ \
            (1-Pola)*npsind(Azm)**2+Pola*npcosd(Azm)**2
          * dpdPola: derivative needed for least squares

        """
        cazm = npcosd(Azm)**2
        sazm = npsind(Azm)**2
        pola = ((1.0-Pola)*cazm+Pola*sazm)*npcosd(Tth)**2+(1.0-Pola)*sazm+Pola*cazm
        dpdPola = -npsind(Tth)**2*(sazm-cazm)
        return pola,dpdPola

    def GetTthAzmG2(self, x, y, data):
        '''Give 2-theta, azimuth & geometric corr. values for detector x,y position;
         calibration info in data - only used in integration - old version
        '''
        'Needs a doc string - checked OK for ellipses & hyperbola'
        tilt = data['tilt']
        dist = data['distance']/npcosd(tilt)
        MN = -np.inner(self.makeMat(data['rotation'],2), self.makeMat(tilt,0))
        dx = x-data['center'][0]
        dy = y-data['center'][1]
        dz = np.dot(np.dstack([dx.T,dy.T,np.zeros_like(dx.T)]),MN).T[2]
        xyZ = dx**2+dy**2-dz**2
        tth0 = npatand(np.sqrt(xyZ)/(dist-dz))
        dzp = self.peneCorr(tth0, data['DetDepth'], dist)
        tth = npatan2d(np.sqrt(xyZ),dist-dz+dzp)
        azm = (npatan2d(dy,dx)+data['azmthOff']+720.)%360.
        distsq = data['distance']**2
        x0 = data['distance']*nptand(tilt)
        x0x = x0*npcosd(data['rotation'])
        x0y = x0*npsind(data['rotation'])
        G = ((dx-x0x)**2+(dy-x0y)**2+distsq)/distsq
        return tth,azm,G

    def Make2ThetaAzimuthMap(self, data, iLim, jLim):
        'Needs a doc string'
        pixelSize = data['pixelSize']
        scalex = pixelSize[0]/1000.
        scaley = pixelSize[1]/1000.
        tay,tax = np.mgrid[iLim[0]+0.5:iLim[1]+.5,jLim[0]+.5:jLim[1]+.5]
        tax = np.asfarray(tax*scalex,dtype=np.float32).flatten()
        tay = np.asfarray(tay*scaley,dtype=np.float32).flatten()
        nI = iLim[1]-iLim[0]
        nJ = jLim[1]-jLim[0]
        TA = np.empty((4,nI,nJ))
        TA[:3] = np.array(self.GetTthAzmG2(np.reshape(tax,(nI,nJ)),np.reshape(tay,(nI,nJ)),data))
        TA[1] = np.where(TA[1]<0,TA[1]+360,TA[1])
        TA[3] = self.Polarization(data['PolaVal'][0],TA[0],TA[1]-90.)[0]
        return TA

if __name__ == "__main__":
    import imageio
    from utilities import parse_imctrl
    numChans = 445
    Controls = parse_imctrl('../data/Si_ch3_d700-00000.imctrl')
    Image = imageio.volread('../data/Ni83_ch3_RTto950_d700-00005.tif')
    mask = MASK(Controls, shape=(2880, 2880))
    result = mask.AutoSpotMask(Image)
    np.save('mask', result)

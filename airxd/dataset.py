import os
import imageio
import numpy as np
from glob import glob

npcosd = lambda x: np.cos(x*np.pi/180.)
npsind = lambda x: np.sin(x*np.pi/180.)
nptand = lambda x: np.tan(x*np.pi/180.)
npatand = lambda x: 180.*np.arctan(x)/np.pi
npatan2d = lambda y,x: 180.*np.arctan2(y,x)/np.pi

class Dataset:
    def __init__(self, n=1, shape=(2880, 2880)):
        ''' n is the number of experiments. ''' 
        self.n = n
        self.shape = shape
        self.TAs = {}
        self.images = {}
        self.labels = {}
        self.controls = []

    def get_data(self, directory_names, image_ext='.tif', label_ext='.tif', ctrl_ext='.imctrl'):
        ''' get images from directories (directory_names). '''
        msg = "The number of experiments (n) doesn't match with number of directories"
        assert self.n == len(directory_names), msg
        
        # get angle maps
        for i, path in enumerate(directory_names):
            tpath = glob(os.path.join(path, f'*{ctrl_ext}'))
            controls = parse_imctrl(tpath[0])
            self.controls.append(controls)
            self.TAs[i] = np.zeros((self.shape[0], self.shape[1]))
            ta = self.Make2ThetaAzimuthMap(controls, 
                                        (0, self.shape[0]), (0, self.shape[1]))[0]
            self.TAs[i] += ta

        # get images
        for i, path in enumerate(directory_names):
            ipath = sorted(glob(os.path.join(path, f'*{image_ext}')))
            self.images[i] = np.zeros((len(ipath), self.shape[0], self.shape[1]))
            for j, ip in enumerate(ipath):
                self.images[i] += imageio.volread(ip)

        # get labels
        for i, path in enumerate(directory_names):
            lpath = sorted(glob(os.path.join(path, 'masks', 'f*{label_ext}')))
            self.labels[i] = np.zeros((len(lpath), self.shape[0], self.shape[1]))
            for j, lp in enumerate(lpath):
                if label_ext == '.tif':
                    self.labels[i] += imageio.volread(lp)
                else:
                    self.labels[i] += np.load(lp)

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

def parse_imctrl(filename):
    controls = {'size': [2880, 2880], 'pixelSize': [150.0, 150.0]}
    keys = ['IOtth', 'PolaVal', 'azmthOff', 'rotation', 'distance', 'center', 'tilt', 'DetDepth']
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ln = line.split(':')
            if ln[0] in keys:
                if ln[1][0] == '[':
                    temp = []
                    temp_list = ln[1].split(',')
                    temp.append(float(temp_list[0][1:]))
                    try:
                        temp.append(float(temp_list[1][:-2]))
                    except:
                        temp.append(False)
                    controls[ln[0]] = temp
                else:
                    controls[ln[0]] = float(ln[1])

    return controls

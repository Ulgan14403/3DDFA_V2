import numpy as np
import math
import scipy 

config_mini_x = {
    'freq' : 30,
    'mincutoff' :1,
    'gamma' : 0,
    'dcutoff' : 1 
    }

config_mini_y = {
    'freq' : 30,
    'mincutoff' :1e-4,
    'gamma' : 1e-5,
    'dcutoff' : 1 
    }


# ----------------------------------------------------------------------------

class LowPassFilter(object):

    def __init__(self, alpha:float) -> None:
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha:float) -> None:
        alpha = float(alpha)
        if alpha<=0 or alpha>1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]"%alpha)
        self.__alpha = alpha

    def __call__(self, value:float, timestamp:float=None, alpha:float=None) -> float:        
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha*value + (1.0-self.__alpha)*self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self) -> float:
        return self.__y
    
    def lastFilteredValue(self) -> float:
        return self.__s
    
    def reset(self) -> None:
        self.__y = None
        
        
#-------------------------------------------------------------------------

class MinilagFilter(object):
    def __init__(self,freq:float,mincutoff:float=1.0,gamma:float=0.0,dcutoff:float=1.0,eta:float=0.95,beta:float=0.25) -> None:
        """ Initializes the One Euro Filter
        
        :param freq: An estimate of the frequency in Hz of the signal (> 0), if timestamps are not available.
        :type freq: float
        :param mincutoff: Min cutoff frequency in Hz (> 0). Lower values allow to remove more jitter.
        :type mincutoff: float, optional
        :param  beta: Parameter to reduce latency (> 0).
        :type beta: float, optional
        :param  dcutoff: Used to filter the derivates. 1 Hz by default. Change this parameter if you know what you are doing.
        :type dcutoff: float, optional
        :raises ValueError: If one of the parameters is not >0
        """
        if freq<=0:
            raise ValueError("freq should be >0")
        if mincutoff<=0:
            raise ValueError("mincutoff should be >0")
        if dcutoff<=0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__gamma = float(gamma)
        self.__eta = float(eta)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = []
        self.__updatedFilteredLast = None
        self.__smoothedValue = None
        self.__deltaFilteredLast = 0
        self.__compensatedValue = None
        self.__deltaCurrentFiltered = None
    
    #-------------------------------------Alpha
    def __alpha(self, cutoff:float) -> float:
        """Computes the alpha value from a cutoff frequency.

        :param cutoff: cutoff frequency in Hz (> 0).
        :type cutoff: float
        :returns:  the alpha value to be used for the low pass filter
        :rtype: float
        """

        te    = 1.0 / self.__freq
        tau   = 1.0 / (2*math.pi*cutoff)
        return  1.0 / (1.0 + tau/te)
        
    #-------------------------------------Backtracking update
    
    
    def fit_curve(self,last_values,n=10,deg=3):
        '''
        fit un polynome de 3e degré sur les 10 dernières données
        '''
        if len(last_values)<10 : 
            raise ValueError('last_values must contain at least 10 values') 
        poly = np.polynomial.polynomial.Polynomial.fit([k*self.__freq for k in range(n)],last_values,deg)
        return(poly(n-2))
    
    def backtracked_value(self,poly_fitted,filtered,alpha = 0.8):
        '''
        Combine les données fittées et filtrées pour donner la nouvelle version de t-1
        '''
        new_value =  alpha * filtered + (1-alpha) * poly_fitted
        
        self.__updatedFilteredLast = new_value
    
    #-------------------------------------Prediction with compensation
    
    def smooth_value(self,Pt,dx):
        
        edx = self.__dx(dx)
        
        c = self.__mincutoff + self.__gamma*(edx)#todo trouver la derivée (utiliser le low pass filter ?)
        tau = 1/(2*c*np.pi)
        lambda_param = 1/(1+tau/self.__freq)
        pred = lambda_param * Pt + (1-lambda_param)*self.__updatedFilteredLast
        
        self.__smoothedValue = pred
        
    def compensate_value(self,Pt):
        
        deltaLast = self.__lasttime[-1] - self.__updatedFilteredLast

        self.__deltaFilteredLast = self.__eta * self.__deltaFilteredLast + (1-self.__eta)*deltaLast 
        
        self.__deltaCurrentFiltered = self.__beta*(Pt-self.__smoothedValue) + (1-self.__beta)* self.__deltaFilteredLast 
        self.__compensatedValue = self.__smoothedValue + self.__deltaCurrentFiltered

        self.__deltaFilteredLast = self.__deltaCurrentFiltered
        
    
    #-------------------------------------Application du filtre  
    def __call__(self,Pt):
        if len(self.__lasttime) < 10:
            self.__lasttime.append(Pt)
            dx=0
            self.__compensatedValue = self.__dx(Pt) #todo initialisation renvoie juste des valeurs qui descendent
            
        else : 
            self.__lasttime.pop(0)
            self.__lasttime.append(Pt)
            
            dx = Pt - self.__dx.lastFilteredValue()
            
            #-----------------------------Backtracking
            poly_value = self.fit_curve(self.__lasttime)
            self.backtracked_value(poly_value,self.__compensatedValue)
            
            #------------------------------Prediction
            self.smooth_value(Pt,dx)
            self.compensate_value(Pt)
            
            
            
            
      
        return(self.__compensatedValue)
            
            
def Euclid2Lie(R):
    A = scipy.linalg.logm(R)
    return(np.array((A[2,1],A[0,2],A[1,0]))) #vecteur créé pour l'algèbre de Lie

def Lie2Euclid(w):
    A = np.array((
        [0,-w[2],w[1]],
        [w[2],0,-w[0]],
        [-w[1],w[0],0]    
    ))
    return(scipy.linalg.logm(A))


class Rotation_Minilag(object):
    def __init__(self,config):
        self.R = None
        self.Rt1 = None
        self.wt1filt = None
        self.filtre_x = MinilagFilter(**config) 
        self.filtre_y = MinilagFilter(**config) 
        self.filtre_z = MinilagFilter(**config) 
        
        
        
    def __call__(self):
        wt = Euclid2Lie(self.R)
        Rt = Lie2Euclid(wt)
        wt = [0,0,0]
        
        #wt1filtree = Euclid2Lie(self.Rt1*Lie2Euclid(self.wt1filt))
        wt_filtrage=[0,0,0]
        wt_filtrage[0] = self.filtre_x(wt[0],t)
        wt_filtrage[0] = self.filtre_y(wt[1],t)
        wt_filtrage[0] = self.filtre_z(wt[2],t)
        # ---------------wtfilt = filtrage(wt,wt1filt) #todo
        wtfilt =Euclid2Lie(Rt*Lie2Euclid(wtfilt))
        
        
        
        return(Lie2Euclid(wtfilt)) 
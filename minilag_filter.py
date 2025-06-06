import numpy as np
import math



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
    def __init__(self,freq:float,mincutoff:float=1.0,beta:float=0.0,dcutoff:float=1.0) -> None:
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
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None
from scipy.fft import fft
import pandas as pd
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import numpy as np

def dft_matrix(N) -> np.ndarray:
    """
    Construct the NxN Discrete Fourier Transform (DFT) matrix.

    Parameters:
        N (int): Size of the DFT matrix (must be positive integer)

    Returns:
        numpy.ndarray: Complex-valued NxN DFT matrix
    """
    # Input validation
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    # Create index arrays
    n = np.arange(N)
    k = n.reshape((N, 1))  # Column vector

    # Compute the DFT matrix using broadcasting
    W = np.exp(-2j * np.pi * k * n / N)
    return W

def ifft_matrix(N) -> np.ndarray:
    """
    Construct the NxN Inverse Discrete Fourier Transform (IDFT) matrix.

    Parameters:
        N (int): Size of the IDFT matrix (must be positive integer)

    Returns:
        numpy.ndarray: Complex-valued NxN IDFT matrix
    """
    # Input validation
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    # Create index arrays
    n = np.arange(N)
    k = n.reshape((N, 1))  # Column vector

    # Compute the IDFT matrix using broadcasting
    W_inv = np.exp(2j * np.pi * k * n / N) / N
    return W_inv

class Inlet:
    def __init__(self, df: pd.DataFrame, radiuscol: str,thetacol:str,valuecol: str):
        self.df = df
        self.radiuscol = radiuscol
        self.thetacol = thetacol
        self.valuecol = valuecol
        self.nradius = len(self.df[self.radiuscol].unique())
        self.ntheta = len(self.df[self.thetacol].unique())

    def fftbyRadius(self) -> dict[int,np.ndarray]:
        """Compute the FFT of the values grouped by radius."""
        fftvalues = {}
        for radius in self.df[self.radiuscol].unique():
            subset = self.df[self.df[self.radiuscol] == radius]
            values = subset[self.valuecol].to_numpy()
            fftvalues[radius] = fft(values)
        return fftvalues
    
    def plotinlet(self):
        """Plots the inlet of the instance of the Inlet Class."""
        tempdf = self.df.copy()
        tempdf['x'] = tempdf[self.radiuscol] * np.cos(tempdf[self.thetacol])
        tempdf['y'] = tempdf[self.radiuscol] * np.sin(tempdf[self.thetacol])
        tris = Triangulation(tempdf['x'], tempdf['y'])
        fig,ax = plt.subplots(figsize = (6,6))
        ax.tricontourf(tris, tempdf[self.valuecol])
        ax.set_aspect('equal')
    
    def calcHarmonic(self,order:int) -> pd.DataFrame:
        """Calculates the harmonic of a specific order and returns a DataFrame with x, y, and value columns."""
        fftvalues = self.fftbyRadius()
        df = pd.DataFrame(index=fftvalues.keys())
        
        theta= self.df[self.thetacol].unique() # Gets the unique theta values from the original dataframe
        # Creates a new dataframe with x, y, and value columns by iterating through the radius and theta values and calculating the corresponding x, y,
        #   and value for each combination of radius and theta
        outdf = pd.DataFrame(columns = ['x','y','value'])
        dflist = []
        for radius in df.index:
            for angle in theta:
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                onefftvalue = self.fftbyRadius()[radius][order]
                value = np.real(onefftvalue * np.exp(1j * order * angle) / self.ntheta)
                dflist.append([x, y, radius,angle, value])
        outdf = pd.DataFrame(columns = ['x','y','r','theta','value'],data = dflist)

        return outdf
    
    def addHarmonics(self, maxorder: int) -> pd.DataFrame:
        """Calculate the sum of harmonics up to a specified order."""
        for i in range(maxorder + 1):
            if i == 0:
                totalorder = self.calcHarmonic(0)['value']
                xy = self.calcHarmonic(0).iloc[:,0:2]
            else:
                totalorder = totalorder + self.calcHarmonic(i)['value']

        return pd.concat([xy, totalorder], axis=1)
    
    def plotHarmonic(self, order: int, onemode: bool = True) -> None:
        """Plots either a specific harmonic or the sum of harmonics up to a specified order."""
        if onemode:
            outdf = self.calcHarmonic(order)
        else:
            outdf = self.addHarmonics(order)
        tris = Triangulation(outdf['x'], outdf['y'])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.tricontourf(tris, outdf['value'])
        ax.set_aspect('equal')
        return None
    
    def convergence(self,maxorder:int) -> float:
        """Calculates the convergence of the sum of harmonics up to a specified order."""
        totalorder = self.addHarmonics(maxorder)['value'].to_numpy()
        original = self.df[self.valuecol].to_numpy()
        return np.linalg.norm(original - totalorder) / np.linalg.norm(original)

    
class InletSet:
    def __init__(self, inlets: list[Inlet]):
        self.inletlist = inlets
    
    def addInlet(self, inlet: Inlet) -> None:
        self.inletlist.append(inlet)

    #def modeArray(self,order: int):
    #   modearray = np.zeros()
     #   for oneinlet in self.inletlist:



            
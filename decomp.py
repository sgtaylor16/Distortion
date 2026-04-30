from scipy.fft import fft, ifft
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
    def __init__(self, df: pd.DataFrame):
        """Initializes the Inlet class with a DataFrame containing the necessary columns."""
        required_columns = ['theta','radius']
        optional_columns = ['pt','Vradial','Vtheta','Vaxial',"MN"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain the required column: '{col}'")
        self.df = df
        self.nradius = len(self.df[].unique())
        self.ntheta = len(self.df['theta'].unique())

    def fftbyRadius(self,valuecol:str) -> dict[int,np.ndarray]:
        """Compute the FFT of the values grouped by radius."""
        if valuecol not in self.df.columns:
            raise ValueError(f"DataFrame must contain the specified value column: '{valuecol}'")
        fftvalues = {}
        for radius in self.df['radius'].unique():
            subset = self.df[self.df['radius'] == radius]
            values = subset[valuecol].to_numpy()
            fftvalues[radius] = fft(values)
        return fftvalues
    
    def plotinlet(self,valuecol:str):
        """Plots the inlet of the instance of the Inlet Class for the specified value column."""
        if valuecol not in self.df.columns:
            raise ValueError(f"DataFrame must contain the specified value column: '{valuecol}'")
        
        tempdf = self.df.copy()
        tempdf['x'] = tempdf['radius'] * np.cos(tempdf['theta'])
        tempdf['y'] = tempdf['radius'] * np.sin(tempdf['theta'])
        tris = Triangulation(tempdf['x'], tempdf['y'])
        fig,ax = plt.subplots(figsize = (6,6))
        ax.tricontourf(tris, tempdf[valuecol])
        ax.set_aspect('equal')

    def orderselect(fft,order,sumorders:bool=False) -> np.ndarray:
        """Selects the specified order from the FFT result."""
        #Check to make sure order is under n/2
        if order >= len(fft) // 2:
            raise ValueError("Order must be less than n/2.")
        redfft = np.zeros(len(fft),dtype=complex)
        if not sumorders:
            redfft[order] = fft[order]
            if order >0:
                redfft[-order] = fft[-order]
            return redfft
        if sumorders:
            redfft[:order+1] = fft[:order+1]
            redfft[-order:] = fft[-order:]
            return redfft
        else:
            raise ValueError("sumorders must be a boolean value.")

    def calcHarmonic(self,order:int,valuecol:str,sumorders:bool=False) -> pd.DataFrame:
        """Calculates the harmonic of a specific order and returns a DataFrame with x, y, and value columns."""
        if valuecol not in self.df.columns:
            raise ValueError(f"DataFrame must contain the specified value column: '{valuecol}'")
        fftvalues = self.fftbyRadius(valuecol)
        df = pd.DataFrame(index=fftvalues.keys())

        theta= self.df['theta'].unique() # Gets the unique theta values from the original dataframe
        # Creates a new dataframe with x, y, and value columns by iterating through the radius and theta values and calculating the corresponding x, y,
        #   and value for each combination of radius and theta
        outdf = pd.DataFrame(columns = ['x','y','value'])
        dflist = []
        for radius in df.index:
            theta = self.df[self.df['radius'] == radius]['theta'].to_numpy()
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)

            #redfit is the array of FFT values for the current radius, with all values set to zero except for the value at the specified order.
            redfft = Inlet.orderselect(fftvalues[radius], order, sumorders)

            value = ifft(redfft)
            #value = ifft_matrix(self.ntheta) @ redfft

            dflist.extend(zip(x, y, [radius]*len(theta), theta, np.real(value.flatten())))
        outdf = pd.DataFrame(columns = ['x','y','r','theta','value'],data = dflist)

        return outdf
    
    def plotHarmonic(self, order: int, valuecol: str, sumorders: bool = True) -> None:
        """Plots either a specific harmonic or the sum of harmonics up to a specified order."""
        if sumorders:
            outdf = self.calcHarmonic(order, valuecol, sumorders=True)
        else:
            outdf = self.calcHarmonic(order, valuecol, sumorders=False)
        tris = Triangulation(outdf['x'], outdf['y'])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.tricontourf(tris, outdf['value'])
        ax.set_aspect('equal')
        return None
    
    def convergence(self,maxorder:int, valuecol: str) -> float:
        """Calculates the convergence of the sum of harmonics up to a specified order."""
        totalorder = self.addHarmonics(maxorder, valuecol)['value'].to_numpy()
        original = self.df[valuecol].to_numpy()
        return np.linalg.norm(original - totalorder) / np.linalg.norm(original)

    
class InletSet:
    def __init__(self, inlets: list[Inlet]):
        self.inletlist = inlets
    
    def addInlet(self, inlet: Inlet) -> None:
        self.inletlist.append(inlet)

    

    #def modeArray(self,order: int):
    #   modearray = np.zeros()
     #   for oneinlet in self.inletlist:



            
import pandas as pd
from typing import List,Tuple
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.fft import fft, ifft
from scipy.signal import resample

def dfcheck(df:pd.DataFrame) -> bool:
    """
    Check if the dataframe has the required columns for SAE16 calculations.
    The required columns are 'radius', 'theta', 'pt', 'ps', 'v_swirl', and 'v_axial'.
    'v_swirl' should be interpreted as the swirl angle.
    """
    required_columns = ['radius', 'theta', 'pt','ps']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
    warn_if_theta_looks_radians(df)
    return True

def warn_if_theta_looks_radians(df: pd.DataFrame, tol: float = 1e-6) -> None:
    """Warn when theta appears to be in radians instead of degrees."""
    if 'theta' not in df.columns:
        return

    theta_values = df['theta'].dropna()
    if theta_values.empty:
        return

    theta_min = float(theta_values.min())
    theta_max = float(theta_values.max())
    if theta_min >= -tol and theta_max <= (2 * np.pi + tol):
        warnings.warn(
            "theta appears to be in radians (range 0 to about 2*pi). Expected degrees in [0, 360).",
            UserWarning,
            stacklevel=2,
        )

def findrings(data:pd.DataFrame,tolerance=0.05) -> List[pd.DataFrame]:
    """
    Split inlet measurements into concentric rings by grouping nearby radius values.

    Rows are first ordered by radius. A new ring is started whenever the current row's
    radius differs from the previous row's radius by more than tolerance.
    """
    if not dfcheck(data):
        raise ValueError("DataFrame does not have the required columns.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    if data.empty:
        return []

    sorted_data = data.sort_values(by='radius').reset_index(drop=True)
    rings = []

    start_idx = 0
    prev_r = float(sorted_data.loc[0, 'radius'])
    for i in range(1, len(sorted_data)):
        current_r = float(sorted_data.loc[i, 'radius'])
        if abs(current_r - prev_r) > tolerance:
            ring_df = sorted_data.iloc[start_idx:i].copy().reset_index(drop=True).sort_values(by='theta').reset_index(drop=True)
            rings.append(ring_df)
            start_idx = i
        prev_r = current_r

    # Add the final ring.
    rings.append(sorted_data.iloc[start_idx:].copy().reset_index(drop=True).sort_values(by='theta').reset_index(drop=True))
    return rings

def findzero_crossing(data:pd.DataFrame,avg:float,value:str='pt') -> List[float]:
    """
    Find the points where the data crosses zero. Interpolates between points to find the exact
    crossing point. Assumes that the data is ordered by 'theta' and that 'pt' is the value being analyzed for crossings with respect to pavg.
    """
    zerocrossings = []
    for i in range(len(data) - 1):
        x0 = data.loc[i, 'theta']
        x1 = data.loc[i+1, 'theta']
        y0 = data.loc[i, value] - avg
        y1 = data.loc[i+1, value] - avg
        if y0 * y1 > 0:
        # No zero crossing
            continue
        else:
        # Linear interpolation to find the exact zero crossing
            zero_crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
            zerocrossings.append(zero_crossing)

    #Check the edge case for wrap-around crossing between the last and first points
    x0 = data.loc[len(data) - 1, 'theta']
    x1 = data.loc[0, 'theta'] + 360 # Add wrap-around
    y0 = data.loc[len(data) - 1, value] - avg
    y1 = data.loc[0, value] - avg
    if y0 * y1 <= 0:
        zero_crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
        zero_crossing = zero_crossing % (360) # Wrap back to [0, 360]
        zerocrossings.append(zero_crossing)
    return zerocrossings

def find_segments(data:pd.DataFrame,avg,value:str='pt') -> List[pd.DataFrame]:
    """
    Find the segments of the data where value is all above or all below the avg value"""
    segments = []
    zero_crossings = findzero_crossing(data,avg,value)
    if len(zero_crossings) < 2:
        return [data] #Return the whole dataframe as a single segment if there are no zero crossings
    # Evaluate all adjacent crossing pairs plus the wrap-around pair (last -> first).
    crossing_pairs = list(zip(zero_crossings, zero_crossings[1:]))
    crossing_pairs.append((zero_crossings[-1], zero_crossings[0]))

    for left, right in crossing_pairs:
        if left <= right:
            segment = data[(data['theta'] >= left) & (data['theta'] <= right)]
            #Add the actual zero crossing points to the segment
            segment = pd.concat([segment, pd.DataFrame({'theta': [left, right], value: [avg, avg]})], ignore_index=True)
            segment = segment.sort_values(by='theta').reset_index(drop=True)
        else:
            # Circular interval that spans the end and beginning of theta.
            leftpart = data[data['theta'] >= left].copy()
            rightpart = data[data['theta'] <= right].copy()
            # Add 360 to the left part to handle the wrap-around correctly when concatenating.
            rightpart['theta'] = rightpart['theta'].apply(lambda x: x +360)
            segment = pd.concat([leftpart, rightpart], ignore_index=True)
            #Add the actual zero crossing points to the segment
            segment = pd.concat([segment, pd.DataFrame({'theta': [left, right+360], value: [avg, avg]})], ignore_index=True)
            segment = segment.sort_values(by='theta').reset_index(drop=True)
        if segment.empty:
            continue
        segments.append(segment)

    return segments

def findnegative_segments(data:pd.DataFrame,avg:float,value:str='pt') -> List[pd.DataFrame]:
    """
    Find the segments of the data where p is below the average value.
    """

    allsegments = find_segments(data,avg,value)
 
    return [segment for segment in allsegments if segment[value].mean() < avg] #Returns only the segments where the mean value is below the average.

def area_bar(segment:pd.DataFrame,avg:float,value:str='pt') -> float:
    """
    Calculate the area of the segment below pavg using the trapezoidal rule.
    Assumes that the segment is ordered by 'theta'.
    """
    if segment.empty:
        return 0.0
    # Subtract pavg from p to get the area below pavg.
    y = segment[value] - avg
    x = segment['theta']
    area = np.trapz(y, x) / (segment['theta'].max() - segment['theta'].min())  # Normalize by the theta range to get an average area.
    return max(0.0, area + avg)  # Area should be positive, so take negative of the result.

def proximity_test(segment1, segment2, critangle:float) -> bool:
    if segment2.start - segment1.end < critangle:
        return True
    else:
        return False
    
def orderselect(fft,order,sumorders:bool=False) -> np.ndarray:
    """
    Selects the specified order from the FFT result.
    if sumorders is True, selects all orders up to and including the specified order.
    """
    #Check to make sure order is under n/2
    if order >= len(fft) // 2:
        raise ValueError("Order must be less than n/2.")
    redfft = np.zeros(len(fft),dtype=complex)
    if not sumorders:
        redfft[order] = fft[order]
        redfft[-order] = fft[-order]
        return redfft
    if sumorders:
        redfft[:order+1] = fft[:order+1]
        redfft[-order:] = fft[-order:]
        return redfft
    else:
        raise ValueError("sumorders must be a boolean value.")
    
def interpfit_fft(x, n, dim=None):
    """
    Interpolate periodic data using the FFT method.

    Matches MATLAB's interpft behavior by operating on the first dimension
    whose size is not 1 when dim is not provided.
    """
    values = np.asarray(x)
    if values.ndim == 0:
        values = values.reshape(1)

    if not np.isscalar(n):
        raise ValueError("n must be a scalar positive integer.")
    n_value = float(n)
    n = int(n_value)
    if n < 1 or n != n_value:
        raise ValueError("n must be a positive integer.")

    if dim is None:
        axis = next((i for i, size in enumerate(values.shape) if size != 1), 0)
    else:
        axis = int(dim)
        if axis < 0:
            axis += values.ndim
        if axis < 0 or axis >= values.ndim:
            raise ValueError("dim is out of range for the input array.")

    return resample(values, n, axis=axis)

def centers_of_equal_area(outer_radius:float, inner_radius:float, num_rings:int) -> List[float]:
    """
    Calculate the center radii of num_rings concentric rings that have equal area between inner_radius and outer_radius.
    """
    if num_rings < 1:
        raise ValueError("num_rings must be a positive integer.")
    if inner_radius < 0 or outer_radius <= inner_radius:
        raise ValueError("Radii must satisfy 0 <= inner_radius < outer_radius.")

    total_area = np.pi * (outer_radius**2 - inner_radius**2)
    area_per_ring = total_area / num_rings
    center_radii = []
    
    for i in range(num_rings):
        ring_inner_area = area_per_ring * i
        ring_outer_area = area_per_ring * (i + 1)
        
        ring_inner_radius = np.sqrt(inner_radius**2 + ring_inner_area / np.pi)
        ring_outer_radius = np.sqrt(inner_radius**2 + ring_outer_area / np.pi)
        
        center_radius = (ring_inner_radius + ring_outer_radius) / 2
        center_radii.append(center_radius)
    
    return center_radii

def stack_df(df,value) -> np.ndarray:
    """
    Stack the DataFrame into a 2D array with shape (n, 1)
    where n is the number of rows in the DataFrame.
    The DataFrame must contain 'theta', 'radius', and the specified value column."""

    #Make sure theta and radius are in df columns
    if 'theta' not in df.columns or 'radius' not in df.columns or value not in df.columns:
        raise ValueError(f"DataFrame must contain 'theta', 'radius', and '{value}' columns.")
    newdf = df[['radius','theta',value]].copy().sort_values(by=['radius','theta']).reset_index(drop=True)
    return newdf[value].to_numpy().reshape(-1,1)

def stack_stacks(df,valuelist:List[str]) -> np.ndarray:
    for value in valuelist:
        if value not in df.columns:
            raise ValueError(f"DataFrame must contain '{value}' column.")
    return np.vstack([stack_df(df, value) for value in valuelist])

def plotutility(df: pd.DataFrame, value: str = 'pt', includepts: bool = False, colorbar: bool = False, cmap: str = 'viridis', ax: plt.Axes = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    tris = Triangulation(df['radius'] * -np.sin(np.deg2rad(df['theta'])), df['radius'] * np.cos(np.deg2rad(df['theta'])))
    
    contour = ax.tricontourf(tris, df[value], cmap=cmap)
    if includepts:
        ax.plot(df['radius'] * -np.sin(np.deg2rad(df['theta'])),
                df['radius'] * np.cos(np.deg2rad(df['theta'])),
                'ko', markersize=2)
    if colorbar:
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(value)
    ax.set_aspect('equal')
    ax.invert_xaxis()
    return ax

def rescale_columns(df:pd.DataFrame) -> Tuple[List[str],List[str]]:
    """Screens dataframe df for columns that can be rescaled in the theta dimension. Retuns a list of eligable column names
    Returns a list of columns to scale and a list of columns to leave unchanged"""
    ineligable_columns = []
    for column in df.columns:
        if (df[column].dtype != float):
            ineligable_columns.append(column)
        if (column == 'radius') or (column == 'theta'):
            ineligable_columns.append(column)
    eligable_columns = [x for x in df.columns if x not in ineligable_columns]
    ineligable_columns = [x for x in ineligable_columns if x not in ['radius','theta']]

    return eligable_columns,ineligable_columns
    
class Segment:
    def __init__(self,segmentdata:pd.DataFrame,avg:float,value:str='pt'):
        self.segmentdata = segmentdata
        self.avg = avg
        self.value = value

        if (self.segmentdata.iloc[0]['theta'] < .01) and (self.segmentdata.iloc[-1]['theta'] > 359.99):
            self.extent = (360 - self.segmentdata.iloc[-1]['theta']) + self.segmentdata.iloc[0]['theta']
            self.start = self.segmentdata.iloc[-1]['theta']
            self.end = self.segmentdata.iloc[0]['theta']
        else:
            self.extent = self.segmentdata['theta'].max() - self.segmentdata['theta'].min()
            self.start = self.segmentdata['theta'].min()
            self.end = self.segmentdata['theta'].max()

    def plotsegment(self):
        fig, ax = plt.subplots()
        ax.plot(self.segmentdata['theta'], self.segmentdata[self.value], label=self.value)
        ax.axhline(self.avg, color='red', linestyle='--', label=f'Average {self.value}')
        ax.set_xlabel('Theta (degrees)')
        ax.set_ylabel(self.value)

class PressureSegment(Segment):
    def __init__(self,segmentdata:pd.DataFrame,pavg:float):
        super().__init__(segmentdata,pavg,'pt')
        self.area_bar = area_bar(segmentdata,pavg,'pt')

class SwirlSegment(Segment):
    def __init__(self,segmentdata:pd.DataFrame):
        super().__init__(segmentdata,0,'v_swirl')

    def avgSwirl(self) -> float:
        """Calculate the average swirl angle for the segment."""
        integrate = np.trapz(self.segmentdata['v_swirl'], self.segmentdata['theta'])
        return integrate / self.extent if self.extent > 0 else 0

class Ring:
    def __init__(self,df:pd.DataFrame):
        """The dataframe should have columns 'radius', 'theta', and 'pt',"""

        dfcheck(df)

        self.df = df
        self.radius = df['radius'].mean()
        self.r = self.radius

        # Find Pressure Segments
        self.pavg = self.df['pt'].mean()
        zerosegments = findnegative_segments(df,self.pavg,'pt')
        # Sort segments by their starting theta value for consistent ordering.
        self.segments = sorted([PressureSegment(seg, self.pavg) for seg in zerosegments], key=lambda seg: seg.start)

        #Find Swirl Segments
        #Make sure swirl is not all zero before calculating swirl segments
        if not np.all(self.df['v_swirl'] == 0):
            swirlsegments = find_segments(df,0,'v_swirl')
            self.swirlsegments = sorted([SwirlSegment(seg) for seg in swirlsegments], key=lambda seg: seg.start)

    def PAV(self) -> float:
        return self.df['pt'].mean()
    
    def extents(self,critangle:float = 25.0) -> List[float]:
        pavg = self.PAV()
        if len(self.segments) == 1:
            return [self.segments[0].extent]
        elif len(self.segments) > 1:
             pass

    def adjacency(self,critangle:float = 25.0) -> dict:
        """
        Create an adjacency list that groups segments together if they are within critangle degrees of each other.
        Returns a dictionary where the keys are the adjacency group identifiers and the values are lists of segment indices that belong to each group.
        """
        adjacency = [-1 for _ in self.segments] #Init adjacency list with -1 to indicate unassigned segments
        for i in range(len(self.segments)):
            if i ==0:
                adjacency[i] = 0
            else:
                if proximity_test(self.segments[i-1], self.segments[i], critangle):
                    adjacency[i] = adjacency[i-1]
                else:
                    adjacency[i] = adjacency[i-1] + 1
        
        return {key: [i for i, val in enumerate(adjacency) if val == key] for key in set(adjacency)}
    
    def CDI(self,critangle:float = 25.0) -> float:
        #Find the regions of the ring where p is below the average value
        pavg = self.PAV()
        #Find the zero crossings of p - pavg
        zero_crossings = findzero_crossing(self.df,pavg)
        if len(zero_crossings) < 2:
            raise ValueError("Not enough zero crossings to define segments.")
        zero_segments = findnegative_segments(self.df,pavg)

        if len(self.segments) == 0:
            return 0.0
    
        elif len(self.segments) == 1:
            pavlow = self.segments[0].area_bar
        
            return (pavg - pavlow) / pavg
        
        else: #More than one segment
            adjacency = self.adjacency(critangle)
            grouplist = []
            for key, segment_indices in adjacency.items():
                weightedCDI, extentsum = self.grouped_CDI(segment_indices)
                grouplist.append((weightedCDI, extentsum))

            df = pd.DataFrame(grouplist, columns=['Weighted CDI', 'Extent'])
            df['CDI'] = df['Weighted CDI'] / df['Extent']
            max_weighted_idx = df['Weighted CDI'].idxmax()
            return float(df.loc[max_weighted_idx, 'CDI'])

    def grouped_CDI(self,segmentlist: List[int]) -> float:
        weightedCDI = 0.0
        extentsum = 0.0
        for segment_index in segmentlist:
            cdi = (self.pavg - self.segments[segment_index].area_bar) / self.pavg
            weightedCDI += cdi* self.segments[segment_index].extent
            extentsum += self.segments[segment_index].extent
        return weightedCDI, extentsum
    
    def plotring(self,value,ax:plt.Axes=None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.df['theta'], self.df[value], label=value)
        ax.axhline(self.df[value].mean(), color='red', linestyle='--', label=f'Average {value}')
        ax.set_xlabel('Theta (degrees)')
        ax.set_ylabel(value)
        return ax
    
    def fft(self,value:str='pt') -> np.ndarray:
        p = self.df[value].to_numpy()
        return fft(p,normalize='forward') #To conform to SAE16 calculation

    def ringfft(self,order:int,value:str='pt',sumorders:bool=False):
        fft_values = self.fft(value)
        return orderselect(fft_values,order,sumorders)
    
    def calcHarmonic(self,order:int,value:str='pt',sumorders:bool=False) -> pd.DataFrame:
        """Calculates the harmonic of a specific order for the ring and returns a DataFrame with x, y, and value columns."""
        selected_fft = self.ringfft(order=order,value=value,sumorders=sumorders)
        harmonic_value = ifft(selected_fft,normalize='forward') #To conform to SAE16 calculation
        outdf = pd.DataFrame({
            'x': self.df['radius'] * np.cos(np.deg2rad(self.df['theta'])),
            'y': self.df['radius'] * np.sin(np.deg2rad(self.df['theta'])),
            'radius': self.df['radius'],
            'theta': self.df['theta'],
            value: np.real(harmonic_value) #This returns the real part of the harmonic value which should be real. This removes any negligible imaginary component.
        })
        return outdf
    
    def resample_df(self, n:int) -> pd.DataFrame:
        """Uses Scipy.signal's resample function to resample the ring data to n points."""
        resampled_theta = np.linspace(0, 360, n, endpoint=False)
        resampled_radius = np.full(n, self.df['radius'].iloc[0]) #Assumes radius is constant within the ring
        resampled_df = pd.DataFrame({
            'radius': resampled_radius,
            'theta': resampled_theta
        })
        # Group the columns into ones that should be interpolated and those that shouldn't.
        fit_columns,constant_columns = rescale_columns(self.df)
        #Interpolate the fit columns
        for col in fit_columns:
            resampled_df[col] = interpfit_fft(self.df[col], n)
        #Leave unchange the constant columns
        for col in constant_columns:
            resampled_df[col] = self.df[col][0]
        return resampled_df
    
    def swirlintensity(self) -> float:
        """Calculates the SAE16 swirl intensity metric for the ring."""
        numerator = 0.0
        for segment in self.swirlsegments:
            numerator += abs(segment.avgSwirl()) * segment.extent
        return numerator / 360.0
    
    def swirlDirectivity(self) -> float:
        """Calculates the SAE16 swirl directivity metric for the ring."""
        numerator = 0.0
        denominator = 0.0
        for segment in self.swirlsegments:
            numerator += segment.avgSwirl() * segment.extent
            denominator += abs(segment.avgSwirl()) * segment.extent
        if denominator == 0:
            raise ValueError("Denominator is zero, cannot calculate swirl directivity.")
        return numerator / denominator

class Face:
    """Class that represents the rings that make up a face and calculates the SAE16 Intensity metric for the face.
    The df expects the following columns: 'radius', 'theta', 'pt', 'ps', 'v_swirl', and 'v_axial'.
    theta should be in degrees and should be in the range [0, 360).
    """

    def __init__(self,datadf:pd.DataFrame,tolerance=0.05,critangle:float = 25.0):
        datadf = datadf.copy()
        # Add v_swirl if column not already in dataframe, just set it to 0.
        if 'v_swirl' not in datadf.columns:
            datadf['v_swirl'] = 0.0
        if 'v_axial' not in datadf.columns:
            datadf['v_axial'] = 0.0
        # Calculate incidence
        dfcheck(datadf)
        datadf['radius'] = datadf['radius'].round(4)
        #Sort the columns in the matrix in a specifc order to facilitate reshaping column vectors.
        datadf = datadf.sort_values(by=['radius','theta']).reset_index(drop=True)
        datadf['incidence'] = np.arctan2(datadf['v_swirl'],datadf['v_axial']) * 180 / np.pi

        integer_columns = datadf.select_dtypes(include=["integer"]).columns
        datadf[integer_columns] = datadf[integer_columns].astype(float)

        ringsegments = findrings(datadf,tolerance)
        self.rings = [Ring(ringdata) for ringdata in ringsegments]
        self.df = datadf
        self.critangle = critangle
        self.Q = self.df['pt'].mean() - self.df['ps'].mean()

    def PFAV(self) -> float:
        return np.mean([ring.PAV() for ring in self.rings])

    def RDI(self,i) -> float:
        return (self.PFAV() - self.rings[i].PAV()) / self.PFAV()
    
    def RDImax(self) -> float:
        rings_RDI = [self.RDI(i) for i in range(len(self.rings))]
        return max(rings_RDI)
    
    def CDI(self,i) -> float:
        return self.rings[i].CDI(self.critangle)
    
    def CDImax(self) -> float:
        rings_CDI = [self.CDI(i) for i in range(len(self.rings))]
        return max(rings_CDI)

    def swirlintensity(self,i) -> float:
        return self.rings[i].swirlintensity()

    def swirlintesitymax(self) -> float:
        rings_swirlintensity = [self.swirlintensity(i) for i in range(len(self.rings))]
        return max(rings_swirlintensity)
    
    def HEI(self,ring:int,value='pt',normalize:bool=True) -> List[float]:
        """Calculates the Harmonic Energy Index for a specific ring."""
        fft_values = self.rings[ring].fft(value)
        q = self.df['pt'].mean() - self.df['ps'].mean()
        if normalize:
            HEI_values = [np.abs(fft_values[n]) / q for n in range(1, len(fft_values)//2)]
        else:
            HEI_values = [np.abs(fft_values[n]) for n in range(1, len(fft_values)//2)]
        return HEI_values

    def plotFace(self, value='pt', includepts:bool=False, colorbar:bool=False, cmap:str='viridis', ax=None) -> plt.Axes:
        return plotutility(self.df, value=value, includepts=includepts, colorbar=colorbar, cmap=cmap, ax=ax)

    def plot_velocity(self, value='pt', colorbar:bool=False, cmap:str='viridis', ax=None, scale=None, quiver_color:str='k') -> plt.Axes:
        """Plots pressure contours with the in-plane velocity (v_radial, v_swirl) overlaid as quiver arrows."""
        ax = self.plotFace(value=value, includepts=False, colorbar=colorbar, cmap=cmap, ax=ax)

        theta_rad = np.deg2rad(self.df['theta'])
        x = self.df['radius'] * -np.sin(theta_rad)
        y = self.df['radius'] * np.cos(theta_rad)

        v_radial = self.df['v_radial'] if 'v_radial' in self.df.columns else 0.0
        v_swirl = self.df['v_swirl']
        vx = v_radial * -np.sin(theta_rad) + v_swirl * -np.cos(theta_rad)
        vy = v_radial * np.cos(theta_rad) + v_swirl * -np.sin(theta_rad)

        ax.quiver(x, y, vx, vy, color=quiver_color, scale=scale)
        return ax

    def calcHarmonic(self,order:int,value = 'pt',sumorders:bool=False) -> pd.DataFrame:
        """Calculates the harmonic of a specific order for each ring and returns a DataFrame with radius, theta and value columns.
        The dataframe is ordered by radius's first, at each radius the theta's are walked through.
        """
        for i,ring in enumerate(self.rings):
            ring_harmonic = ring.calcHarmonic(order=order,value=value,sumorders= sumorders)
            if i == 0:
                harmonics_by_ring = ring_harmonic
            else:
                harmonics_by_ring = pd.concat([harmonics_by_ring, ring_harmonic], ignore_index=True)
        return harmonics_by_ring
    
    def plotHarmonic(self, order:int,value='pt', ax=None, sumorders:bool = False) -> plt.axes:
        """Plot either a specific harmonic or cumulative harmonics up to order."""
        outdf = self.calcHarmonic(order,value,sumorders=sumorders)
        tris = Triangulation(-outdf['y'], outdf['x'])  # Rotate the points so that 0 degrees is at the top of the plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.tricontourf(tris, outdf[value])
        ax.set_aspect('equal')
        ax.invert_xaxis()
        return ax
    
    def resample_theta(self, n:int) -> 'Face':
        """Resample each ring to n points and return a new Face object with the resampled data."""
        resampled_rings = []
        for ring in self.rings:
            resampled_df = ring.resample_df(n)
            resampled_rings.append(resampled_df)
        resampled_data = pd.concat(resampled_rings, ignore_index=True)
        return Face(resampled_data, critangle=self.critangle)
        
    def resample_r(self,r_n: int) -> 'Face':
        """Resample the rings to r_n rings with equal area and return a new Face object with the resampled data."""
        outer_radius = self.df['radius'].max()
        inner_radius = self.df['radius'].min()
        center_radii = centers_of_equal_area(outer_radius, inner_radius, r_n)

        sorted_rings = sorted(self.rings, key=lambda ring: ring.df['radius'].iloc[0])
        ring_radii = np.array([ring.df['radius'].iloc[0] for ring in sorted_rings])

        resampled_rings = []
        for center_r in center_radii:
            idx_above = np.searchsorted(ring_radii, center_r, side='right')
            idx_below = idx_above - 1

            if idx_below < 0:
                resampled_rings.append(sorted_rings[0].df.assign(radius=center_r))
            elif idx_above >= len(sorted_rings):
                resampled_rings.append(sorted_rings[-1].df.assign(radius=center_r))
            else:
                ring_below = sorted_rings[idx_below]
                ring_above = sorted_rings[idx_above]
                r_below = ring_radii[idx_below]
                r_above = ring_radii[idx_above]
                t = (center_r - r_below) / (r_above - r_below)

                thetas = ring_below.df['theta'].values

                fit_columns,constant_columns = rescale_columns(self.df)

                dfdict={}
                for onecolumn in fit_columns:
                    values_below = ring_below.df[onecolumn].values
                    values_above = ring_above.df[onecolumn].values
                    values_interp = (1-t) * values_below + t * values_above
                    dfdict[onecolumn] = values_interp

                for onecolumn in constant_columns:
                    dfdict[onecolumn] = len(thetas) * [self.df[onecolumn][0]]

                dfdict['theta'] = thetas
                dfdict['radius'] = len(thetas) * [center_r]

                resampled_rings.append(pd.DataFrame(dfdict))

        resampled_data = pd.concat(resampled_rings, ignore_index=True)
        return Face(resampled_data, critangle=self.critangle)
    
    def resample(self, r_n: int, theta_n: int) -> 'Face':
        """Resample the face to r_n rings and theta_n points per ring, returning a new Face object with the resampled data."""
        return self.resample_r(r_n).resample_theta(theta_n)

    def stacks(self,valuelist:list[str]) -> np.ndarray:
        """Stack the specified values from the face's dataframe into a 2D array."""
        return stack_stacks(self.df,valuelist)

    def calc_stackheight(self) -> int:
        """Calculate the stack height of the face, defined as the number of unique measurement locations on a face"""
        return self.df.shape[0]

    def plot_ring_harmonics(self,maxorder:int,value='pt'):
        """
        Plots the magnitude of the fourier transform for each order, by radius
        """
        radiuslist = [x.radius for x in self.rings]
        fftlist = [abs(x.ringfft(maxorder,value,True)) for x in self.rings]

        fig,ax = plt.subplots()

        for i in [i for i in range(maxorder) if i !=0]:
            ax.plot([x[i] for x in fftlist],radiuslist,label=f"order {i}")
            ax.legend()


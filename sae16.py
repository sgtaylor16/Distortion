import pandas as pd
from typing import List
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.fft import fft, ifft
from scipy.signal import resample


def dfcheck(df:pd.DataFrame) -> bool:
    """
    Check if the dataframe has the required columns for SAE16 calculations.
    The required columns are 'r', 'theta', and 'pt' and swirl. Swirl should be interpreted
    as the swirl angle.
    """
    required_columns = ['r', 'theta', 'pt','ps','swirl']
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
    Split inlet measurements into concentric rings by grouping nearby r values.

    Rows are first ordered by r. A new ring is started whenever the current row's
    r differs from the previous row's r by more than tolerance.
    """
    if not dfcheck(data):
        raise ValueError("DataFrame does not have the required columns.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    if data.empty:
        return []

    sorted_data = data.sort_values(by='r').reset_index(drop=True)
    rings = []

    start_idx = 0
    prev_r = float(sorted_data.loc[0, 'r'])
    for i in range(1, len(sorted_data)):
        current_r = float(sorted_data.loc[i, 'r'])
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

class Segment:
    def __init__(self,segmentdata:pd.DataFrame,avg:float,value:str='pt'):
        self.segmentdata = segmentdata
        self.avg = avg
        self.value = value
        self.area_bar = area_bar(segmentdata,avg,value)
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

class SwirlSegment(Segment):
    def __init__(self,segmentdata:pd.DataFrame):
        super().__init__(segmentdata,0,'swirl')

    def avgSwirl(self) -> float:
        """Calculate the average swirl angle for the segment."""
        integrate = np.trapz(self.segmentdata['swirl'], self.segmentdata['theta'])
        return integrate / self.extent if self.extent > 0 else 0

class Ring:
    def __init__(self,ringdata:pd.DataFrame):
        """The ringdata dataframe should have columns 'r', 'theta', and 'pt',"""

        dfcheck(ringdata)

        self.ringdata = ringdata
        self.r = ringdata['r'].mean()

        # Find Pressure Segments
        self.pavg = self.ringdata['pt'].mean()
        zerosegments = findnegative_segments(ringdata,self.pavg,'pt')
        # Sort segments by their starting theta value for consistent ordering.
        self.segments = sorted([PressureSegment(seg, self.pavg) for seg in zerosegments], key=lambda seg: seg.start)

        #Find Swirl Segments
        #Make sure swirl is not all zero before calculating swirl segments
        if not np.all(self.ringdata['swirl'] == 0):
            swirlsegments = find_segments(ringdata,0,'swirl')
            self.swirlsegments = sorted([SwirlSegment(seg) for seg in swirlsegments], key=lambda seg: seg.start)

    def PAV(self) -> float:
        return self.ringdata['pt'].mean()
    
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
        zero_crossings = findzero_crossing(self.ringdata,pavg)
        if len(zero_crossings) < 2:
            raise ValueError("Not enough zero crossings to define segments.")
        zero_segments = findnegative_segments(self.ringdata,pavg)

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
    
    def plotring(self,value) -> plt.axes:
        fig, ax = plt.subplots()
        ax.plot(self.ringdata['theta'], self.ringdata[value], label=value)
        ax.axhline(self.ringdata[value].mean(), color='red', linestyle='--', label=f'Average {value}')
        ax.set_xlabel('Theta (degrees)')
        ax.set_ylabel(value)
        return ax
    
    def fft(self):
        p = self.ringdata['pt'].to_numpy()
        return fft(p)
    
    def calcHarmonic(self,order:int,sumorders:bool=False) -> pd.DataFrame:
        """Calculates the harmonic of a specific order for the ring and returns a DataFrame with x, y, and value columns."""
        fft_values = self.fft()
        selected_fft = orderselect(fft_values, order, sumorders)
        harmonic_value = ifft(selected_fft).real #Do I nead the real?
        outdf = pd.DataFrame({
            'x': self.ringdata['r'] * np.cos(self.ringdata['theta']),
            'y': self.ringdata['r'] * np.sin(self.ringdata['theta']),
            'r': self.ringdata['r'],
            'theta': self.ringdata['theta'],
            'value': harmonic_value
        })
        return outdf
    
    def resample_df(self, n:int) -> pd.DataFrame:
        """Uses Scipy.signal's resample function to resample the ring data to n points."""
        resampled_pt = interpfit_fft(self.ringdata['pt'], n)
        resampled_ps = interpfit_fft(self.ringdata['ps'], n)
        resampled_theta = np.linspace(0, 360, n, endpoint=False)
        resampled_theta = np.array([(x+180)%360 for x in resampled_theta]) #Shift theta by 180 degrees
        resampled_r = np.full(n, self.ringdata['r'].iloc[0]) #Assumes r is constant within the ring
        resampled_df = pd.DataFrame({
            'r': resampled_r,
            'theta': resampled_theta,
            'pt': resampled_pt,
            'ps': resampled_ps
        })
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
    The df expects the following columns: 'r', 'theta', and 'pt'.
    theta should be in degrees and should be in the range [0, 360).
    """

    def __init__(self,datadf:pd.DataFrame,tolerance=0.05,critangle:float = 25.0):
        # Add Swirl if column not already in dataframe, just set it to 0.
        if 'swirl' not in datadf.columns:
            datadf['swirl'] = 0.0
        dfcheck(datadf)
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
    
    def HEI(self,ring:int) -> List[float]:
        """Calculates the Harmonic Energy Index for a specific ring."""
        fft_values = self.rings[ring].fft()
        q = self.df['pt'].mean() - self.df['ps'].mean()
        HEI_values = [(len(fft_values)//2) * np.abs(fft_values[n]) / q for n in range(1, len(fft_values)//2)]
        return HEI_values

    def plotFace(self,includepts:bool=False,value='pt', colorbar:bool=False) -> plt.axes:
        fig, ax = plt.subplots(figsize=(6, 6))
        tris = Triangulation(self.df['r'] * np.cos(self.df['theta']*np.pi/180), self.df['r'] * np.sin(self.df['theta']*np.pi/180))
        contour = ax.tricontourf(tris, self.df[value])
        if includepts:
            ax.plot(self.df['r'] * np.cos(self.df['theta']*np.pi/180),
                    self.df['r'] * np.sin(self.df['theta']*np.pi/180),
                    'ko', markersize=2)
        if colorbar:
            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label(value)
        ax.set_aspect('equal')
        return ax

    def calcHarmonic(self,order:int,sumorders:bool=False) -> pd.DataFrame:
        """Calculates the harmonic of a specific order for each ring and returns a DataFrame with r and value columns."""
        for i,ring in enumerate(self.rings):
            ring_harmonic = ring.calcHarmonic(order, sumorders)
            if i == 0:
                harmonics_by_ring = ring_harmonic
            else:
                harmonics_by_ring = pd.concat([harmonics_by_ring, ring_harmonic], ignore_index=True)
        return harmonics_by_ring
    
    def plotHarmonic(self, order: int, sumorders: bool = True) -> plt.axes:
        """Plot either a specific harmonic or cumulative harmonics up to order."""
        outdf = self.calcHarmonic(order, sumorders=sumorders)
        tris = Triangulation(outdf['x'], outdf['y'])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.tricontourf(tris, outdf['value'])
        ax.set_aspect('equal')
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
        outer_radius = self.df['r'].max()
        inner_radius = self.df['r'].min()
        center_radii = centers_of_equal_area(outer_radius, inner_radius, r_n)

        sorted_rings = sorted(self.rings, key=lambda ring: ring.ringdata['r'].iloc[0])
        ring_radii = np.array([ring.ringdata['r'].iloc[0] for ring in sorted_rings])

        resampled_rings = []
        for center_r in center_radii:
            idx_above = np.searchsorted(ring_radii, center_r, side='right')
            idx_below = idx_above - 1

            if idx_below < 0:
                resampled_rings.append(sorted_rings[0].ringdata.assign(r=center_r))
            elif idx_above >= len(sorted_rings):
                resampled_rings.append(sorted_rings[-1].ringdata.assign(r=center_r))
            else:
                ring_below = sorted_rings[idx_below]
                ring_above = sorted_rings[idx_above]
                r_below = ring_radii[idx_below]
                r_above = ring_radii[idx_above]
                t = (center_r - r_below) / (r_above - r_below)

                thetas = ring_below.ringdata['theta'].values
                pt_below = ring_below.ringdata['pt'].values
                pt_above = ring_above.ringdata['pt'].values
                pt_interp = (1 - t) * pt_below + t * pt_above

                ps_below = ring_below.ringdata['ps'].values
                ps_above = ring_above.ringdata['ps'].values
                ps_interp = (1 - t) * ps_below + t * ps_above

                resampled_rings.append(pd.DataFrame({'r': center_r, 'theta': thetas, 'pt': pt_interp, 'ps': ps_interp}))

        resampled_data = pd.concat(resampled_rings, ignore_index=True)
        return Face(resampled_data, critangle=self.critangle)
    
    def resample(self, r_n: int, theta_n: int) -> 'Face':
        """Resample the face to r_n rings and theta_n points per ring, returning a new Face object with the resampled data."""
        return self.resample_r(r_n).resample_theta(theta_n)


import pandas as pd
from typing import List
import numpy as np


def dfcheck(df:pd.DataFrame) -> bool:
    """
    Check if the dataframe has the required columns for SAE16 calculations.
    """
    required_columns = ['r', 'theta', 'p']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
    return True

def findrings(data:pd.DataFrame,tolerance=0.05) -> List[pd.DataFrame]:
    """
    Find the rings of concentric measurements in an inlet plane
    """
    if not dfcheck(data):
        raise ValueError("DataFrame does not have the required columns.")
    rmax = data['r'].max()
    tol = rmax * tolerance
    
    

def findzero_crossing(data:pd.DataFrame,pavg:float) -> List[float]:
    """
    Find the points where the data crosses zero. Interpolates between points to find the exact
    crossing point. Assumes that the data is ordered by 'theta' and that 'p' is the value being analyzed for crossings with respect to pavg.
    """
    zerocrossings = []
    for i in range(len(data) - 1):
        x0 = data.loc[i, 'theta']
        x1 = data.loc[i+1, 'theta']
        y0 = data.loc[i, 'p'] - pavg
        y1 = data.loc[i+1, 'p'] - pavg
        if y0 * y1 > 0:
    # No zero crossing
            continue
        else:
    # Linear interpolation to find the exact zero crossing
            zero_crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
            zerocrossings.append(zero_crossing)
    return zerocrossings

def findnegative_segments(data:pd.DataFrame,pavg:float) -> List[pd.DataFrame]:
    """
    Find the segments of the data where p is below the average value.
    """
    segments = []
    zero_crossings = findzero_crossing(data,pavg)
    if len(zero_crossings) < 2:
        raise ValueError("Not enough zero crossings to define segments.")
    # Evaluate all adjacent crossing pairs plus the wrap-around pair (last -> first).
    crossing_pairs = list(zip(zero_crossings, zero_crossings[1:]))
    crossing_pairs.append((zero_crossings[-1], zero_crossings[0]))

    for left, right in crossing_pairs:
        if left <= right:
            segment = data[(data['theta'] >= left) & (data['theta'] <= right)]
            #Add the actual zero crossing points to the segment
            segment = pd.concat([segment, pd.DataFrame({'theta': [left, right], 'p': [pavg, pavg]})], ignore_index=True)
            segment = segment.sort_values(by='theta').reset_index(drop=True)
        else:
            # Circular interval that spans the end and beginning of theta.
            segment = data[(data['theta'] >= left) | (data['theta'] <= right)]
            #Add the actual zero crossing points to the segment
            segment = pd.concat([segment, pd.DataFrame({'theta': [left, right], 'p': [pavg, pavg]})], ignore_index=True)
            segment = segment.sort_values(by='theta').reset_index(drop=True)
        if segment.empty:
            continue
        if segment['p'].iloc[0] < pavg:
            segments.append(segment)
    return segments

#def calculate_area(segment:pd.DataFrame,pavg:float) -> float:

class Ring:
    def __init__(self,ringdata:pd.DataFrame):

        if "r" not in ringdata.columns:
            raise ValueError("Ring data must contain 'r' column.")
        if "p" not in ringdata.columns:
            raise ValueError("Ring data must contain 'p' column.")
        if "theta" not in ringdata.columns:
            raise ValueError("Ring data must contain 'theta' column.")
        self.ringdata = ringdata

    def PAV(self) -> float:
        return self.ringdata['p'].mean()
    
    def CDI(self) -> float:
        #Find the regions of the ring where p is below the average value
        pavg = self.PAV()
        #Find the zero crossings of p - pavg
        zero_crossings = findzero_crossing(self.ringdata,pavg)
        if len(zero_crossings) < 2:
            raise ValueError("Not enough zero crossings to define segments.")
        zero_segments = findnegative_segments(self.ringdata,pavg)

        #Interpolate the zero crossings to find the exact points where p crosses pavg
        zero_crossings_interp = []
        for i in range(len(zero_crossings)-1):
            x0 = self.ringdata['theta'].iloc[zero_crossings[i]]
            x1 = self.ringdata['theta'].iloc[zero_crossings[i]+1]
            y0 = self.ringdata['p'].iloc[zero_crossings[i]] - pavg
            y1 = self.ringdata['p'].iloc[zero_crossings[i]+1] - pavg
            if y0 * y1 > 0:
                continue
            zero_crossing_interp = x0 - y0 * (x1 - x0) / (y1 - y0)
            zero_crossings_interp.append(zero_crossing_interp)
        zero_crossings_interp = np.array(zero_crossings_interp)


        for i in range(len(zero_crossings)-1):
            left = zero_crossings[i]
            right = zero_crossings[i+1]
            segment = self.ringdata.iloc[left:right+1]
            if segment['p'].iloc[0] < pavg:
                segments.append(segment)
        if len(segments) == 0:
            return 0.0
        areas = []
    
    
class Face:
    """Class that represents the rings that make up a face and calculates the SAE16 Intensity metric for the face."""

    def __init__(self,rings:List[Ring]):
        self.rings = rings

    def PFAV(self) -> float:
        return np.mean([ring.PAV() for ring in self.rings])

    def RDI(self,i) -> float:
        return (self.PFAV() - self.rings[i].PAV()) / self.PFAV()
    
    

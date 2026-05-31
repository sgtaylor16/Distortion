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
            ring_df = sorted_data.iloc[start_idx:i].copy().reset_index(drop=True)
            rings.append(ring_df)
            start_idx = i
        prev_r = current_r

    # Add the final ring.
    rings.append(sorted_data.iloc[start_idx:].copy().reset_index(drop=True))
    return rings

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
        if segment['p'].mean() < pavg:
            segments.append(segment)
    return segments

def findnegative_segments_split_wrap(data:pd.DataFrame,pavg:float) -> List[pd.DataFrame]:
    """
    Find segments where p is below pavg, but never create a wrapped segment.

    If the interval between two consecutive zero crossings wraps around theta
    (last crossing -> first crossing), split it into two independent segments:
    one at the end of theta and one at the beginning.
    """
    segments = []
    zero_crossings = findzero_crossing(data, pavg)
    if len(zero_crossings) < 2:
        raise ValueError("Not enough zero crossings to define segments.")

    theta_min = data['theta'].min()
    theta_max = data['theta'].max()

    # Adjacent, non-wrapping crossing pairs.
    for left, right in zip(zero_crossings, zero_crossings[1:]):
        segment = data[(data['theta'] >= left) & (data['theta'] <= right)]
        segment = pd.concat(
            [segment, pd.DataFrame({'theta': [left, right], 'p': [pavg, pavg]})],
            ignore_index=True
        )
        segment = segment.sort_values(by='theta').reset_index(drop=True)
        if not segment.empty and segment['p'].mean() < pavg:
            segments.append(segment)

    # Split the wrap-around interval into two non-wrapping segments.
    left = zero_crossings[-1]
    right = zero_crossings[0]

    end_segment = data[data['theta'] >= left]
    end_segment = pd.concat(
        [end_segment, pd.DataFrame({'theta': [left, theta_max], 'p': [pavg, pavg]})],
        ignore_index=True
    )
    end_segment = end_segment.sort_values(by='theta').reset_index(drop=True)
    if not end_segment.empty and end_segment['p'].mean() < pavg:
        segments.append(end_segment)

    start_segment = data[data['theta'] <= right]
    start_segment = pd.concat(
        [start_segment, pd.DataFrame({'theta': [theta_min, right], 'p': [pavg, pavg]})],
        ignore_index=True
    )
    start_segment = start_segment.sort_values(by='theta').reset_index(drop=True)
    if not start_segment.empty and start_segment['p'].mean() < pavg:
        segments.append(start_segment)

    return segments

def area_bar(segment:pd.DataFrame,pavg:float) -> float:
    """
    Calculate the area of the segment below pavg using the trapezoidal rule.
    Assumes that the segment is ordered by 'theta'.
    """
    if segment.empty:
        return 0.0
    # Subtract pavg from p to get the area below pavg.
    y = segment['p'] - pavg
    x = segment['theta']
    area = np.trapz(y, x) / (segment['theta'].max() - segment['theta'].min())  # Normalize by the theta range to get an average area.
    return max(0.0, -area)  # Area should be positive, so take negative of the result.

class Segment:
    def __init__(self,segmentdata:pd.DataFrame,pavg:float):
        self.segmentdata = segmentdata
        self.pavg = pavg
        self.area_bar = area_bar(segmentdata,pavg)
        if (self.segmentdata.iloc[0]['theta'] < .01) and (self.segmentdata.iloc[-1]['theta'] > 359.99):
            self.extent = (360 - self.segmentdata.iloc[-1]['theta']) + self.segmentdata.iloc[0]['theta']
        else:
            self.extent = self.segmentdata['theta'].max() - self.segmentdata['theta'].min()

class Ring:
    def __init__(self,ringdata:pd.DataFrame):

        if "r" not in ringdata.columns:
            raise ValueError("Ring data must contain 'r' column.")
        if "p" not in ringdata.columns:
            raise ValueError("Ring data must contain 'p' column.")
        if "theta" not in ringdata.columns:
            raise ValueError("Ring data must contain 'theta' column.")
        self.ringdata = ringdata

        # Find Segments
        self.pavg = self.ringdata['p'].mean()
        zerosements = findnegative_segments(ringdata,self.pavg)
        self.segments = [Segment(seg, self.pavg) for seg in zerosements]

    def PAV(self) -> float:
        return self.ringdata['p'].mean()
    
    def extents(self) -> List[float]:
        pavg = self.PAV()
        zero_crossings = findzero_crossing(self.ringdata,pavg)
        zero_segments = findnegative_segments(self.ringdata,pavg)

    
    def CDI(self) -> float:
        #Find the regions of the ring where p is below the average value
        pavg = self.PAV()
        #Find the zero crossings of p - pavg
        zero_crossings = findzero_crossing(self.ringdata,pavg)
        if len(zero_crossings) < 2:
            raise ValueError("Not enough zero crossings to define segments.")
        zero_segments = findnegative_segments_split_wrap(self.ringdata,pavg)

        if len(zero_segments) == 0:
            return 0.0
    
        elif len(zero_segments) == 1:
            pavlow = area_bar(zero_segments[0],pavg)

            return (pavg - pavlow) / pavg
    
    
class Face:
    """Class that represents the rings that make up a face and calculates the SAE16 Intensity metric for the face."""

    def __init__(self,datadf:pd.DataFrame,tolerance=0.05):
        ringsegments = findrings(datadf,tolerance)
        self.rings = [Ring(ringdata) for ringdata in ringsegments]

    def PFAV(self) -> float:
        return np.mean([ring.PAV() for ring in self.rings])

    def RDI(self,i) -> float:
        return (self.PFAV() - self.rings[i].PAV()) / self.PFAV()
    
    def CDI(self,i) -> float:
        return self.rings[i].CDI()
    
    

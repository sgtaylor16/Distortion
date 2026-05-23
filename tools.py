import numpy as np
from typing import List

def findzero_crossing(data:np.ndarray,pavg:float) -> float:
    """
    Find the point where the data crosses zero.
    """
    x0 = data[0][0]
    x1 = data[1][0]
    y0 = data[0][1] - pavg
    y1 = data[1][1] - pavg
    if y0 * y1 > 0:
        # No zero crossing
        return np.nan
    # Linear interpolation to find the exact zero crossing
    return x0 - y0 * (x1 - x0) / (y1 - y0)

def stringbetween(s:str, start:str, end:str) -> str:
    """
    Extract a substring from s that is between the first occurrence of start and end.
    """
    start_index = s.find(start)
    if start_index == -1:
        return ""
    start_index += len(start)
    end_index = s.find(end, start_index)
    if end_index == -1:
        return ""
    return s[start_index:end_index]

def findzeros(data:np.ndarray,pavg:float=0.0) -> np.ndarray:
    """
    Find all zero crossings in the data.
    """
    zeros = []
    for i in range(len(data) - 1):
        y0 = data[i][1] - pavg
        y1 = data[i + 1][1] - pavg
        if y0 * y1 <= 0:
            zero_crossing = findzero_crossing(data[i:i + 2],pavg)
            if not np.isnan(zero_crossing):
                zeros.append(zero_crossing)
    return np.sort(np.array(zeros))

def findzeropairs(zeros:List[float],data:np.ndarray,pavg:float) -> List[tuple]:
    """
    Pair up zero crossings to define intervals.
    """
    zeropairs = []
    if (data[0,1] < pavg) and (data[-1,1] < pavg):
        zeropairs.append((zeros[-1], zeros[0]))
    for i in range(len(zeros) - 1):
        left = zeros[i]
        right = zeros[i + 1]
        if left <= right:
            includeddata = data[(data[:,0] >= left) & (data[:,0] <= right)]
        else:
            includeddata = data[(data[:,0] >= left) | (data[:,0] <= right)]
        if includeddata[0,1] < pavg:
            zeropairs.append((left, right))

    return zeropairs

def split_data_by_zeropairs(zeropairs:List[tuple],data:np.ndarray,pavg:float) -> List[np.ndarray]:
    """
    Split the data into segments defined by zeropairs.
    """
    newy = data[:,1] - pavg
    data = np.column_stack((data[:,0], newy))
    segments = []
    for left, right in zeropairs:
        if left <= right:
            segment = data[(data[:,0] >= left) & (data[:,0] <= right)]
        else:
            segment = data[(data[:,0] >= left) | (data[:,0] <= right)]
        group = {'left':left,'right':right,'data':segment}
        segments.append(group)
    return segments



class Area():
    """
    Class to calculate the area under a curve using the trapezoidal rule.
    """
    def __init__(self,leftzero:float,rightzero:float, data:np.ndarray):
        self.leftzero = np.array([[leftzero,0]])
        self.rightzero = np.array([[rightzero,0]])
        if self.leftzero[0][0] <= self.rightzero[0][0]:
            self.pts = np.concatenate((self.leftzero,data,self.rightzero), axis=0)
        else:
            enddata = data[data[:,0] >= self.leftzero[0][0]]
            begdata = data[data[:,0] <= self.rightzero[0][0]]
            self.pts =  np.concatenate((self.leftzero,enddata,begdata,self.rightzero))


    def area(self) -> float:
        """
        Compute the area under the curve between leftzero and rightzero.
        """
        if self.leftzero[0][0] <= self.rightzero[0][0]:
            # Calculate area using the trapezoidal rule
            area = np.trapz(self.pts[:,1], self.pts[:,0])
        else:
            # Wrap around case
            enddata = self.pts[self.pts[:,0] >= self.pts[0,0]]
            begdata = self.pts[self.pts[:,0] <= self.pts[-1,0]]
            area = np.trapz(enddata[:,1], enddata[:,0]) + np.trapz(begdata[:,1], begdata[:,0])

        return area
    
    def pavlow(self) -> float:
        """
        Calculate the average value of y over the interval [leftzero, rightzero].
        """
        area = self.area()
        if self.leftzero[0][0] <= self.rightzero[0][0]:
            interval_length = self.pts[-1,0] - self.pts[0,0]
        else:
            enddata = self.pts[self.pts[:,0] >= self.pts[0,0]]
            begdata = self.pts[self.pts[:,0] <= self.pts[-1,0]]
            interval_length = (enddata[-1,0] - enddata[0,0]) + (begdata[-1,0] - begdata[0,0])
        if interval_length == 0:
            return 0.0
        return area / interval_length
    
    def extent(self) -> float:
        """
        Calculate the extent of the interval [leftzero, rightzero].
        """
        if self.leftzero[0][0] <= self.rightzero[0][0]:
            return self.pts[-1,0] - self.pts[0,0]
        else:
            enddata = self.pts[self.pts[:,0] >= self.pts[0,0]]
            begdata = self.pts[self.pts[:,0] <= self.pts[-1,0]]
            return (enddata[-1,0] - enddata[0,0]) + (begdata[-1,0] - begdata[0,0])
    
    def intensity(self,pavg:float) -> float:
        """
        Calculate the SAE16 Intensity metric.
        """
        if pavg == 0:
            raise ValueError("pavg cannot be zero")
        return (pavg - self.pavlow()) / pavg


class Distortionring():

    def __init__(self,segments:List[dict],pavg:float):
        self.segments = segments

        self.areas = [Area(seg['left'],seg['right'],seg['data']) for seg in segments]
        
        total_extent = 0.0
        total_area = 0.0
        for area in self.areas:
            total_extent += area.extent()
            total_area += area.area()
        self.total_extent = total_extent
        self.total_area = total_area
        self.pavelow = total_area / total_extent
        self.intensity = (pavg - self.pavelow) / pavg



    



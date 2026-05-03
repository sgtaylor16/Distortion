from typing import List,Callable
import numpy as np

def convert_to_cart(data:List[dict]) -> List[dict]:
    for d in data:
        d['x'] = d['r']*np.cos(d['theta'])
        d['y'] = d['r']*np.sin(d['theta'])
    return data

class Setup:
    def __init__(self,diameter:float):
        self.diameter = diameter

    def checkrakes(rakes:List[dict]) -> bool:
        for rake in rakes:
            if 'r' not in rake.keys() or 'theta' not in rake.keys():
                raise ValueError('Rake must have keys r and theta')
        return True

    def set_rakes(self,rakes:List[dict]):
        if self.checkrakes(rakes):
            self.rakes = rakes
        

    def createDataPolar(self,measFunc:Callable) -> List[dict]:
        data = []
        for rake in self.rakes:
            value = measFunc(rake['r'],rake['theta'])
            data.append({'r':rake['r'],'theta':rake['theta'],'meas':value})
        return data
    
    



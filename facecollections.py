from distortion.sae16 import Face, Ring
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FaceCollection:

    def __init__(self,faces:List[Face],conditionlist:List[int]=None):
        self.faces = faces
        self.nfaces = len(faces)
        self.conditionlist = conditionlist

        #Check that all faces have the same stackheight
        heightlist = [face.calc_stackheight() for face in self.faces]
        if not all(height == heightlist[0] for height in heightlist):
            raise ValueError("All faces must have the same stack height")

        self.feature_height = heightlist[0]

    def harmonic_matrix(self, order:int,valuelist:List[str]) -> np.ndarray:
        """Calculate the harmonic matrix for the collection of faces.
        Each column corresponds to a face. The rows correspond to the stacked harmonic values for each value in valuelist."""

        #Initialize an empty numpy array
        harmonic_matrix = np.zeros((self.feature_height * len(valuelist), self.nfaces))
        for k,face in enumerate(self.faces):
            harmonics_dict = {}
            for value in valuelist:
                #Create a new face with the mean subtracted.
                newface = Face(face.df.copy() - face.df[value].mean())
                harmonics_dict[value] = newface.calcHarmonic(order, value)[value].to_numpy().reshape(-1,1,order='F')
            harmonic_matrix[:,k] = np.vstack([harmonics_dict[value] for value in valuelist]).flatten()

        return harmonic_matrix

    def harmonic_svd(self,order:int,valuelist:List[str]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Calculate the SVD of the harmonic matrix for the collection of faces."""
        harmonic_matrix = self.harmonic_matrix(order,valuelist)

        #Subtract mean 
        harmonic_matrix = harmonic_matrix - np.tile(np.mean(harmonic_matrix, axis=1, keepdims=True), (1, harmonic_matrix.shape[1]))

        U, S, VT = np.linalg.svd(harmonic_matrix, full_matrices=False)
        return U, S, VT

    def extract_svd_feature(self,svd_order:int,mode_num:int,valuelist:List[str],feature:str) -> np.ndarray:
        """Extract a specific feature from the SVD of the harmonic matrix for the collection of faces."""

        if feature not in valuelist:
            raise ValueError(f"Feature '{feature}' not found in valuelist.")

        U, S, VT = self.harmonic_svd(mode_num,valuelist=valuelist)

        if svd_order >= U.shape[1]:
            raise ValueError(f"Order {svd_order} is out of bounds for the harmonic matrix with columns {U.shape[1]}.")

        #Extract the correct column
        U_order = U[:, svd_order]

        #Extract the correct feature
        for idx, val in enumerate(valuelist):
            if val == feature:
                feature_idx = idx
                break
        feature_vector = U_order[feature_idx * self.feature_height : (feature_idx * self.feature_height + self.feature_height)]

        return feature_vector

    def restack_svd_feature(self,order:int,mode_num:int,valuelist:List[str],feature:str) -> pd.DataFrame:
        """
        Takes the column vector caclulated by svd and places it back in a df formatted the same as face.
        """
        columnvector = self.extract_svd_feature(order,mode_num,valuelist,feature)

        #Grab an arbitrary face dataframe from the collection, they should all be the same.
        df = self.faces[0].df[['radius','theta']].copy()

        #Check to make sure that the rows of df match the length of the column vector
        if df.shape[0] != len(columnvector):
            raise ValueError()

        df[feature] = columnvector

        return df

    def plot_collection(self,value='pt',ncols:int=4) -> plt.Figure:
        """Plot every face in the collection on a grid, ncols faces across."""
        nrows = int(np.ceil(self.nfaces / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
        flat_axes = axes.ravel()

        for k, face in enumerate(self.faces):
            face.plotFace(value=value, includepts=False, colorbar=False, ax=flat_axes[k])
            if self.conditionlist is not None:
                flat_axes[k].set_title(str(self.conditionlist[k]), fontsize=8)

        for ax in flat_axes[self.nfaces:]:
            ax.axis('off')

        fig.tight_layout()
        return fig

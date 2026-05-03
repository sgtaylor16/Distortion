import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.tri import Triangulation
import numpy as np

def plot_inlet_data(df: pd.DataFrame,valuecol:str) -> None:
    """
    Plots the inlet data using a triangulation plot.

    Parameters:
        df (pd.DataFrame): DataFrame containing the inlet data with columns 'theta', 'radius', and the specified value column.
        valuecol (str): The name of the column in the DataFrame to be plotted as the color values.

    Returns:
        None: Displays a plot of the inlet data.
    """
    # Ensure required columns are present
    required_columns = ['theta', 'radius', valuecol]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain the required column: '{col}'")

    # Create triangulation
    theta_array = df['theta'].to_numpy()
    radius_array = df['radius'].to_numpy()
    x:np.array = radius_array * np.cos(theta_array)
    y:np.array = radius_array * np.sin(theta_array)

    tri = Triangulation(x,y)
    min_radius = float(radius_array.min())

    # Mask triangles whose centroids fall inside the inner measured radius.
    triangles = tri.triangles
    x_tri = x[triangles]
    y_tri = y[triangles]
    centroid_radius = np.sqrt(np.mean(x_tri, axis=1) ** 2 + np.mean(y_tri, axis=1) ** 2)
    tri.set_mask(centroid_radius < min_radius)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.tricontourf(tri, df[valuecol], cmap='viridis')
    plt.show()



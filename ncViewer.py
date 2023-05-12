import xarray as xr
import numpy as np
import pandas as pd

fn = './nomiss-merra4/Aurora.0200801.nc4'


ds= xr.open_dataset('./MERRA2_400_202008_2d/MERRA2_400.inst1_2d_lfo_Nx.20200824.nc4.nc4')
#ds = xr.open_dataset('./my-nomiss-merra-new/Humboldt-0200803.nc4')
#data array (one variable)
#da = ds["t"]

#show dataset variables
#ds.data_vars

#ds.info()
print(ds.info())
#ds.T[0].plot()




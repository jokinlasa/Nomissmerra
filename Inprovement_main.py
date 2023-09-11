import xarray as xr
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def log_interpolation(var_name, **kwargs):
    var_in = np.array(var_name, dtype=np.float64)
    nans, x = np.isnan(var_in), lambda z: z.nonzero()[0]
    if var_name == tin:
        var_in[nans] = np.exp(np.interp(np.log(x(nans)), np.log(x(~nans)), np.log(var_in[~nans])))

        var_out = var_in[:-1]

    elif var_name == qin:
        var_in[nans] = np.interp(x(nans), x(~nans), var_in[~nans])
        var_out = var_in[:-1]
    elif var_name == o3_merra:
        '''pout_rmvmsng = pout
        nan_idx = np.argwhere(np.isnan(np.array(o3_merra)))
        nan_idx = [ijk for ijk in nan_idx]
        for idx in sorted(nan_idx, reverse=True):
            del pout_rmvmsng[int(idx)]
        print(len(pout_rmvmsng))
        print(len(var_in[~nans]))'''

        var_in[nans] = np.interp(x(nans), x(~nans), var_in[~nans])
        # var_in = np.exp(np.interp(np.log(plev), np.log(pout_rmvmsng), np.log(var_in[~nans])))
        # var_in = np.exp(np.interp(np.log(plev), np.log(pout), np.log(var_in)))
        var_out = var_in

    return var_out


nplev = 72  # number of levels
indir2d = './MERRA2_400_202008_2d/'
indir3d = './MERRA2_400_202008_3d/'

station_info = pd.read_csv('station_radiation.txt')
stn_names = station_info.iloc[:, 1].values
latitude = station_info.iloc[:, 2].values
longitude = station_info.iloc[:, 3].values

list2d = os.listdir(indir2d)
list3d = os.listdir(indir3d)

for numfiles, file in enumerate(list2d):
    break
    #%%
    file = 'MERRA2_400.inst1_2d_lfo_Nx.20200801.nc4.nc4'
    date = file[27:-8]
    file2d = indir2d + file
    file3d = indir3d + list3d[numfiles]
    print('Reading', file, 'and', list3d[numfiles])

    # these two files below are the same for all sites
    ds2d = xr.open_dataset(file2d)
    ds3d = xr.open_dataset(file3d).resample(time='H').nearest()
    # we need to create an hour 00 and 23 basically identical to the existing
    # hours 01 and 22. We first make two slices, then update the times, then merge.
    first_hour = ds3d.isel(time=0)
    first_hour['time'] = first_hour.time - pd.Timedelta(hours=1)
    last_hour = ds3d.isel(time=-1)
    last_hour['time'] = last_hour.time + pd.Timedelta(hours=1)
    ds3d = xr.concat((first_hour, ds3d, last_hour), dim='time')

    # now we can actually merge the two object
    ds_day = xr.merge((ds2d, ds3d))

    for station, lat, lon in zip(stn_names, latitude, longitude):
        # break
        # # %%
        # station, lat, lon = ('Swiss Camp 10m', 69.5556, -49.3647)
        ds_day_site = ds_day.interp(lat=lat, lon=lon, method='nearest')
        # > ds_day_site
        # Out[147]:
        # <xarray.Dataset>
        # Dimensions:  (time: 24, lev: 72)
        # Coordinates:
        #     lat      float64 69.5
        #     lon      float64 -49.38
        #   * time     (time) datetime64[ns] 2020-08-01 ... 2020-08-01T23:00:00
        #   * lev      (lev) float64 1.0 2.0 3.0 4.0 5.0 6.0 ... 68.0 69.0 70.0 71.0 72.0
        # Data variables:
        #     QLML     (time) float32 ...
        #     TLML     (time) float32 ...
        #     PS       (time) float32 9.131e+04 9.131e+04 ... 9.133e+04 9.133e+04
        #     T        (lev, time) float32 172.2 172.2 172.2 170.9 ... 276.0 276.0 276.0
        #     PL       (lev, time) float32 1.5 1.5 1.5 ... 9.065e+04 9.065e+04 9.065e+04
        #     QL       (lev, time) float32 0.0 0.0 0.0 ... 6.527e-09 6.527e-09 6.527e-09
        #     QV       (lev, time) float32 4.067e-06 4.067e-06 ... 0.004387 0.004387
        #     O3       (lev, time) float32 6.484e-06 6.484e-06 ... 5.024e-08 5.024e-08

        # for hr in hours:
        # > We now deal with all time steps at the same time

        # Pressure at various levels
        pin = ds_day_site['PL'] / 100

        # surface pressure
        ps_merra = ds_day_site['PS'] / 100

        # plt.figure()
        # for lev in range(len(pin.lev)):
        #     pin.isel(lev=lev).plot(marker='o')
        # conclusion: last layer is highest, so closest to surface

        # plt.figure()
        # pin.isel(lev=-1).plot(marker='o')
        # ps_merra.plot(marker='^')
        # conclusion: surface pressure is even higher than the last layer
        # in pin. So we can append it as a new (73rd) layer.
        # That is what
        #                 pin[-1] = ps_merra
        # was doing.

        ps_merra = ps_merra.expand_dims(dim={"lev": [73.]}).rename('PL')
        pin = xr.merge((pin, ps_merra))['PL']

        # plt.figure()
        # pin.isel(lev=-1).plot(marker='o')
        # ps_merra.plot(marker='^')
        # now the 73rd level is the surface

        # Temperature
        tin = ds_day_site['T']

        # here TLML is vailable in ds_temp2d
        # the problem is that ds_temp3d is at 3h resolution
        # while ds_temp2d is at 1h resolution
        # We solve this by interpolating the 3-hourly 3d data into hourly data

        # surface temperature
        ta_merra = ds_day_site['TLML']
        # adding ta_merra as extra layer in tin (tin[-1] = ta_merra)
        ta_merra = ta_merra.expand_dims(dim={"lev": [73.]}).rename('T')
        tin = xr.merge((tin, ta_merra))['T']

        ts_merra = ta_merra.copy()  # here making a copy of ta_merra, no idea why
        # I'm assuming that since we are on ice, surface temp cannot be over melting point
        ts_merra = xr.where(ts_merra > 273.15, 273.15, ts_merra)

        qin = ds_day_site['QV']  # Water vapor mixing ratio
        o3_merra = ds_day_site['O3']  # ozone

        # take data 2d
        # tin, qin = take_data_2d_files(ds_temp2d, tin, qin)
        # ts_merra = 0

        # Get index of all values, where pressure >= surface pressure
        # Set the values at above indexes as missing
        pressureSafetyMeasure = 5

        for lev in range(1, 73):
            msk = pin.loc[dict(lev=lev)] > (pin.isel(lev=-1) - pressureSafetyMeasure)
            tin.loc[dict(lev=lev)] = tin.loc[dict(lev=lev)].where(~msk)
            qin.loc[dict(lev=lev)] = qin.loc[dict(lev=lev)].where(~msk)
            pin.loc[dict(lev=lev)] = pin.loc[dict(lev=lev)].where(~msk)

        # Interpolate the above missing values
        assert (pin.isnull().sum() > 0, 'need for interpolation')
        assert (tin.isnull().sum() > 0, 'need for interpolation')
        assert (o3_merra.isnull().sum() > 0, 'need for interpolation')
        # pin = pd.Series(pin)
        # pin = pin.interpolate(limit_direction='both').values.tolist()

        # I don't get the part below
        # pout = pin[:-1]
        # aod_count = 0
        # for press in pout:
        #     if (pin[-1] - 100) < press < pin[-1]:
        #         aod_count += 1

        greater = (pin.loc[dict(lev=slice(0,72))] > (pin.isel(lev=-1) - 100))
        lower = (pin.loc[dict(lev=slice(0,72))] < pin.isel(lev=-1))
        aod_count = (greater & lower).sum(dim='lev') 

        # here there some interpolation being done, but first the log_interpolation
        # function doesn't make sense, then there should not be gaps in
        # the model output
        # if checkna(tin) and checkna(qin) and checkna(o3_merra):
        #     tout = log_interpolation(tin)
        #     qout = log_interpolation(qin)
        #     o3_out = log_interpolation(o3_merra, plev=plev, pout=pout)

        ds = xr.merge((
            pin.rename('plev'),
            tin.rename('t'),
            qin.rename('q'),
            o3_merra.rename('o3'),
            ta_merra.rename('ta'),
            ts_merra.rename('ts'),  # there must be a difference between the two
            ps_merra.rename('ps'),
            aod_count.rename('aod_count'),
        ))

        #  print(ds.info())
        # Write to netCDF file
        if not os.path.exists('nomiss-merra-result-summer2013/' + station):
            os.makedirs('nomiss-merra-result-summer2013/' + station)


        outfile = 'nomiss-merra-result-summer2013/' + station + '/' + station + '-' + date + '.nc'
        ds.to_netcdf(outfile)



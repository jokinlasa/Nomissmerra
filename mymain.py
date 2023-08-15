import xarray as xr
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')


def get_ds_latlon(infile):
    ds = xr.open_dataset(infile)

    var_lat = ds['lat'].values
    var_lon = ds['lon'].values

    lat_lon_var = []

    for i in var_lat:
        for j in var_lon:
            lat_lon_var.append([i, j])

    return lat_lon_var


def haversine_np(latlon1, latlon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    latlon1, latlon2 = map(np.radians, [latlon1, latlon2])

    i = 0
    stn_new = []

    for item in latlon1:
        j = 0
        dist = []
        for value in latlon2:
            dlat = latlon1[i][0] - latlon2[j][0]
            dlon = latlon1[i][1] - latlon2[j][1]

            a = np.sin(dlat / 2.0) ** 2 + np.cos(latlon2[j][0]) * np.cos(latlon1[i][0]) * np.sin(dlon / 2.0) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            km = 6367 * c
            dist.append(km)
            j += 1
        idx = dist.index(min(dist))
        stn_new.append(latlon2[idx])
        i += 1

    stn_new = list(map(np.degrees, stn_new))
    return stn_new


def get_stn_latlonname(station_file):
    lst_stn = pd.read_csv(station_file)
    stn_names = lst_stn['network_name'].tolist()

    latstn = lst_stn['lat'].tolist()
    lonstn = lst_stn['lon'].tolist()
    lonstn = [360 + i if i < 0 else i for i in lonstn]

    lat_lon_stn = np.column_stack((latstn, lonstn))

    return stn_names, lat_lon_stn


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


def checkna(variable):
    variable = pd.to_numeric(variable, errors='coerce')
    if np.count_nonzero(~np.isnan(variable)) >= len(variable) / 3:
        return True
    else:
        return False


def clean_multiple_list(list_name):
    cleaned_list = [item for sublist in list_name for item in sublist if str(item) != 'nan']

    return cleaned_list


def clean_single_list(list_name):
    cleaned_list = [item for item in list_name if str(item) != 'nan']
    val = cleaned_list[0]

    return val


def subset_dataset_for_dimension(ds, stn_new, x_coord):  # for both of the files
    y_coord = 0
    temp = {'lat': stn_new[x_coord][y_coord],
            'lon': stn_new[x_coord][y_coord + 1]
            }
    try:
        ds_sub = ds.sel(temp)  # Subset the dataset for only these dimensions
    except:
        if stn_new[x_coord][y_coord + 1] == -5.920304394294029e-13:
            temp = {'lat': round(stn_new[x_coord][y_coord], 1),
                    'lon': stn_new[x_coord][y_coord + 1]
                    }
        else:
            temp = {'lat': round(stn_new[x_coord][y_coord], 1),
                    'lon': round(stn_new[x_coord][y_coord + 1], 3)
                    }
        ds_sub = ds.sel(temp)  # Subset the dataset for only these dimensions
    return ds_sub


def take_data_2d_files(ds_temp, tin, qin):
    ta_merra = clean_single_list(ds_temp['TLML'].values.tolist())
    if not np.isnan(ta_merra):
        tin[-1] = ta_merra

    qs_merra = clean_single_list(ds_temp['QLML'].values.tolist())
    if not np.isnan(qs_merra):
        qin[-1] = qs_merra

    return tin, qin


def take_data_3d_files(tin, qin, pin, ds_temp, nplev):
    # Pressure
    plev = ds_temp['PL'].values.tolist()
    plev = clean_multiple_list(plev)
    plev = [i / 100 for i in plev]
    pin[:nplev] = plev[:]

#PS MERRA ONLY HAS ONE VALUE THAT IS NOT NAN --> SHOULD BE LIKE THAT??????
    ps_merra = ds_temp['PS'].values.tolist()
    ps_merra = clean_single_list(ps_merra)
    ps_merra = ps_merra / 100
    if not np.isnan(ps_merra):
        pin[-1] = ps_merra

    # Temperature
    tin[:nplev] = clean_multiple_list(ds_temp['T'].values.tolist())
    #THERE IS NO SUCH A 'TLML' VARIABLE IN THE 3D FILE
    ta_merra = clean_single_list(ds_temp['TLML'].values.tolist())
    if not np.isnan(ta_merra):
       tin[-1] = ta_merra
    #TS IS EMPTY BECAUSE TLML DOESNT EXIST --> FROM WHERE SHOULD I TAKE THE TS DATA???
    ts_merra = ta_merra
   ####################################

    # Water vapor mixing ratio
    qin[:nplev] = clean_multiple_list(ds_temp['QV'].values.tolist())

    # ozone
    o3_merra = clean_multiple_list(ds_temp['O3'].values.tolist())

    return tin, qin, pin, o3_merra, plev, ps_merra


def takeFiles(path):
    fileslist = os.listdir(path)

    return fileslist


def main():
    nplev = 72
    indir2d = './MERRA2_400_202008_2d/'
    indir3d = './MERRA2_400_202008_3d/'

    nplev = 72

    global tin, qin, o3_merra
    #TIN AND QIN ARE NO CREATED --> WHY???? --> THATS WHY TIN IS EMPTY AND TA DOESNT HAVE DATA INSIDE
    pin, tin, qin = ([None] * (nplev + 1) for _ in range(3))

    stn_names, lat_lon_stn = get_stn_latlonname('station_radiation.txt')

    hours = [1, 4, 7, 10, 13, 16, 19, 22]

    list2d = takeFiles(indir2d)
    list3d = takeFiles(indir3d)

    numfiles = 0

    for file in list2d:
        date = file[27:-8]
        file2d = indir2d + file
        file3d = indir3d + list3d[numfiles]
        ds = xr.open_dataset(file2d)

        x_coord = 0
        lat_lon_var = get_ds_latlon(file2d)
        # calculate the circle to be investigated
        stn_new = haversine_np(lat_lon_stn, lat_lon_var)

        for item in stn_new:
            pout_final, tout_final, qout_final, o3_out_final = ([] for _ in range(4))
            #  print(file2d)
            ds2d = xr.open_dataset(file2d)
            #  print(list3d[numfiles])
            ds3d = xr.open_dataset(file3d)

            sub2d = subset_dataset_for_dimension(ds2d, stn_new, x_coord)
            sub3d = subset_dataset_for_dimension(ds3d, stn_new, x_coord)

            for hr in hours:

                ds_temp2d = sub2d.where(sub2d['time.hour'] == hr)
                ds_temp3d = sub3d.where(sub3d['time.hour'] == hr)
                tin, qin, pin, o3_merra, plev, ps_merra = take_data_3d_files(tin, qin, pin, ds_temp3d, nplev)
                tin, qin = take_data_2d_files(ds_temp2d, tin, qin)

                ts_merra = 0

                # Get index of all values, where pressure >= surface pressure
                pressureSafetyMeasure = 5
                pin_sub = pin[:-1]  # All values except surface pressure
                x = 0
                idx_lst = []
                for val in pin_sub:
                    if val > (pin[-1] - pressureSafetyMeasure):
                        idx_lst.append(pin_sub.index(pin_sub[x]))
                    x += 1

                # Set the values at above indexes as missing
                for idx in idx_lst:
                    for lst in [pin, tin, qin]:
                        lst[idx] = None

                # Interpolate the above missing values
                pin = pd.Series(pin)
                pin = pin.interpolate(limit_direction='both').values.tolist()
                pout = pin[:-1]

                aod_count = 0
                for press in pout:
                    if (pin[-1] - 100) < press < pin[-1]:
                        aod_count += 1

                if checkna(tin) and checkna(qin) and checkna(o3_merra):
                    tout = log_interpolation(tin)
                    qout = log_interpolation(qin)
                    o3_out = log_interpolation(o3_merra, plev=plev, pout=pout)
                else:
                    x_coord += 1
                    continue

                pout_final.append(pout)
                tout_final.append(tout)
                qout_final.append(qout)
                o3_out_final.append(o3_out)

            #   print(pout_final, tout_final, qout_final, o3_out_final)

            print(file2d)
            basename = os.path.basename(file2d)
            date = basename[-16:-8]
            time = pd.to_datetime([date + str(i).zfill(2) for i in [1, 4, 7, 10, 13, 16, 19, 22]], format='%Y%m%d%H')

            ds = xr.Dataset({'plev': (('time', 'PLEV'), pout_final),
                             't': (('time', 'PLEV'), tout_final),
                             'q': (('time', 'PLEV'), qout_final),
                             'o3': (('time', 'PLEV'), o3_out_final),
                             'ts': ts_merra,
                             'ta': tin[-1],
                             'ps': ps_merra,
                             'aod_count': aod_count},
                            coords=dict(
                                time=("time", time),
                                PLEV=("PLEV", np.arange(1, 73)),
                            )
                            )

            dw = ds["plev"]
            print(dw)
            #  print(ds.info())
            # Write to netCDF file
            if not os.path.exists('directories-per-location-time/' + stn_names[x_coord]):
                os.makedirs('directories-per-location-time/' + stn_names[x_coord])
            basename = os.path.basename(file2d)
            outfile = 'directories-per-location-time/' + stn_names[x_coord] + '/' + stn_names[x_coord] + '-' + basename[
                                                                                                               -15:-7] + 'nc4'

            ds.to_netcdf(outfile)

            dat = xr.open_dataset(outfile)

            dk = dat["plev"]

            print(outfile)
            print(dk)

            x_coord += 1
        numfiles += 1


if __name__ == '__main__':
    main()

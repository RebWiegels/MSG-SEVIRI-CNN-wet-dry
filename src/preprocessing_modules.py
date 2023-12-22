import numpy as np
import xarray as xr
from sklearn.feature_extraction import image

def find_nearest_pixelcoods(requested_lat, requested_lon, da_nearest_lat, da_nearest_lon):
    '''
    Find nearest pixel coordinates of given latitude and longitude information.
    -------
    requested_lat: array of float
        latitude of requested point
    requested_lon: array of float
        longitude of requested point
    da_nearest_lat: array of float
        latitudes of which to find the closest
    da_nearest_lon: array of float
        longitudes of which to find the closest
    ------
    Return
    
    '''
    # find the index of thr grid point nearest at specific lat/lon
    abslat = np.abs(da_nearest_lat - requested_lat)
    abslon = np.abs(da_nearest_lon - requested_lon)
    c = np.maximum(abslon, abslat)
    [(xloc), (yloc)] = np.where(c == np.min(c)) # !!! Watch out - exchange of x and y was necessary for the code to work
    # if more than one point is found, choose only one
    if len(xloc)>1:
        print('More than one location was found, but only one is choosen')
        print(xloc[0])
        xloc=[xloc[0]]
        yloc=[yloc[0]]
    return [(xloc), (yloc)]

def create_patch_dataset(ds_input, ds_target, i, patch_size=(9, 9)):
    '''
    Create patches out of two datasets for a certain timestep i.

    Parameters:
        ds_input (xarray.Dataset): Input dataset containing multiple channels.
        ds_target (xarray.Dataset): Target dataset containing the target variable (rainfall_amount).
        i (int): Timestep index.
        patch_size (tuple): Size of the patches to be extracted. Default is (9, 9).

    Returns:
        xarray.Dataset: Dataset containing input and target patches.
    '''
    # Create numpy array with needed channels in correct shape
    sev_array = np.asarray([np.asarray(ds_input.isel(time=i).IR_016), 
                            np.asarray(ds_input.isel(time=i).IR_039),
                            np.asarray(ds_input.isel(time=i).IR_087),
                            np.asarray(ds_input.isel(time=i).IR_108),
                            np.asarray(ds_input.isel(time=i).IR_120),
                            np.asarray(ds_input.isel(time=i).VIS006),
                            np.asarray(ds_input.isel(time=i).WV_062),
                            np.asarray(ds_input.isel(time=i).WV_073),
                            np.asarray(ds_input.LAT),         
                            np.asarray(ds_input.LON)
                           ])
    sev_arr = np.transpose(sev_array, [1, 2, 0])
    max_patches = (ds_input.y.size * ds_input.x.size) / (patch_size[0]*patch_size[1])
    
    # extract patches of input dataset
    patches = image.extract_patches_2d(sev_arr, patch_size, max_patches=int(max_patches))
    
    # initialize input and target lists
    target_patches = []
    input_patches = []
    ds_list = []
    for p in range (0, patches.shape[0]):
        
        # define corners of patch
        ll_lon, ll_lat = patches[p, 0, 0, 9], patches[p, 0, 0, 8]
        ur_lon, ur_lat = patches[p, patch_size[0]-1, patch_size[0]-1, 9], patches[p, patch_size[0]-1, patch_size[0]-1, 8]
        lr_lon, lr_lat = patches[p, 0, patch_size[0]-1, 9], patches[p, 0, patch_size[0]-1, 8]
        ul_lon, ul_lat = patches[p, patch_size[0]-1, 0, 9], patches[p, patch_size[0]-1, 0, 8]
        
        # get pixel coordinates of corners
        ([ur_yloc], [ur_xloc]) = find_nearest_pixelcoods(ur_lat, ur_lon, ds_target.LAT, ds_target.LON)
        ([ll_yloc], [ll_xloc]) = find_nearest_pixelcoods(ll_lat, ll_lon, ds_target.LAT, ds_target.LON)
        ([lr_yloc], [lr_xloc]) = find_nearest_pixelcoods(lr_lat, lr_lon, ds_target.LAT, ds_target.LON)
        ([ul_yloc], [ul_xloc]) = find_nearest_pixelcoods(ul_lat, ul_lon, ds_target.LAT, ds_target.LON)


        # cut of RADKLIM to target_patch
        # Check if patch size same to input patch size, else correct
        if not (ur_xloc-ll_xloc)==patch_size[1]:
            if (ur_xloc-ll_xloc)<patch_size[1]:
                ur_xloc = ur_xloc + 1
            if (ur_xloc-ll_xloc)>patch_size[1]:
                ur_xloc = ur_xloc - 1
        if not (ur_yloc-ll_yloc)==patch_size[0]:
            if (ur_yloc-ll_yloc)<patch_size[0]:
                ur_yloc = ur_yloc + 1
            if (ur_yloc-ll_yloc)>patch_size[0]:
                ur_yloc = ur_yloc - 1
        target_values = ds_target.rainfall_amount.isel(time=i).sel(x=slice(ll_xloc, ur_xloc), y=slice(ll_yloc, ur_yloc)) # adapt: loc=same as of seviri patch
        target_patch = np.asarray(target_values.values)

        # If target patch includes no NaNs the patches can be appended
        target_null = np.isnan(target_patch)
        if np.count_nonzero(target_null)==0:
            timestep = np.datetime_as_string(ds_input.time.isel(time=i).values)
            ds_patches = xr.Dataset(
                data_vars={
                        'input':(['ID', 'y1', 'x1', 'channels'], patches[p, :, :, :8].reshape(1, patches[p, :, :, :8].shape[0], patches[p, :, :, :8].shape[1], patches[p, :, :, :8].shape[2])),
                        'target': (['ID', 'y2', 'x2', 'rainfall'], target_patch.reshape(1, target_patch.shape[0], target_patch.shape[1], 1)),
                        'times': (['ID', 'time'], ds_input.time.isel(time=i).data.reshape(1, 1))
                         },
                coords={
                    'lat_low': (['ID', 'y1', 'x1'], patches[p, :, :, 8].reshape(1, patches[p, :, :, 8].shape[0], patches[p, :, :, 8].shape[1])),
                    'lon_low': (['ID', 'y1', 'x1'], patches[p, :, :, 9].reshape(1, patches[p, :, :, 9].shape[0], patches[p, :, :, 9].shape[1])),
                    'lat_high': (['ID', 'y2', 'x2'], target_values.LAT.values.reshape(1, target_values.LAT.values.shape[0], target_values.LAT.values.shape[1])),
                    'lon_high': (['ID', 'y2', 'x2'], target_values.LON.values.reshape(1, target_values.LON.values.shape[0], target_values.LON.values.shape[1]))
                }
                )

            ds_list.append(ds_patches)
                
        # If patch includes NaNs, the patch will be skipped
        else:
            continue

    if len(ds_list)==0: 
        ds_patches = np.nan
    else:
        ds_patches = xr.concat(ds_list, dim='ID')

    return ds_patches
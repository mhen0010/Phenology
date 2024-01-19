## dea_functions.py
'''
Description: This file contains a set of python functions to perform fractional cover
and lidar derived products analysis.


Functions included:
    
    load_fpar
    veg_classes
    stats_month
    stats_class

Last modified: Aug 2023

'''
# Import required packages
import datacube
import rasterio.crs
import geopandas as gpd
import pandas as pd
from datacube.utils import geometry
from datacube.utils import masking
from datacube.testutils.io import rio_slurp_xarray
from datacube.utils.cog import write_cog
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import warnings

from scipy.signal import savgol_filter, find_peaks
#Non-parametric ANOVA with post hoc tests
from scipy import stats

import sys
#sys.path.append('../Scripts')
#sys.path.insert(1, '.../dev/dea-notebooks/Tools/')
sys.path.insert(1,'/home/jovyan/dev/dea-notebooks/Tools') 

from dea_tools.spatial import xr_rasterize
from dea_tools.datahandling import wofs_fuser, load_ard, mostcommon_crs
#import temporal
from dea_tools.bandindices import calculate_indices


def chebyshev(ds, prob):
    
    std_ds = ds.std(dim=['time'])
    mean_ds = ds.mean(dim=['time'])
    
    #prob = 1/(k^2)
    k = np.sqrt(1/prob)
    
    upper_bound = mean_ds + (k*std_ds)
    lower_bound = mean_ds - (k*std_ds)
    
    
    ds = ds.where(ds<= upper_bound)
    ds = ds.where(ds>= lower_bound)
    
    #ds = xr.where(ds > upper_bound, np.nan, ds)
    #ds = xr.where(ds < lower_bound, np.nan, ds)
    
    return ds

def interpol_ts(ds):
    
    #ds_interp = ds.interp(coords=['time'], method='linear')
    ds_interp = ds.interpolate_na(dim='time', method='linear')
    
    return ds_interp


def smooth_savgol(ds, window_length, polyorder):
    
    def smoother(da, window_length, polyorder):
        return da.apply(savgol_filter, window_length=window_length, polyorder=polyorder, axis=0)

    # create kwargs dict
    kwargs = {'window_length': window_length, 'polyorder': polyorder}
    
    temp = xr.full_like(ds, fill_value=np.nan)
    ds = xr.map_blocks(smoother, ds, template=temp, kwargs=kwargs)
    
    return ds


def load_ndvi (dc, query, waterhole_shape, lidar_cov, lidar_height, perc_good=0):

    # dictionaries to save the outputs
    #ndvi = {}
    #chms_cov = {}
    #chms_avg = {}
    #waterhole = {}
    
    #Load waterhole shapefile and set col with the waterhole name
    gdf = gpd.read_file(waterhole_shape)
    
    # Extract the feature's geometry as a datacube geometry object
    geom = geometry.Geometry(geom=gdf.iloc[0].geometry, crs=gdf.crs)

    # Update the query to include our geopolygon
    query.update({'geopolygon': geom})
    
    # define native landsat crs
    native_ls = mostcommon_crs(dc, product=['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'], query=query)
    
    # Load landsat 3 collection
    ds = load_ard(dc=dc, products = ['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'], 
                  measurements = ['nbart_red', 'nbart_nir'],
                  output_crs =native_ls, resolution = (-30,30), align=(15,15), min_gooddata=0,
                  group_by = 'solar_day', **query, dask_chunks = {"time": 1}
                 )
    
    #print(ds)
    # Load LiDAR derived products
    chm_cov = rio_slurp_xarray(lidar_cov, gbox=ds.geobox, resampling='nearest')
    chm_avg = rio_slurp_xarray(lidar_height, gbox=ds.geobox, resampling='nearest')
    
    #print(chm_cov)
    
    # Generate a polygon mask to keep only data within the polygon
    mask = xr_rasterize(gdf.iloc[[0]], ds)
    
    # use the function calculate_indices to calculate the NDVI using ds
    calculate_indices(ds, index='NDVI', drop=True, collection='ga_ls_3', inplace=True)
    
    # use the polygon mask to remove everything that is not included in the 500 m buffer waterhole shapefile
    ds = ds.where(mask)
    ds = masking.mask_invalid_data(ds)

    chm_cov = chm_cov.where(mask)
    chm_avg = chm_avg.where(mask)
       
    # create a boolean raster to show the pixels that are included in the study area (500 m buffer)
    ds_notnull = chm_cov.notnull()
    
    # get the number of pixels that are included in the study area
    count_NDVIpixels = ds_notnull.where(ds_notnull==True).sum(dim=['x', 'y'])
    
    # define native landsat crs
    native_wofls = mostcommon_crs(dc, product=['ga_ls_wo_3'], query=query)
    
    # lod dea wo like landsat
    wofls = dc.load(product = 'ga_ls_wo_3', fuse_func = wofs_fuser, 
                    group_by = 'solar_day', like=ds, dask_chunks = {"time": 1})
    
    #print(wofls)
    
    wo_mask = masking.make_mask(wofls.water, dry=True)
    index_no_water = ds.where(wo_mask)
    
    index_no_water = index_no_water.compute()
    #print(index_no_water)
    
    wo_wet = masking.make_mask(wofls, wet=True)
    wo_dry = masking.make_mask(wofls, dry=True)
    wo_clear = wo_wet + wo_dry
    wo_clear = masking.make_mask(wofls, cloud_shadow=False, cloud=False, nodata=False)
    wo_masked = wo_wet.where(wo_clear).water
    wofl_freq = wo_masked.mean(dim=['time'])
    
    wofl_freq = wofl_freq.compute()
    #print(wofl_freq)
    
    # remove the pixels that are identified as water for 80% of the time
    chm_cov = chm_cov.where(wofl_freq <= 0.80)
    #print(chm_cov)
    #chm_cov.plot()
    
    chm_avg = chm_avg.where(wofl_freq <= 0.80)
    
    # set the base waterhole area as the pixels identified as water for at least 95% of the time
    min_waterhole = wofl_freq.where(wofl_freq > 0.80).count(['x', 'y'])
    waterhole_perc = min_waterhole / count_NDVIpixels
    
    # boolean raster with pixels flaged as dry
    nowater_data = index_no_water.NDVI.notnull()
    # number of good pixels (no water and no clouds)
    count_goodNDVI = nowater_data.where(nowater_data==True).sum(dim=['x', 'y'])

    # percentage of good quality pixels
    percent_gooddata = count_goodNDVI / count_NDVIpixels
    
    # keep only the observations percentage of good pixels higher or equal to perc_good
    index_no_water = index_no_water.sel(time=percent_gooddata >= perc_good)
    
    # resample and interpolate the data
    index_no_water = index_no_water.resample(time="16D").max('time')
    #print(index_no_water)
    
    # saving outputs
    #ndvi.update({str(gdf[attribute_col][0]): index_no_water})
    #chms_cov.update({str(gdf[attribute_col][0]): chm_cov})
    #chms_avg.update({str(gdf[attribute_col][0]): chm_avg})

    return index_no_water, chm_cov, chm_avg


def ndvi_minmax(waterhole_ndvi, waterhole_veg_dict):
    
    ndvi_max_values = pd.DataFrame(columns=['Vegetation', 'NDVI_max', 'NDVI_mean', 'NDVI_min', 'NDVI_std'])
    ndvi_min_values = pd.DataFrame(columns=['Vegetation', 'NDVI_max', 'NDVI_mean', 'NDVI_min', 'NDVI_std'])
    
    for vclass in waterhole_veg_dict.keys():
        
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
            
            # extract the pixels from a vegetation class
            vegetation_class = waterhole_ndvi.where(waterhole_veg_dict[vclass]>-1)
            
            # removes pixels with ndvi < 0
            #vegetation_class = vegetation_class.where(vegetation_class >=0)
            
            # calculated de max ndvi along the time dimension
            ndvi_max = vegetation_class.quantile(0.99, dim=['time'], skipna=True)
            ndvi_max = ndvi_max.where(ndvi_max >= 0)
            
            ndvi_max = ndvi_max.compute()
            
            vegetation_sample = waterhole_veg_dict[vclass].count()
            
            ndvi_sample_max = ndvi_max.NDVI.count()
            
            ndvi_max_max = ndvi_max.quantile(0.99, dim=['x', 'y'], skipna=True)
            ndvi_max_min = ndvi_max.quantile(0.01, dim=['x', 'y'], skipna=True)
            ndvi_max_mean = ndvi_max.mean(dim=['x', 'y'], skipna=True)
            ndvi_max_std = ndvi_max.std(dim=['x', 'y'], skipna=True)

            ndvi_min = vegetation_class.quantile(0.01, dim=['time'], skipna=True)
            ndvi_min = ndvi_min.where(ndvi_min >= 0)

            ndvi_min = ndvi_min.compute()
        
            ndvi_sample_min = ndvi_min.NDVI.count()
        
            ndvi_min_max = ndvi_min.quantile(0.99, dim=['x', 'y'], skipna=True)
            ndvi_min_min = ndvi_min.quantile(0.01, dim=['x', 'y'], skipna=True)
            ndvi_min_mean = ndvi_min.mean(dim=['x', 'y'], skipna=True)
            ndvi_min_std = ndvi_min.std(dim=['x', 'y'], skipna=True)
        
        
            ndvi_max_out = pd.DataFrame(data={'Vegetation':str(vclass), 
                                              'NDVI_max':[float(ndvi_max_max.NDVI)], 
                                              'NDVI_mean':[float(ndvi_max_mean.NDVI)], 
                                              'NDVI_min':[float(ndvi_max_min.NDVI)], 
                                              'NDVI_std': [float(ndvi_max_std.NDVI)],
                                              'VegSample': [int(vegetation_sample)],
                                              'NDVISample': [int(ndvi_sample_max)]})
        
            ndvi_max_values = pd.concat([ndvi_max_values, ndvi_max_out], axis=0)
    
    
            ndvi_min_out = pd.DataFrame(data={'Vegetation':str(vclass), 
                                              'NDVI_max':[float(ndvi_min_max.NDVI)], 
                                              'NDVI_mean':[float(ndvi_min_mean.NDVI)], 
                                              'NDVI_min':[float(ndvi_min_min.NDVI)], 
                                              'NDVI_std':[float(ndvi_min_std.NDVI)],
                                              'Sample': [int(vegetation_sample)],
                                              'NDVISample': [int(ndvi_sample_min)]})
        
            ndvi_min_values = pd.concat([ndvi_min_values, ndvi_min_out], axis=0)
    
    
    return ndvi_max_values, ndvi_min_values


def lidar_veg(lidar_cov, lidar_height):
        
    dense_trees = lidar_cov.where(lidar_cov>=0.80)
    dense_trees = dense_trees.where(lidar_height>=2.5)
            
    dense_shrubs = lidar_cov.where(lidar_cov>=0.80)
    dense_shrubs = dense_shrubs.where(lidar_height<2.5)
    dense_shrubs = dense_shrubs.where(lidar_height>0.5)
            
    mid_dense_trees = lidar_cov.where(lidar_cov<0.80)
    mid_dense_trees = mid_dense_trees.where(lidar_cov>=0.50)
    mid_dense_trees = mid_dense_trees.where(lidar_height>=2.5)
            
    mid_dense_shrubs = lidar_cov.where(lidar_cov<0.80)
    mid_dense_shrubs = mid_dense_shrubs.where(lidar_cov>=0.50)
    mid_dense_shrubs = mid_dense_shrubs.where(lidar_height<2.5)
    mid_dense_shrubs = mid_dense_shrubs.where(lidar_height>0.5)
            
    sparse_trees = lidar_cov.where(lidar_cov<0.50)
    sparse_trees = sparse_trees.where(lidar_cov>0.20)
    sparse_trees = sparse_trees.where(lidar_height>=2.5)
            
    sparse_shrubs = lidar_cov.where(lidar_cov<0.50)
    sparse_shrubs = sparse_shrubs.where(lidar_cov>0.20)
    sparse_shrubs = sparse_shrubs.where(lidar_height<2.5)
    sparse_shrubs = sparse_shrubs.where(lidar_height>0.5)
        
    water_body = lidar_cov.where(lidar_cov==np.nan)
    # water_body = water_body.where(lidar_height[key]==np.nan)
        
    isolated_trees = lidar_cov.where(lidar_cov<=0.20)
    isolated_trees = isolated_trees.where(lidar_cov>0.01)
    isolated_trees = isolated_trees.where(lidar_height>=2.5)
    
    isolated_shrubs = lidar_cov.where(lidar_cov<=0.20)
    isolated_shrubs = isolated_shrubs.where(lidar_cov>0.01)
    isolated_shrubs = isolated_shrubs.where(lidar_height<2.5)
    isolated_shrubs = isolated_shrubs.where(lidar_height>0.5)
        
    grasses_bare = lidar_cov.where(lidar_cov<=0.01)
    
    mask_dict = {'Dense trees':dense_trees, 'Mid-dense trees':mid_dense_trees, 'Sparse trees':sparse_trees, 
                 'Isolated trees':isolated_trees, 'Dense shrubs':dense_shrubs, 
                 'Mid-dense shrubs':mid_dense_shrubs, 'Sparse shrubs': sparse_shrubs, 
                 'Isolated shrubs':isolated_shrubs, 'Grasses and bare soil':grasses_bare}
    
    out_classes = lidar_cov
    out_classes = xr.where(dense_trees>=0, 1, out_classes)
    out_classes = xr.where(mid_dense_trees>=0, 2, out_classes)
    out_classes = xr.where(sparse_trees>=0, 3, out_classes)
    out_classes = xr.where(isolated_trees>=0, 4, out_classes)
        
    out_classes = xr.where(dense_shrubs>=0, 5, out_classes)
    out_classes = xr.where(mid_dense_shrubs>=0, 6, out_classes)
    out_classes = xr.where(sparse_shrubs>=0, 7, out_classes)
    out_classes = xr.where(isolated_shrubs>=0, 8, out_classes)

    out_classes = xr.where(grasses_bare>=0, 9, out_classes)
        
    return mask_dict, out_classes


# the next functions are basically just a repetition of the others. This is just to avoind changing the old jupyter notebook that extracts the phenology. 


def load_data (dc, query, waterhole_shape, perc_good=0):

    #Load waterhole shapefile and set col with the waterhole name
    gdf = gpd.read_file(waterhole_shape)
    
    # Extract the feature's geometry as a datacube geometry object
    geom = geometry.Geometry(geom=gdf.iloc[0].geometry, crs=gdf.crs)

    # Update the query to include our geopolygon
    query.update({'geopolygon': geom})
    
    # define native landsat crs
    native_ls = mostcommon_crs(dc, product=['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'], query=query)
    
    # Load landsat 3 collection
    ds = load_ard(dc=dc, products = ['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'], 
                  measurements = ['nbart_red', 'nbart_nir'],
                  output_crs =native_ls, resolution = (-30,30), align=(15,15), min_gooddata=0,
                  group_by = 'solar_day', **query, dask_chunks = {"time": 1}
                 )
    
    # use the function calculate_indices to calculate the NDVI using ds
    calculate_indices(ds, index='NDVI', drop=True, collection='ga_ls_3', inplace=True)
    
    
    # lod dea wo like landsat
    wofls = dc.load(product = 'ga_ls_wo_3', fuse_func = wofs_fuser,group_by = 'solar_day', 
                    like=ds, dask_chunks = {"time": 1})
    
    
    return ds, wofls, gdf


def prepare_data(ds, wofls, gdf, lidar_cov, lidar_height, perc_good=0):
    
    # Load LiDAR derived products
    chm_cov = rio_slurp_xarray(lidar_cov, gbox=ds.geobox, resampling='nearest')
    chm_avg = rio_slurp_xarray(lidar_height, gbox=ds.geobox, resampling='nearest')
    
    # Generate a polygon mask to keep only data within the polygon
    mask = xr_rasterize(gdf.iloc[[0]], ds)
    
    # use the polygon mask to remove everything that is not included in the 500 m buffer waterhole shapefile
    ds = ds.where(mask)
    chm_cov = chm_cov.where(mask)
    chm_avg = chm_avg.where(mask)
    
    # create a boolean raster to show the pixels that are included in the study area
    ds_notnull = chm_cov.notnull()
    
    # get the number of pixels that are included in the study area
    count_NDVIpixels = ds_notnull.where(ds_notnull==True).sum(dim=['x', 'y'])
    
    wo_mask = masking.make_mask(wofls.water, dry=True)
    index_no_water = ds.where(wo_mask)
    
    index_no_water = index_no_water.compute()
    
    wo_wet = masking.make_mask(wofls, wet=True)
    wo_dry = masking.make_mask(wofls, dry=True)
    wo_clear = wo_wet + wo_dry
    wo_clear = masking.make_mask(wofls, cloud_shadow=False, cloud=False, nodata=False)
    wo_masked = wo_wet.where(wo_clear).water
    wofl_freq = wo_masked.mean(dim=['time'])
    
    wofl_freq = wofl_freq.compute()
    
    # remove the pixels that are identified as water for 80% of the time
    chm_cov = chm_cov.where(wofl_freq <= 0.80)
    chm_avg = chm_avg.where(wofl_freq <= 0.80)
    
    # set the base waterhole area as the pixels identified as water for at least 95% of the time
    min_waterhole = wofl_freq.where(wofl_freq > 0.80).count(['x', 'y'])
    waterhole_perc = min_waterhole / count_NDVIpixels
    
    # boolean raster with pixels flaged as dry
    nowater_data = index_no_water.NDVI.notnull()
    # number of good pixels (no water and no clouds)
    count_goodNDVI = nowater_data.where(nowater_data==True).sum(dim=['x', 'y'])

    # percentage of good quality pixels
    percent_gooddata = count_goodNDVI / count_NDVIpixels
    
    # keep only the observations percentage of good pixels higher or equal to perc_good
    index_no_water = index_no_water.sel(time=percent_gooddata >= perc_good)
    
    # resample and interpolate the data
    #index_no_water = index_no_water.resample(time="16D").max('time')
    
    
    return index_no_water, chm_cov, chm_avg, min_waterhole


def landsat_fpar_classes(mask_dict, fpar):
    
    classes_fpar_dict = {}
            
    fpar = xr.where(fpar>1, 1, fpar)
    fpar_key = fpar
    total_veg_pixels = 0
    for vclass in mask_dict.keys():
                
        mask_count = mask_dict[vclass].count()
        if mask_count>=10:
    
            class_fpar = fpar_key.where(mask_dict[vclass]>-1)
            class_fpar = class_fpar.median(dim=['x','y'], skipna=True)
                    
            fpar_raw = class_fpar
                    
            months = class_fpar.time.dt.month
            years =  class_fpar.time.dt.year
                    
            class_out_fpar = pd.DataFrame(data={'Fpar':class_fpar, 'Fpar_raw':fpar_raw,
                                                'Time':class_fpar.time,'Year':years,
                                                'Sample':int(mask_count)}).assign(Vegetation=str(vclass))
                    
            # Append results to a dictionary
            classes_fpar_dict.update({str(vclass): class_out_fpar})
    
        else:
        
            class_out_fpar = pd.DataFrame(data={'Fpar':np.nan, 'Time':class_fpar.time, 'Year':years,
                                                'Sample': int(mask_count)}).assign(Vegetation=str(vclass))

            # Append results to a dictionary
            classes_fpar_dict.update({str(vclass): class_out_fpar})
        
    return classes_fpar_dict  


def cov_height(ds, wofls, gdf, lidar_cov, lidar_height, perc_good=0):
    
    # Load LiDAR derived products
    chm_cov = rio_slurp_xarray(lidar_cov, gbox=ds.geobox, resampling='nearest')
    chm_avg = rio_slurp_xarray(lidar_height, gbox=ds.geobox, resampling='nearest')


################# ALTERNATIVE VERSION OF THE FUNCTIONS THAT DO NOT USE LIDAR COVER AND HEIGHT RASTER #####################

def load_ndvi2 (dc, query, waterhole_shape):
    
    #Load waterhole shapefile and set col with the waterhole name
    gdf = gpd.read_file(waterhole_shape)
    
    # Extract the feature's geometry as a datacube geometry object
    geom = geometry.Geometry(geom=gdf.iloc[0].geometry, crs=gdf.crs)

    # Update the query to include our geopolygon
    query.update({'geopolygon': geom})
    
    # define native landsat crs
    native_ls = mostcommon_crs(dc, product=['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'], query=query)
    
    # Load landsat 3 collection
    ds = load_ard(dc=dc, products = ['ga_ls5t_ard_3', 'ga_ls7e_ard_3', 'ga_ls8c_ard_3'], 
                  measurements = ['nbart_red', 'nbart_nir'],
                  output_crs =native_ls, resolution = (-30,30), align=(15,15), min_gooddata=0,
                  group_by = 'solar_day', **query, dask_chunks = {"time": 1}
                 )
    
    # Generate a polygon mask to keep only data within the polygon
    mask = xr_rasterize(gdf.iloc[[0]], ds)
    
    # use the function calculate_indices to calculate the NDVI using ds
    calculate_indices(ds, index='NDVI', drop=True, collection='ga_ls_3', inplace=True)
    
    # use the polygon mask to remove everything that is not included in the 500 m buffer waterhole shapefile
    ds = ds.where(mask)
    ds = masking.mask_invalid_data(ds)
    
    # define native landsat crs
    native_wofls = mostcommon_crs(dc, product=['ga_ls_wo_3'], query=query)
    
    # lod dea wo like landsat
    wofls = dc.load(product = 'ga_ls_wo_3', fuse_func = wofs_fuser, 
                    group_by = 'solar_day', like=ds, dask_chunks = {"time": 1})
    
    wo_mask = masking.make_mask(wofls.water, dry=True)
    index_no_water = ds.where(wo_mask)
    
    index_no_water = index_no_water.compute()
    #print(index_no_water)
    
    wo_wet = masking.make_mask(wofls, wet=True)
    wo_dry = masking.make_mask(wofls, dry=True)
    wo_clear = wo_wet + wo_dry
    wo_clear = masking.make_mask(wofls, cloud_shadow=False, cloud=False, nodata=False)
    wo_masked = wo_wet.where(wo_clear).water
    wofl_freq = wo_masked.mean(dim=['time'])
    
    wofl_freq = wofl_freq.compute()
    
    index_no_water = index_no_water.where(wofl_freq <= 0.80)
    
    # resample and interpolate the data
    index_no_water = index_no_water.resample(time="16D").max('time')

    return index_no_water


def ndvi_minmax2(waterhole_ndvi):
    
    ndvi_max_values = pd.DataFrame(columns=['NDVI_max', 'NDVI_mean', 'NDVI_min', 'NDVI_std'])
    ndvi_min_values = pd.DataFrame(columns=['NDVI_max', 'NDVI_mean', 'NDVI_min', 'NDVI_std'])
        
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
            
            
        # calculated de max ndvi along the time dimension
        ndvi_max = waterhole_ndvi.quantile(0.99, dim=['time'], skipna=True)
        ndvi_max = ndvi_max.where(ndvi_max >= 0)
            
        ndvi_max = ndvi_max.compute()
              
            
        ndvi_max_max = ndvi_max.quantile(0.99, dim=['x', 'y'], skipna=True)
        ndvi_max_min = ndvi_max.quantile(0.01, dim=['x', 'y'], skipna=True)
        ndvi_max_mean = ndvi_max.mean(dim=['x', 'y'], skipna=True)
        ndvi_max_std = ndvi_max.std(dim=['x', 'y'], skipna=True)

        ndvi_min = waterhole_ndvi.quantile(0.01, dim=['time'], skipna=True)
        ndvi_min = ndvi_min.where(ndvi_min >= 0)

        ndvi_min = ndvi_min.compute()
        
        
        ndvi_min_max = ndvi_min.quantile(0.99, dim=['x', 'y'], skipna=True)
        ndvi_min_min = ndvi_min.quantile(0.01, dim=['x', 'y'], skipna=True)
        ndvi_min_mean = ndvi_min.mean(dim=['x', 'y'], skipna=True)
        ndvi_min_std = ndvi_min.std(dim=['x', 'y'], skipna=True)
        
        
        ndvi_max_out = pd.DataFrame(data={
                                          'NDVI_max':[float(ndvi_max_max.NDVI)], 
                                          'NDVI_mean':[float(ndvi_max_mean.NDVI)], 
                                          'NDVI_min':[float(ndvi_max_min.NDVI)], 
                                          'NDVI_std': [float(ndvi_max_std.NDVI)]
                                          })
        
        ndvi_max_values = pd.concat([ndvi_max_values, ndvi_max_out], axis=0)
    
    
        ndvi_min_out = pd.DataFrame(data={
                                          'NDVI_max':[float(ndvi_min_max.NDVI)], 
                                          'NDVI_mean':[float(ndvi_min_mean.NDVI)], 
                                          'NDVI_min':[float(ndvi_min_min.NDVI)], 
                                          'NDVI_std':[float(ndvi_min_std.NDVI)],
                                          })
        
        ndvi_min_values = pd.concat([ndvi_min_values, ndvi_min_out], axis=0)
    
    
    return ndvi_max_values, ndvi_min_values



def prepare_data2(ds, gdf, lidar_cov, perc_good=0):

    chm_cov = rio_slurp_xarray(lidar_cov, gbox=ds.geobox, resampling='nearest')
    
    # Generate a polygon mask to keep only data within the polygon
    mask = xr_rasterize(gdf.iloc[[0]], ds)
    
    # use the polygon mask to remove everything that is not included in the 500 m buffer waterhole shapefile
    index_no_water = ds.where(mask)
    lidar_cov = chm_cov.where(mask)
    
    index_no_water = index_no_water.compute()
    
    # resample and interpolate the data
    #index_no_water = index_no_water.resample(time="16D").max('time')
    
    
    return index_no_water, lidar_cov
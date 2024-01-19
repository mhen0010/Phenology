import sys
import dask
import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import datetime



def PoS(pixel_ts, peak_threshold):
    
    # mean of the time series to define a height threshold
    #h = float(pixel_ts.mean(['time']))
    h = peak_threshold
    # find the peaks that will constitute the pos
    pixel_peaks = find_peaks(pixel_ts, height = h, prominence=0.04, distance = 8, width=2)
    
    # pos values
    pos_v = pixel_peaks[1]['peak_heights']
    
    # pos dates
    pos = pixel_ts.time[pixel_peaks[0]]
    
    # pos day of year
    pos_doy = pos.time.dt.dayofyear
    pos_year = pos.time.dt.year
    
    # save results in a pandas dataframe
    pos_stats = pd.DataFrame(data={'POS_year':pos_year, 'POS_dt':pos, 'POS_doy':pos_doy, 'POS_v':pos_v})

    return pos_stats


def SoS(pixel_ts, pos_stats, threshold=0.2):
    
    # steps to define SOS
    # templete of the dataframe where results will be concatenated
    sos_stats = pd.DataFrame(columns=['SOS_dt','SOS_doy'])
    
    # iterate through the pos to define the sos for each peak
    for doy in range(len(pos_stats)):
        
        # slice the time series to inclue the observations before the pos
        #slice_end = pos_stats.POS_dt[doy]
        #slice_start = slice_end-datetime.timedelta(150)
        
        # slice the time series to inclue the observations before the pos
        slice_end = pos_stats.POS_dt[doy]
        
        if doy<1:
            slice_start = slice_end-datetime.timedelta(250)
        else:
            slice_start = pos_stats.POS_dt[doy-1]
            duration = slice_end - slice_start
            if duration >=datetime.timedelta(300):
                slice_start = slice_end-datetime.timedelta(258)
        
        # keep only the observations before the pos
        greenup = pixel_ts.sel(time=slice(slice_start, slice_end))
        # keep only the observations between the pos and local minima located before the pos
        greenup = greenup.where(greenup.time >= greenup.isel(time=greenup.argmin("time")).time).dropna(dim='time')
        
        # calculate the greenup threshold based on the max and min values 
        thrs = (greenup - greenup.min("time"))/(greenup.max("time") - greenup.min("time"))
        
        # keep only the observations that equal or greater thant the threshold parameter
        green_thrs = thrs.where(thrs>=threshold)
        
        # check if green_thrs is all nan 
        mask = green_thrs.isnull().all("time")
        
        green_thrs = green_thrs.dropna(dim='time')
        
        if mask == False:
            # sos value is selected by using the fPAR value of the minimum thrs
            sos_v = float(greenup.sel(time=green_thrs.isel(time=green_thrs.argmin("time")).time))
            #sos_v = float(greenup.sel(time=green_thrs.isel(time=0).time))
            
            # isolate the year of the sos
            sos_year = int(greenup.sel(time=green_thrs.isel(time=green_thrs.argmin("time")).time).time.dt.year)
            #sos_year = int(green_thrs.isel(time=0).time.dt.year) 
            
            # isolate the month of the sos
            sos_month = int(greenup.sel(time=green_thrs.isel(time=green_thrs.argmin("time")).time).time.dt.month)
            #sos_month = int(green_thrs.isel(time=0).time.dt.month)
            
            # isolate the day of the sos
            sos_day = int(greenup.sel(time=green_thrs.isel(time=green_thrs.argmin("time")).time).time.dt.day)
            #sos_day = int(green_thrs.isel(time=0).time.dt.day)
        
            # create a dataframe with the sos date in the datetime format
            sos_date = pd.DataFrame({'year':[sos_year], 'month':[sos_month], 'day':[sos_day]})
            sos = pd.to_datetime(sos_date)
       
            # sos day of year
            sos_doy = green_thrs.isel(time=green_thrs.argmin("time")).time.dt.dayofyear
            #sos_doy = green_thrs.isel(time=0).time.dt.dayofyear
        
            # save results in a pandas dataframe
            row_sos = pd.DataFrame(data={'SOS_dt':sos, 'SOS_doy':int(sos_doy), 'SOS_v':sos_v})
            sos_stats = pd.concat([sos_stats, row_sos])                           
    
    
    # reset index and drop redundant column 
    sos_stats = sos_stats.reset_index().drop(['index'], axis=1)
    
    return sos_stats


def EoS(pixel_ts, pos_stats, threshold=0.2):
    
    # steps to define EOS
    # templete of the dataframe where results will be concatenated
    eos_stats = pd.DataFrame(columns=['EOS_dt','EOS_doy'])
    
    # iterate through the pos to define the sos for each peak (last to first)
    for doy in range(len(pos_stats)-1,-1,-1):
        
        # slice the time series to inclue the observations after the pos
        slice_start = pos_stats.POS_dt[doy]
        #slice_end = slice_start + datetime.timedelta(150)
        
        if doy == (len(pos_stats)-1):
            slice_end = slice_start + datetime.timedelta(250)

        else:
            slice_end = pos_stats.POS_dt[doy+1]
            duration = slice_end - slice_start
            if duration >=datetime.timedelta(300):
                slice_end = slice_start + datetime.timedelta(258)
        
        # keep only the observations after the pos
        senesce = pixel_ts.sel(time=slice(slice_start, slice_end))
        # keep only the observations between the pos and local minima located after the pos
        senesce = senesce.where(senesce.time <= senesce.isel(time=senesce.argmin("time")).time).dropna(dim='time')

        # calculate the senescence threshold based on the max and min values
        thrs = (senesce - senesce.min("time"))/(senesce.max("time") - senesce.min("time"))
        
        # keep only the observations that equal or greater thant the threshold parameter
        senesce_thrs = thrs.where(thrs>=threshold)
        
        # check if senesce_thrs is all nan
        mask = senesce_thrs.isnull().all("time")
        
        senesce_thrs = senesce_thrs.dropna(dim='time')
        
        if mask == False:
            # eos value is selected by using the fPAR value of the minimum thrs
            eos_v = float(senesce.sel(time=senesce_thrs.isel(time=senesce_thrs.argmin("time")).time))
            #eos_v = float(senesce.sel(time=senesce_thrs.isel(time=len(senesce_thrs)-1).time))
        
            # isolate the year of the eos
            eos_year = int(senesce.sel(time=senesce_thrs.isel(time=senesce_thrs.argmin("time")).time).time.dt.year)
            #eos_year = int(senesce_thrs.isel(time=len(senesce_thrs)-1).time.dt.year)
            
            # isolate the month of the eos
            eos_month = int(senesce.sel(time=senesce_thrs.isel(time=senesce_thrs.argmin("time")).time).time.dt.month)
            #eos_month = int(senesce_thrs.isel(time=len(senesce_thrs)-1).time.dt.month)
            
            # isolate the day of the eos
            eos_day = int(senesce.sel(time=senesce_thrs.isel(time=senesce_thrs.argmin("time")).time).time.dt.day)
            #eos_day = int(senesce_thrs.isel(time=len(senesce_thrs)-1).time.dt.day)
        
            # create a dataframe with the eos date in the datetime format
            eos_date = pd.DataFrame({'year':[eos_year], 'month':[eos_month], 'day':[eos_day]})
            eos = pd.to_datetime(eos_date)
        
            # eos day of year
            eos_doy = senesce_thrs.isel(time=senesce_thrs.argmin("time")).time.dt.dayofyear
            #eos_doy = senesce_thrs.isel(time=len(senesce_thrs)-1).time.dt.dayofyear
        
            # save results in a pandas dataframe
            row_eos = pd.DataFrame(data={'EOS_dt':eos, 'EOS_doy':int(eos_doy), 'EOS_v':eos_v})
            eos_stats = pd.concat([eos_stats, row_eos])      
        
        
    # invert the order of the dataframe, reset index and drop redundant column 
    eos_stats = eos_stats.iloc[::-1].reset_index().drop(['index'], axis=1)
    
    return eos_stats


def pheno_interpolation(pixel_ts, method='spline'):
    
    pixel_ts = pixel_ts.chunk({'time': -1})
    pixel_interp = pixel_ts.interpolate_na(dim='time', method=method)
    
    return pixel_interp


def smooth_savgol(ds, window_length, polyorder):
    
    def smoother(da, window_length, polyorder):
        return da.apply(savgol_filter, window_length=window_length, polyorder=polyorder, axis=0)

    # create kwargs dict
    kwargs = {'window_length': window_length, 'polyorder': polyorder}
    
    temp = xr.full_like(ds, fill_value=np.nan)
    ds = xr.map_blocks(smoother, ds, template=temp, kwargs=kwargs)
    
    return ds


def calc_fpar(ds, min_ndvi=0.1, max_ndvi=0.71, reference=True):
    
    # list of bands to drop after fPAR is calculated => fPAR is the only one needed
    bands_to_drop = list(ds.data_vars)
    
    # variables to calculate fPAR using NDVI accorind to Roderick (1999) and Donohue (2008)
    max_fpar = 0.95
    min_fpar = 0
    
    if reference == True:
        
        min_ndvi = min_ndvi
        max_ndvi = max_ndvi
        
    else:
    # set the min and max ndvi using quantiles 
        max_ndvi = ds.NDVI.max('time')
        max_ndvi = max_ndvi.quantile(0.95, dim=['x','y'], skipna = True)
        
        min_ndvi = ds.NDVI.min('time')
        min_ndvi = min_ndvi.quantile(0.10, dim=['x','y'], skipna = True)
    
    # set the limits of the max and min ndvi
    ndvi = ds.NDVI
    ndvi = xr.where(ndvi<min_ndvi, min_ndvi, ndvi)
    ndvi = xr.where(ndvi>max_ndvi, max_ndvi, ndvi)
    
    
    # calculate total fPAR
    Fpar = (((max_fpar - min_fpar)*(ndvi - min_ndvi))/(max_ndvi - min_ndvi)) + min_fpar
    Fpar = xr.where(Fpar<min_fpar, min_fpar, Fpar)
    
    ds['fPAR'] = Fpar
    ds = ds.drop(bands_to_drop)
    
    
    return ds


def decomp_fpar(Fpar):
    
    # decomposing fPAR into the preliminary persistent fraction according to Donohue (2008)
    Fp1_fpar = Fpar['fPAR'].rolling(time=15, min_periods=1, center=True).min()
    
    Fp2_fpar = Fp1_fpar.rolling(time=19, min_periods=1, center=True).mean()
    
    # decomposing fPAR into the preliminary recurrent fraction according to Donohue (2008)
    Fr1_fpar = Fpar['fPAR'] - Fp2_fpar
    
    # Final persistent fraction of fPAR
    Fp2_fpar = xr.where(Fr1_fpar < 0, Fp2_fpar - abs(Fr1_fpar), Fp2_fpar)
    Fpar['fPAR_P'] = Fp2_fpar
    
    # Final recurrent fraction of fPAR
    Fr_fpar = Fpar['fPAR'] - Fp2_fpar
    Fpar['fPAR_R'] = Fr_fpar
    
    return Fpar


def filter_dropout_pre(fPAR_ds):
    
    long_mean = fPAR_ds.mean()
    long_std = fPAR_ds.std()
    
    for obs in range(len(fPAR_ds)-2):
        
        if fPAR_ds.isel(time=obs) > (long_mean+(3*long_std)):
            
            #fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = long_mean+(3*long_std)
            fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = np.nan
            
        if fPAR_ds.isel(time=obs) < (long_mean-(1.5*long_std)):
        
            #fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = long_mean-(1.5*long_std)
            fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = np.nan
        
    return fPAR_ds

def filter_dropout_pos(fPAR_ds):
    
    long_mean = fPAR_ds.mean()
    long_std = fPAR_ds.std()
    
    for obs in range(len(fPAR_ds)-2):
        
        if fPAR_ds.isel(time=obs) > (long_mean+(3*long_std)):
            
            fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = long_mean+(3*long_std)
        
        if fPAR_ds.isel(time=obs) < (long_mean-(1.5*long_std)):
        
            fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = long_mean-(1.5*long_std)
            
        #print(fPAR_ds.time.isel(time=obs))
        if obs > 1:
            obs_m2 = fPAR_ds.isel(time=obs-2)
            obs_m1 = fPAR_ds.isel(time=obs-1)
            obs_ = fPAR_ds.isel(time=obs)
            obs_p1 = fPAR_ds.isel(time=obs+1)
            obs_p2 = fPAR_ds.isel(time=obs+2)
            
            sub = obs_m1 - obs_
            if sub > 0.06:
                fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = (obs_m2+obs_m1 + obs_p1+obs_p2)/4
                #fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = (obs_m1 + obs_p1)/2
    
    return fPAR_ds

#def filter_dropout(fPAR_ds):
#    
#    long_mean = fPAR_ds.mean()
#    long_std = fPAR_ds.std()
#    
#    for obs in range(len(fPAR_ds)-2):
#        
#        if fPAR_ds.isel(time=obs) > (long_mean+(3*long_std)):
#            
#            fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = long_mean+(3*long_std)
#        
#        if fPAR_ds.isel(time=obs) < (long_mean-(1*long_std)):
#        
#            fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = long_mean-(1*long_std)
#        
#        if obs > 1:
#            obs_m2 = fPAR_ds.isel(time=obs-2)
#            obs_m1 = fPAR_ds.isel(time=obs-1)
#            obs_ = fPAR_ds.isel(time=obs)
#            obs_p1 = fPAR_ds.isel(time=obs+1)
#            obs_p2 = fPAR_ds.isel(time=obs+2)
#            
#            sub = obs_m1 - obs_
#            if sub > 0.07:
#                
#                fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = (obs_m2+obs_m1 + obs_p1+obs_p2)/4
#                #fPAR_ds.loc[fPAR_ds.time.isel(time=obs)] = (obs_m1 + obs_p1)/2
#    
#    return fPAR_ds


def phenopix(fPAR_ds, out_classes, veg_class='Dense trees', threshold=0.2, comp='fPAR'):
    
    # isolate the LiDAR vegetation class
    class_ds = fPAR_ds.where(out_classes[veg_class]>-1)
    
    threshold_ts = class_ds['fPAR'].mean(dim=['x', 'y'])
    
    threshold_ts = filter_dropout_pre(threshold_ts)
    threshold_ts = threshold_ts.interpolate_na(dim='time', method='cubic')
    threshold_ts = filter_dropout_pos(threshold_ts)
    threshold_savgol = savgol_filter(threshold_ts, window_length=11, polyorder=3)
    threshold_smooth = xr.Dataset(data_vars=dict(fPAR= (["time"],threshold_savgol)), coords=dict(time=threshold_ts.time))
    
    threshold_smooth = decomp_fpar(threshold_smooth)
    
    class_fillers = threshold_smooth['fPAR']
    
    # calculate min peak value criteria
    peak_threshold = float(threshold_smooth[comp].mean('time'))
    
    peak_threshold_t = float(threshold_smooth['fPAR'].mean('time'))
    peak_threshold_r = float(threshold_smooth['fPAR_R'].mean('time'))
    peak_threshold_p = float(threshold_smooth['fPAR_P'].mean('time'))
    
    # isolate the coordinates of the pixels that belong veg_class
    class_coord = out_classes[veg_class].where(out_classes[veg_class]>-1)
    class_coord = class_coord.to_series().reset_index().dropna().reset_index()
    
    # dataframe so save alll the results
    pheno_df = pd.DataFrame(columns=['SOS_dt','SOS_doy','SOS_v','POS_dt','POS_doy','POS_v','EOS_dt','EOS_doy','EOS_v','x','y'])
    pheno_df_r = pd.DataFrame(columns=['SOS_dt','SOS_doy','SOS_v','POS_dt','POS_doy','POS_v','EOS_dt','EOS_doy','EOS_v','x','y'])
    pheno_df_p = pd.DataFrame(columns=['SOS_dt','SOS_doy','SOS_v','POS_dt','POS_doy','POS_v','EOS_dt','EOS_doy','EOS_v','x','y'])
    
    # iterate through all the pixels that belong to vegetation class using the coordinates
    for coords in range(len(class_coord)):
        
        # isolate the time series of a pixel
        pixel_ts = class_ds.sel(x=class_coord.x[coords], y=class_coord.y[coords])
        pixel_ts = pixel_ts['fPAR']
        
        total_ts = len(pixel_ts.time)
        good_ts = pixel_ts.notnull().sum()
        
        perc_good_ts = float(good_ts/total_ts)
        
        if perc_good_ts >=0.7:
            
            pixel_interp = filter_dropout_pre(pixel_ts)
            
            # interpolate time series
            pixel_interp = pixel_ts.interpolate_na(dim='time', method='cubic', max_gap=datetime.timedelta(64), fill_value=(0,0.95), bounds_error=False)
            pixel_interp = xr.where(pixel_interp.notnull()==True, pixel_interp, class_fillers)
            pixel_interp = pixel_interp.interpolate_na(dim='time', method='cubic', max_gap=datetime.timedelta(64), fill_value=(0,0.95), bounds_error=False)
            
            pixel_interp = xr.where(pixel_interp<=0, class_fillers, pixel_interp)
            pixel_interp = xr.where(pixel_interp>0.95, class_fillers, pixel_interp)
            
            # remove dropouts
            pixel_interp = filter_dropout_pos(pixel_interp)
            
            # smooth the time series using the savinsky golay filter
            pixel_savgol = savgol_filter(pixel_interp, window_length=11, polyorder=3)
        
            # create new dataset with the smooth time series
            pixel_smooth = xr.Dataset(data_vars=dict(fPAR= (["time"],pixel_savgol)), coords=dict(time=pixel_interp.time))
            
            # decompose the smooth fPAR time series
            pixel_smooth = decomp_fpar(pixel_smooth)
            
            # select the desired component
            #pixel_smooth = pixel_smooth[comp]
            
            pixel_t = pixel_smooth['fPAR']
            pixel_r = pixel_smooth['fPAR_R']
            pixel_p = pixel_smooth['fPAR_P']
            
            # retrieve land surface phenology metrics
            
            # get the peak of season
            pos_ = PoS(pixel_t, peak_threshold_t)
            # get the start of season
            sos_ = SoS(pixel_t, pos_, threshold)
            # get the end of season
            eos_ = EoS(pixel_t, pos_, threshold)
            
            pos_r = PoS(pixel_r, peak_threshold_r)
            # get the start of season
            sos_r = SoS(pixel_r, pos_r, threshold)
            # get the end of season
            eos_r = EoS(pixel_r, pos_r, threshold)
            
            pos_p = PoS(pixel_p, peak_threshold_p)
            # get the start of season
            sos_p = SoS(pixel_p, pos_p, threshold)
            # get the end of season
            eos_p = EoS(pixel_p, pos_p, threshold)
            
            
            # concat the results in one single dataframe
            pheno_ = pd.concat([ sos_, pos_, eos_], axis=1).assign(x=class_coord.x[coords], y=class_coord.y[coords], Vegetation = veg_class)
            pheno_df = pd.concat([pheno_df, pheno_])
    
            # concat the results in one single dataframe
            pheno_r = pd.concat([ sos_r, pos_r, eos_r], axis=1).assign(x=class_coord.x[coords], y=class_coord.y[coords], Vegetation = veg_class)
            pheno_df_r = pd.concat([pheno_df_r, pheno_r])
            
            # concat the results in one single dataframe
            pheno_p = pd.concat([ sos_p, pos_p, eos_p], axis=1).assign(x=class_coord.x[coords], y=class_coord.y[coords], Vegetation = veg_class)
            pheno_df_p = pd.concat([pheno_df_p, pheno_p])
    
    # calculate the lengh of season
    #pheno_df['LOS'] = pheno_df.EOS_doy - pheno_df.SOS_doy
    #pheno_df['LOS'] = pheno_df['LOS'].where(pheno_df['LOS']>0, pheno_df['LOS']+365)
    if pheno_df.empty == False:
        
        pheno_df['LOS'] = (pheno_df.EOS_dt - pheno_df.SOS_dt).dt.days
    
    
    if pheno_df_r.empty == False:
        
        pheno_df_r['LOS'] = (pheno_df_r.EOS_dt - pheno_df_r.SOS_dt).dt.days
    
    
    if pheno_df_p.empty == False:
        
        pheno_df_p['LOS'] = (pheno_df_p.EOS_dt - pheno_df_p.SOS_dt).dt.days
    
    # reset index and drop redundant column 
    pheno_df = pheno_df.reset_index().drop(['index'], axis=1)
    
    # reset index and drop redundant column 
    pheno_df_r = pheno_df_r.reset_index().drop(['index'], axis=1)
    
    # reset index and drop redundant column 
    pheno_df_p = pheno_df_p.reset_index().drop(['index'], axis=1)
    
    return pheno_df, pheno_df_r, pheno_df_p
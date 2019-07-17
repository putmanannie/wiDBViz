# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 17:00:19 2019
V3 switches to cartopy from basemap since basemap is buggy
@author: u0929173
"""

import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy as ctp
import datetime as dt
import pandas as pd
from shapely.geometry import Point
import os
from copy import copy

directory = "G:\My Drive\Writing\DatabasePaper\PaperCode\"
picklefolder = "c:pickle"
picklename = 'DatabaseFigSource.pkl'
figfolder = r"c:Figs"

os.chdir(picklefolder)
df = pd.read_pickle(picklename)
#%% Figure 2 - Number of samples of any type in each country

needcounts = True
#higher resolution means more marginal points are captured
shp = ctp.io.shapereader.natural_earth(resolution = '50m', category = 'cultural', name = 'admin_0_countries')
reader = ctp.io.shapereader.Reader(shp)
countrylist = reader.records()

#get the number of samples in each country using a shapely point/polygon compare
sites, indexes, counts = np.unique(df['Site_ID'].values, return_index = True, return_counts=True)
oceanmask = df['Type'].values[indexes] != 'Ocean'
sites = sites[oceanmask]
indexes= indexes[oceanmask]
counts1 = counts[oceanmask]

points = zip(df['Longitude'].values[indexes], df['Latitude'].values[indexes])

nsamplelist = []
print('Counting samples in countries:\n')
for country in countrylist:
    nsamples = 0
    print(country.attributes['NAME'])
    for (point, count) in zip(points, counts1):
        inside = country.geometry.contains(Point(point))
        if inside:
            nsamples = nsamples+count
    nsamplelist.append(nsamples)

#set the patch colors based on the number of sites in a country
cbins = np.array([0, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000])
cm = mpl.pyplot.get_cmap('GnBu')
scheme = [cm(float(i)/(len(cbins))) for i in range(len(cbins))]
#re-initialize the counter    
countrylist = reader.records()    
fig = mpl.pyplot.figure(figsize = (7, 5.5))
ax = mpl.pyplot.axes(projection = ccrs.Robinson(central_longitude = 0, globe = None))

print('Plotting countries:')
for (nsamples, country) in zip(nsamplelist, countrylist):
    print(nsamples, country.attributes['NAME'])
    if nsamples > 0:          
        for j in range(len(cbins)-1):
            if (nsamples > cbins[j]) & (nsamples <= cbins[(j+1)]):
                color = scheme[j+1]
    else:
        color = '#dddddd'
            
    ax.add_geometries(country.geometry, ccrs.PlateCarree(), edgecolor = 'dimgray', 
                      facecolor = color, linewidth = 0.1, zorder = 2)
#ax.coastlines()        
ax_legend = fig.add_axes([0.35, 0.14, 0.3, 0.03], zorder=3)
cmap = mpl.colors.ListedColormap(scheme)
cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=range(len(cbins)), boundaries=range(len(cbins)), orientation='horizontal')
cb.ax.set_xticklabels([str(int(i)) for i in cbins], rotation = -45)
cb.set_label('Number of samples')
 
mpl.pyplot.savefig(os.path.join(figfolder, 'Fig2.png'), dpi = 500, bbox_inches='tight', pad_inches=.2)
#%% Figure 3 - the temporal coverage of samples grouped by type

scheme = ["#69BFBF", "#F2776D", "#085DA0", "#C7A89F"]

typedict = {'Spring, \n Cave Drip, Ground': np.array(['Cave_drip','Spring', 'Ground']), 
            'Precipitation \n Cloud, Fog ': np.array(['Cloud_or_fog', 'Precipitation']), 
            'Canal, River, Stream, \n Lake': np.array(['Canal', 'River_or_stream', 'Lake']),
            'Tap, Bottled': np.array(['Bottled', 'Tap']), 
            'Firn_core, Snow pit, \n Ice Core': np.array(['Firn_core', 'Snow_pit', 'Ice_core']), 
            'Ocean': np.array(['Ocean']), 
            'Soil, Stem, \n Mine, Sprinkler': 
                np.array(['Mine','Soil', 'Stem', 'Sprinkler'])}
plotorder = np.array([1, 4, 6, 3, 0, 5, 2])
naxes = len(typedict.keys())
fig, axes = mpl.pyplot.subplots(figsize = (7, 5.5), nrows = naxes, ncols = 1, sharex= True)
yearbinctrs = np.arange(1959, 2021, 1)
yearbinedges = ((yearbinctrs[1:]-yearbinctrs[:-1])/2.0)+yearbinctrs[:-1]
yearbinctrs = yearbinctrs[1:]
#remove any members of df 
test = [(item is not None) and (dt.datetime.strptime(str(item), '%Y-%m-%d %H:%M:%S') > dt.datetime(1900, 1, 1, 0, 0)) for item in df['Collection_Date'].values]
df_sub = df[test].reset_index()

for plorder, ax in zip(plotorder, axes):
    i = typedict.keys()[plorder]
    #this should make progressively larger violins, which should
    #be laid below one another, so that the largest violin is on the bottom
    histbins = np.zeros(len(yearbinctrs)-1)
    fighandles = []
    for j, k in zip(typedict[i], range(len(typedict[i]), 0, -1)):
        inds = np.where((df_sub['Type'].values == j)& (df_sub['Collection_Date'].values>pd.datetime(1960,1,1,0,0)))[0]
        sampleyr = df_sub.loc[inds, 'Collection_Date'].dt.year
        sampleyr = sampleyr[~np.isnan(sampleyr)]
        samplehist, binedges = np.histogram(sampleyr, bins = yearbinedges, density = False)
        histbins = np.add(histbins, samplehist)
        print(i, j, np.nanmax(histbins))
        p1 = ax.fill_between(yearbinctrs[:-1], np.ma.array(histbins/2, mask = histbins == 0), 
                        np.ma.array(-histbins/2, mask = histbins == 0), 
                        facecolor = scheme[(k-1)], edgecolor = scheme[(k-1)], linewidth = 0.2, 
                        interpolate = True, zorder = (k-1))
        fighandles.append(p1)
        
    ax.get_yaxis().set_ticks([])
    ax.set_xticks([])  
    ax.tick_params(length = 0)     
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none') 
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    #ax.set_ylabel(i, rotation = 'horizontal', labelpad = 10)  
    ax.legend(fighandles, list(typedict[i]), loc = 'center left', 
        bbox_to_anchor = [-0.1, 1] , frameon = False, ncol = 2, fontsize = 8)
    ymin, ymax = ax.get_ylim()
    if ymax > 1000:
        newylims = round(ymax/100, 0)*100
    else:
        newylims = round(ymax/10, 0)*10
    ax.set_ylim([-newylims, newylims])
    ax.annotate("+/-{}".format(int(newylims)), xy = (yearbinctrs[-1], 0), xytext = (yearbinctrs[-1]+2, 0), xycoords = 'data')

ymin, ymax = axes[0].get_ylim()    
axes[0].annotate('Y scale\n limits', xy = (yearbinctrs[-1], ymax), xytext= (yearbinctrs[-1]+2.5, ymax*1.10), xycoords = 'data')
ax.spines['bottom'].set_color('k')
xtickinds = np.where(yearbinctrs%10 == 0)[0]
ax.set_xticks(yearbinctrs[xtickinds])
ax.set_xticklabels((yearbinctrs[xtickinds]-1970).astype('datetime64[Y]'), rotation = -45)
os.chdir(figfolder)  
mpl.pyplot.savefig(os.path.join(figfolder, 'Fig3.png'), dpi = 500)
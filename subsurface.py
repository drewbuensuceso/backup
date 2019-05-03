import mysql.connector as mc
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
#from IPython.display import display
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad
import filterdata as filt
import analysis.querydb as qdb
import matplotlib
import matplotlib.pyplot as plt

import rtwindow as rtw
from pyfinance.ols import PandasRollingOLS
from datetime import datetime, date, time, timedelta
pd.set_option('display.max_columns', 12)
matplotlib.use('Qt5Agg')

def get_raw_accel_data(tsm_name, node_id, type_num):
    
    mydb = mc.connect(
        host="192.168.150.75",
        user="contributor_lvl2",
        passwd="additionaldelete2018",
        database="senslopedb"
        )
    
    query = "SELECT ts, node_id, type_num, xval, yval, zval from senslopedb.tilt_%s WHERE ts >= '2016-06-05 00:00:00'" %(tsm_name)
    query += "and ts <= '2019-01-01 00:00:00' AND node_id = %s AND type_num = %s" %(node_id, type_num)

    tilt_df = pd.read_sql(query, mydb)
    tilt_df = tilt_df.drop_duplicates(['ts']).drop(columns=['type_num'])
    tilt_df.columns = ['ts', 'node_id', 'x', 'y', 'z']
    tilt_df = filt.apply_filters(tilt_df, orthof=True)
    
    mydb.close()
    
    return(tilt_df)
    
def magnitude(df):
    df['magnitude'] = np.round((np.sqrt((df['x'] - df.iloc[0].x) ** 2 + (df['y'] - df.iloc[0].y) ** 2 + (df['z'] - df.iloc[0].z) ** 2) * seg_len / 1024), 4)
    
    return df

def theta_yz(df):
    df['theta_yz'] = np.arccos(df['y']/(np.sqrt(df['y']** 2 + df['z']** 2)))
#    df['theta_yz'] = np.arctan(df['z']/df['y'])
    
    return(df)

def signed_magnitude(df):
    df['signed_magnitude'] = np.where((df['theta_yz'] > df['theta_yz'].shift(1)), df['magnitude'], df['magnitude'] * -1)
    
    return(df)
    
def beta_filter(df):
    df['beta'] = np.arccos(df['x'] / np.sqrt(df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2))
    
    return(df)
    
def node_inst_vel(df, roll_window_numpts, start):
    lr_xyz = PandasRollingOLS(y=df.magnitude, x=df.td, window=roll_window_numpts).beta

    df = df.loc[df.index >= start]

    velocity = lr_xyz[lr_xyz.index >= start]['feature1'].values
    df.loc[:, 'velocity'] = np.round(velocity, 4)
    
    return(df)

def old_node_inst_vel(df, roll_window_numpts, start):
    lr_xz = PandasRollingOLS(y=df.xz, x=df.td, window=roll_window_numpts).beta
    lr_xy = PandasRollingOLS(y=df.xy, x=df.td, window=roll_window_numpts).beta

    df = df.loc[df.index >= start]

    vel_xz = lr_xz[lr_xz.index >= start]['feature1'].values
    vel_xy = lr_xy[lr_xy.index >= start]['feature1'].values
    
    df.loc[:, 'vel_xz'] = np.round(vel_xz, 4)
    df.loc[:, 'vel_xy'] = np.round(vel_xy, 4)
    
    return(df)

def old_processing(df):
    
    df['theta_xz'] = np.arctan(df['z'] / (np.sqrt(df['x']**2 + df['y']**2)))
    df['theta_xy'] = np.arctan(df['y'] / (np.sqrt(df['x']**2 + df['z']**2)))
    df['xz'] = seg_len * np.round(np.sin(df['theta_xz']), 4)
    df['xy'] = seg_len * np.round(np.sin(df['theta_xy']), 4)
#    df['mag'] = np.sqrt(df['xz'] ** 2 + df['xy'] ** 2)
    
    return(df)

def old_fill_smooth(df, offsetstart, end, roll_window_numpts, to_smooth, to_fill):    
    if to_fill:
        # filling NAN values
        df = df.fillna(method = 'pad')
        
        #Checking, resolving and reporting fill process    
        if df.isnull().values.any():
            for n in ['xz', 'xy']:
                if df[n].isnull().values.all():
#                    node NaN all values
                    df.loc[:, n] = 0
                elif np.isnan(df[n].values[0]):
#                    node NaN 1st value
                    df.loc[:, n] = df[n].fillna(method='bfill')

    #dropping rows outside monitoring window
    df=df[(df.index >= offsetstart) & (df.index <= end)]
    
    if to_smooth and len(df)>1:
        df.loc[:, ['xz', 'xy']] = df[['xz', 'xy']].rolling(window=roll_window_numpts, min_periods=1).mean()
        df = np.round(df[roll_window_numpts-1:], 4)
        
    return(df)    

def fill_smooth(df, offsetstart, end, roll_window_numpts, to_smooth, to_fill):    
    if to_fill:
        # filling NAN values
        df = df.fillna(method = 'pad')
        
        #Checking, resolving and reporting fill process    
        if df.isnull().values.any():
            for n in ['displacement']:
                if df[n].isnull().values.all():
#                    node NaN all values
                    df.loc[:, n] = 0
                elif np.isnan(df[n].values[0]):
#                    node NaN 1st value
                    df.loc[:, n] = df[n].fillna(method='bfill')

    #dropping rows outside monitoring window
    df=df[(df.index >= offsetstart) & (df.index <= end)]
    
    if to_smooth and len(df)>1:
        df.loc[:, ['displacement']] = df[['displacement']].rolling(window=roll_window_numpts, min_periods=1).mean()
        df = np.round(df[roll_window_numpts-1:], 4)
        
    return(df)

def alert_generator(df, disp_threshold, vel_threshold):
    df['alert'] = np.where(abs(df['displacement'] > disp_threshold), 'L1', 'L0')
    df['alert'] = np.where(abs(df['velocity'] > vel_threshold), 'L1 velocity', df['alert'])
#    df['alert'] = np.where(abs(df['velocity'] > 0.5), 'L2 velocity', df['alert'])

    return(df)

def rolling_spearman(df, seqa, seqb, window):
    stridea = seqa.strides[0]
    ssa = as_strided(seqa, shape=[len(seqa) - window + 1, window], strides=[stridea, stridea])
    strideb = seqa.strides[0]
    ssb = as_strided(seqb, shape=[len(seqb) - window + 1, window], strides =[strideb, strideb])
    ar = pd.DataFrame(ssa)
    br = pd.DataFrame(ssb)
    ar = ar.rank(1)
    br = br.rank(1)
    corrs = ar.corrwith(br, 1)
    df['spearman'] = pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)
    
    return(df['spearman'])
    
def alert_filter(df):
    df['alert'] = np.where((abs(df['yz_trend']) > 0.6) | abs(df['beta_trend'] > 0.6), df['alert'], 'L0')

    return(df)
    
def old_alert_generator(df):
    df['alert'] = np.where((abs(df['xz'] > 0.05)) | (abs(df['xy']) > 0.05), 'L1', 'L0')
    df['alert'] = np.where((abs(df['vel_xz'] > 0.032)) | (abs(df['vel_xy'] > 0.032)), 'L1 velocity', df['alert'])
    df['alert'] = np.where((abs(df['vel_xz'] > 0.5)) | (abs(df['vel_xy'] > 0.5)), 'L2 velocity', df['alert'])
    
    return(df)
    
def is_accelerating(df, df_old):
    df['actual'] = np.where((df['velocity'] > df['velocity'].shift(1)) & (abs(df['velocity']) > 0.032), 'p', 'n')
    df['predicted'] = np.where(abs(df['velocity']) > 0.032, 'p', 'n')
    
    return(df)

def confusion_matrix(df):
    df['con_mat'] = np.where((df['actual'] == 'p') & (df['predicted'] == 'p'), 'true positive', 'none')
    df['con_mat'] = np.where((df['actual'] == 'n') & (df['predicted'] == 'n'), 'true negative', df['con_mat'])
    df['con_mat'] = np.where((df['actual'] == 'p') & (df['predicted'] == 'n'), 'false negative', df['con_mat'])
    df['con_mat'] = np.where((df['actual'] == 'n') & (df['predicted'] == 'p'), 'false positive', df['con_mat'])
    
    return(df)
    
def roc(df):
    tp = len(df.loc[df['con_mat'] == 'true positive'])
    tn = len(df.loc[df['con_mat'] == 'true negative'])
    fp = len(df.loc[df['con_mat'] == 'false positive'])
    fn = len(df.loc[df['con_mat'] == 'false negative'])
    
    tp_rate = tp/tp+fp
    fp_rate = fn/tn+fn
    
    roc_point = (fp_rate, tp_rate)
    
    return(roc_point)
    
seg_len = 1
sub_sensor = 'magta'
node_id = 15
type_num = 32
df = get_raw_accel_data(sub_sensor, node_id, type_num)


#new processing of subsurface data

df = magnitude(df)
df = df.set_index('ts')
df = df.resample('30Min').pad()
window, sc = rtw.get_window('2017-10-01 00:00:00', 365)
#df_old = df_old.drop(columns=['magnitude', 'theta_yz', 'td', 'slope', 'intercept', 'linreg', 'MA', 'displacement', 'beta'])
df = theta_yz(df)
#df = signed_magnitude(df)

df.loc[:, 'td'] = df.index.values - df.index.values[0]
df.loc[:, 'td'] = df['td'].apply(lambda x: x / \
                                            np.timedelta64(1,'D'))

df['slope'] = PandasRollingOLS(y=df.magnitude, x=df.td, window=7).beta
df['intercept'] = PandasRollingOLS(y=df.magnitude, x=df.td, window=7).alpha
df['linreg'] = (df['slope'] * df['td']) + df['intercept']
df['MA'] = df['linreg'].rolling(7).mean()
df['displacement'] = np.round(df['MA'] - df['MA'].shift(144), 4)
df = beta_filter(df)
df = fill_smooth(df, window.offsetstart, window.end, 7, 1, 1)
df['beta_trend'] = rolling_spearman(df, df.td, df.beta, 144)
df = df.drop(columns=['spearman'])
df['yz_trend'] = rolling_spearman(df, df.td, df.theta_yz, 144)
df = df.drop(columns=['spearman'])


df_disp = df.reset_index()
'''
    Old subsurface computation starts here 
'''
df_old = get_raw_accel_data(sub_sensor, node_id, type_num)
df_old = df_old.set_index('ts')
df_old = df_old.resample('30Min').pad()
df_old = old_processing(df_old)
df_old.loc[:, 'td'] = df_old.index.values - df_old.index.values[0]
df_old.loc[:, 'td'] = df_old['td'].apply(lambda x: x / \
                                            np.timedelta64(1,'D'))
df_old = old_fill_smooth(df_old, window.offsetstart, window.end, 7, 1, 1)
df_old = old_node_inst_vel(df_old, roll_window_numpts=7, start=window.start)
df_old = old_alert_generator(df_old)


'''
    Plotting of proposed subsurface data processing
'''

fig = plt.figure(figsize = (10, 30))
plt.suptitle(sub_sensor.upper() + ' Node ' + str(node_id) + ' 1 Year Subsurface Data (Proposed)', fontsize = 24)
ax1 = fig.add_subplot(711)
ax1.set_ylabel('x')
ax1.plot(df_old.index, np.array(df_old.x)/1024)
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')


ax2 = fig.add_subplot(712, sharex=ax1)
ax2.set_ylabel('y')
ax2.get_xaxis().set_visible(False)
ax2.plot(df_old.index, np.array(df_old.y)/1024)
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')

ax3 = fig.add_subplot(713, sharex=ax1)
ax3.set_ylabel('z')
ax3.get_xaxis().set_visible(False)
ax3.plot(df_old.index, np.array(df_old.z)/1024)
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')

ax4 = fig.add_subplot(714, sharex=ax1)
ax4.get_xaxis().set_visible(False)
ax4.set_ylabel('displacement')
#ax1.set_title('Time series of displacement')
#ax1.plot(df.index, df.magnitude, label='raw')
#ax1.plot(df.index, df.linreg, label='linreg')
ax4.plot(df.index, df.displacement, label='displacement')
plt.axhline(y=0.05, color='r', linestyle='--')
plt.axhline(y=-0.05, color='r', linestyle='--')
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')
plt.legend()


df = node_inst_vel(df, roll_window_numpts=7, start=window.start)
#df_old = df_old.apply(old_node_inst_vel, roll_window_numpts=window.numpts, start=window.start)

df = df.reset_index()
df = alert_generator(df, 0.05, 0.032)
#df = df.dropna()
df = alert_filter(df)



print(df.index)
print(df)

ax5 = fig.add_subplot(715, sharex=ax1)
ax5.set_ylabel('velocity')
ax5.get_xaxis().set_visible(False)
ax5.plot('ts', 'velocity', data = df)
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')

plt.axhline(y=0.032, color='r', linestyle='--')
plt.axhline(y=-0.032, color='r', linestyle='--')
plt.axhline(y=0.5, color='y', linestyle='-')
plt.axhline(y=-0.5, color='y', linestyle='-')
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')

#
ax6 = fig.add_subplot(716, sharex=ax1)
ax6.set_ylabel('Beta')
ax6.get_xaxis().set_visible(False)
ax6.plot('ts', 'beta', data = df)
ax6.plot('ts', 'beta_trend', data = df)
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')


ax7 = fig.add_subplot(717, sharex=ax1)
ax7.set_xlabel('ts')
ax7.set_ylabel('theta_yz')
#ax7.get_xaxis().set_visible(False)
ax7.plot('ts', 'theta_yz', data = df)
ax7.plot('ts', 'yz_trend', data = df)

fig.subplots_adjust(hspace = 0.02)
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')
plt.xticks(rotation=45)
plt.show()

'''
    Plotting of old subsurface data processing
'''

fig = plt.figure(figsize = (10, 20))
plt.suptitle(sub_sensor.upper() + ' Node ' + str(node_id) + ' 1 Year Subsurface Data (Current)', fontsize = 24)
ax1 = fig.add_subplot(711, sharex=ax1)
ax1.set_ylabel('x')
ax1.get_xaxis().set_visible(False)
ax1.plot(df_old.index, np.array(df_old.x)/1024)
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')

ax2 = fig.add_subplot(712, sharex=ax1)
ax2.set_ylabel('y')
ax2.get_xaxis().set_visible(False)
ax2.plot(df_old.index, np.array(df_old.y)/1024)
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')

ax3 = fig.add_subplot(713, sharex=ax1)
ax3.set_ylabel('z')
ax3.get_xaxis().set_visible(False)
ax3.plot(df_old.index, np.array(df_old.z)/1024)
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')


ax4 = fig.add_subplot(714, sharex=ax1)
ax4.set_ylabel('xz')
ax4.get_xaxis().set_visible(False)
#ax1.plot(df.index, df.magnitude, label='raw')
#ax1.plot(df.index, df.linreg, label='linreg')
ax4.plot(df_old.index, df_old.xz)
plt.axhline(y=0.05, color='r', linestyle='--')
plt.axhline(y=-0.05, color='r', linestyle='--')
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')

ax5 = fig.add_subplot(715, sharex=ax1)
ax5.set_ylabel('xy')
ax5.get_xaxis().set_visible(False)
ax5.plot(df_old.index, df_old.xy)
plt.axhline(y=0.05, color='r', linestyle='--')
plt.axhline(y=-0.05, color='r', linestyle='--')
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')

ax6 = fig.add_subplot(716, sharex=ax1)
ax6.set_ylabel('vel_xz')
ax6.get_xaxis().set_visible(False)
ax6.plot(df_old.index, df_old.vel_xz)
plt.axhline(y=0.032, color='r', linestyle='--')
plt.axhline(y=-0.032, color='r', linestyle='--')
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
#plt.axvline(x='2016-10-18 00:00:00', color='r', linestyle='-')

ax7 = fig.add_subplot(717, sharex=ax1)
ax7.set_ylabel('vel_xy')
ax7.set_xlabel('ts')
ax7.plot(df_old.index, df_old.vel_xy)
plt.axhline(y=0.032, color='r', linestyle='--')
plt.axhline(y=-0.032, color='r', linestyle='--')
#plt.axvline(x='2016-10-10 00:00:00', color='r', linestyle='-')
plt.xticks(rotation=45)
fig.subplots_adjust(hspace = 0.02)
plt.show()


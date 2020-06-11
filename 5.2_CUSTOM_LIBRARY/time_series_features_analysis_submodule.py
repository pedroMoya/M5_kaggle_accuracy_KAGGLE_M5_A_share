import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
ro.numpy2ri.activate()
base = importr('base')
importr('tsfeatures')
importr('zoo')

print('R packages and libraries imported')

b = np.genfromtxt('../2_RAW_DATA_DIR/sales_train_validation_for_R.csv',
                  delimiter=',', dtype=None, encoding=None)
b = b[0: 1, :]
b = b.flatten()
nr = b.shape[0]
Br = ro.r.matrix(b, nr, 1)
days = ro.r.seq(base.as_Date("2011-01-29"), base.as_Date("2016-04-24"), 'day')
ts = ro.r.zoo(Br, days)
print('time_series loaded to embedded R\n')

acf_features = ro.r.tsfeatures(ts, ro.r.c('acf_features'))
stl_features = ro.r.tsfeatures(ts, ro.r.c('stl_features'))
entropy = ro.r.tsfeatures(ts, ro.r.c('entropy'))
flat_spots = ro.r.tsfeatures(ts, ro.r.c('flat_spots'))
crossing_points = ro.r.tsfeatures(ts, ro.r.c('crossing_points'))
non_linearity = ro.r.tsfeatures(ts, ro.r.c('nonlinearity'))
arch_stat = ro.r.tsfeatures(ts, ro.r.c('arch_stat'))

acf_features_acf_x = acf_features[0][0]
print('auto_correlation coefficient of the time series:', acf_features_acf_x)
acf_features_acf_e = acf_features[0][2]
print('auto_correlation coefficient of the difference:', acf_features_acf_e)
entropy = entropy[0][0]
print('entropy:', entropy)
trend = stl_features[0][2]
print('trend:', trend)
spikiness = stl_features[0][3]
print('spikiness:', spikiness)
linearity = stl_features[0][4]
print('linearity:', linearity)
curvature = stl_features[0][5]
print('curvature:', curvature)
flat_spots = flat_spots[0][0]
print('flat_spots:', flat_spots)
crossing_points = crossing_points[0][0]
print('crossing_points:', crossing_points)
non_linearity = non_linearity[0][0]
print('non_linearity:', non_linearity)
arch_stat = arch_stat[0][0]
print('arch_stats:', arch_stat)

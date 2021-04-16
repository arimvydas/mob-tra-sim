#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@package trafficds
Mobile network traffic datasets
"""
#
# trafficds.py
#
# @brief
# ______________________________
# $Rev:: 2.6.1                 $
# $Author:: arimvydas          $
# $Date:: 2021-01-18 13:50:00  $
# $Notes:: Functions moved from traffic.py     $
# ______________________________
# $Rev:: 2.7.1                 $
# $Author:: arimvydas          $
# $Date:: 2021-03-17 10:49:00  $
# $Notes:: Added dataset generation, alpha-stable distribution, updated functions and descriptions  $
# ______________________________
# $Rev:: 2.7.2                 $
# $Author:: arimvydas          $
# $Date:: 2021-03-22 17:27:00  $
# $Notes:: Small fixes and additions to traffic dataset  $

#  trafficds - Mobile network traffic datasets
#  Copyright (C) 2020-2021 Rimvydas Aleksiejunas
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd
from scipy.special import erf, erfc
from scipy.stats import levy_stable
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Define location of traffic CSV file
file_traffic_csv = 'data/traffic.csv'

# Traffic datasets
ds_traffic = {
    # Daily
    'daily': ['laner12',
              'earth12',
              'feknous14_orange_ds_fixed',
              'feknous14_orange_ds_mobile',
              'feknous14_orange_us_fixed',
              'feknous14_orange_us_mobile',
              'wang15_avg',
              'wang15_park',
              'wang15_campus',
              'wang15_cbd',
              'xu17_daily',
              'okic19_2_hour',
              'okic19_2_min',

              # Daily (weekday(workday)/weekend): wkdy/wknd
              'wkdy_trinh17_1', 'wkdy_trinh17_2',
              'wknd_trinh17_1', 'wknd_trinh17_2',
              'wknd_feldmann_isp_ce_feb22',
              'wkdy_feldmann_isp_ce_mar25',
              'wkdy_moreira_pre_lock_feb19',
              'wkdy_moreira_pre_lock_may19',
              'wkdy_moreira_pre_lock_jul19',
              'wkdy_moreira_pre_lock_oct19',
              'wknd_moreira_pre_lock_feb19',
              'wknd_moreira_pre_lock_may19',
              'wknd_moreira_pre_lock_jul19',
              'wknd_moreira_pre_lock_oct19',
              'wkdy_moreira_lock_mar20',
              'wkdy_moreira_lock_apr20',
              'wkdy_moreira_lock_may20',
              'wkdy_moreira_lock_jun20',
              'wknd_moreira_lock_mar20',
              'wknd_moreira_lock_apr20',
              'wknd_moreira_lock_may20',
              'wknd_moreira_lock_jun20'
              ],

    # Weekly
    'weekly': ['xu17',
               'xu17_residential',
               'xu17_office',
               'xu17_transport',
               'xu17_entertainment',
               'okic19_1_dl',
               'okic19_1_ul',
               'italy_jan',
               'italy_mar',
               'seoul_jan',
               'seoul_mar',
               'feldmann_isp_ce_mar',
               'feldmann_isp_ce_apr',
               'feldmann_isp_ce_jun',
               'milan13_w1_sid4259',
               'milan13_w2_sid4259',
               'milan13_w1_sid4456',
               'milan13_w2_sid4456',
               'milan13_w1_sid5060',
               'milan13_w2_sid5060',
               'milan13_w1_sid5200',
               'milan13_w2_sid5200',
               'milan13_w1_sid5085',
               'milan13_w2_sid5085'
               ]
}


def concat_t_days(a, b, td=2):
    """
    @brief Concatenates daily traffic with smoothing erfc() function

    @param a  First daily traffic data array
    @param b  Second daily traffic data array
    @param td Time step width
    @retval   Concatenated array with smoothing functions

    ______________________________
    $Rev:: 2.5.1                 $
    $Author:: arimvydas          $
    $Date:: 2020-04-16 13:37:00  $
    $Notes:: Initial version     $
    ______________________________
    $Rev:: 2.5.2                 $
    $Author:: arimvydas          $
    $Date:: 2020-04-16 20:08:00  $
    $Notes:: Fixed smoothing functions     $
    ______________________________
    $Rev:: 2.5.3                 $
    $Author:: arimvydas          $
    $Date:: 2020-04-17 20:08:00  $
    $Notes:: Fixed smoothing functions     $
    ______________________________
    $Rev:: 2.5.4                 $
    $Author:: arimvydas          $
    $Date:: 2020-04-24 10:27:00  $
    $Notes:: td added to params  $
    """

    if len(a) == 0:
        return b

    if len(b) == 0:
        return a

    assert (len(a) > 0) or (len(b) > 0), 'Both arrays are empty!'

    N = len(a)

    t = np.arange(len(a) + len(b))
    t1 = t[:N]
    t2 = t[N:]

    def pmerge1(t, tdelta=td):
        return 1 - 0.5 * erfc(t / tdelta)

    def pmerge2(t, tdelta=td):
        return 0.5 * erfc(t / tdelta)

    y1 = pmerge1(t1 - N + 1)
    y2 = pmerge2(t2 - N)
    df = b[0] - a[-1]

    c = np.concatenate((a + 0.5 * df * y1, b - 0.5 * df * y2))

    return c


def combine_traffic(data_seq, df_traffic, day_trend=None, max_thp_mbps=90,
                    coeff_wknd=0.8, week_start=0, seed=None):
    """
    @brief Combines daily/weekly traffic patterns into time-referenced datasets

    @param data_seq     Tuple of column name and number of days or weeks
    @param df_traffic   Dataframe with daily traffic patterns
    @param day_trend    Daily trend -- fraction of traffic increase per day, due
                        to normal or anomalous traffic trend
    @param max_thp_mbps Maximum normal throughput (Mbps)
    @param coeff_wknd   Weekend traffic multiplier
    @param week_start   Integer number indicating the first day of the week:
                        0 - Mon, 1 - Tue, 2 - Wed, etc.
                        week_start = -1 means random week start, useful for
                        randmized traffic simulations without weekly cycle
    @param seed         Random seed is applied in case week_start = -1
    @retval Dataframe consisting of daily time index and throughput column

    Usage example:

        ##

        # Normal traffic growth 30% anually
        normal_inc_day = 0.30 / 365

        # Anomalous trend increase
        anom_inc_day = 0.2 / 7

        inc_day = 2*7*[normal_inc_day] + 2*7*[anom_inc_day]

        df_gen = combine_traffic([('xu17_residential', 2), # weeks
                                  ('wkdy_trinh17_1', 5),   # days
                                  ('wknd_trinh17_1', 2),   # days
                                  ('wkdy_trinh17_1', 5),   # days
                                  ('wknd_trinh17_1', 2)    # days
                                 ], df, day_trend=inc_day, max_thp_mbps=90,
                                 coeff_wknd=0.8)

    ______________________________
    $Rev:: 2.6.1                 $
    $Author:: arimvydas          $
    $Date:: 2020-10-08 15:45:00  $
    $Notes:: Initial version     $
    ______________________________
    $Rev:: 2.6.2                 $
    $Author:: arimvydas          $
    $Date:: 2020-12-22 15:11:00  $
    $Notes:: Option week_start   $
    ______________________________
    $Rev:: 2.6.3                 $
    $Author:: arimvydas          $
    $Date:: 2021-03-15 14:46:00  $
    $Notes:: Updated thp_mult calculation for daily traffic   $
    ______________________________
    $Rev:: 2.7.1                 $
    $Author:: arimvydas          $
    $Date:: 2021-04-15 09:18:00  $
    $Notes:: Randomized option week_start = -1   $
    ______________________________
    $Rev:: 2.7.2                 $
    $Author:: arimvydas          $
    $Date:: 2021-04-16 13:06:00  $
    $Notes:: Random seed option  $
    """

    thp_data = []

    week_days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

    # Adjust week start day
    if week_start > 0:
        week_days = week_days[week_start:] + week_days[:week_start]

    wknd_idx = np.ravel(np.where(np.isin(week_days, ['sat', 'sun'])))

    if week_start == -1:
        np.random.seed(seed)

    # Time sampling interval: 10 min in days scale
    dt = 10 / (60 * 24)
    t0 = 0

    thp = max_thp_mbps
    tot_days = 0
    iday = 0
    for s in data_seq:

        if s[0] in ds_traffic['weekly']:

            #
            # Weekly traffic
            #

            # Loop over weeks
            for w in range(s[1]):

                # If randomized week day order, shuffle week days randomly
                if week_start == -1:
                    np.random.shuffle(week_days)
                    wknd_idx = np.ravel(np.where(np.isin(week_days, ['sat', 'sun'])))

                # Add weekly traffic
                for i, c in enumerate(
                        map(lambda x: 'thp_' + x + '_' + s[0], week_days)):
                    thp += thp * day_trend[iday]
                    thp0 = thp
                    if i in wknd_idx:
                        thp0 *= coeff_wknd
                    thp_data = concat_t_days(thp_data,
                                             thp0 * df_traffic[c].to_numpy())
                    iday += 1
                tot_days += 7

        elif s[0] in ds_traffic['daily']:

            #
            # Daily traffic
            #

            #  Traffic multiplier for weekend
            thp_mult = 1.0

            c = 'thp_' + s[0]
            for d in range(s[1]):

                if s[0][:4] == 'wknd':
                    thp_mult = coeff_wknd
                elif (s[0][:4] != 'wkdy' and
                      ((iday % (7 - week_start) == 5) or
                       (iday % (7 - week_start) == 6))):
                    thp_mult = coeff_wknd
                else:
                    thp_mult = 1.0

                thp += thp * day_trend[iday]
                thp0 = thp
                thp0 *= thp_mult
                thp_data = concat_t_days(thp_data,
                                         thp0 * df_traffic[c].to_numpy())
                iday += 1
                tot_days += 1

        else:
            assert 0, 'Unknown traffic pattern: {}'.format(s[0])


    # N weeks
    Ndays = int(tot_days / dt)
    tdays = np.arange(0, Ndays) * dt + t0

    df_gen = pd.DataFrame(data=tdays, columns=['t_day'])
    df_gen['thp_mbps'] = thp_data

    return df_gen


def thp_time_func(t, area_t=''):
    """
    @brief Generates througput temporal evolution: mean value and with lognormal random variations

    Throughput generation: adds constant value and up to three frequency components:
    24, 12 and 8 hour cycles.

    Use thp_add_randvar() function to add alpha-stable distributed variations.

    Based on publication: S. Wang, X. Zhang, J. Zhang, J. Feng, W. Wang, and K. Xin,
    ``An Approach for Spatial-Temporal Traffic Modeling in Mobile Cellular Networks,"
    in 2015 27th International Teletraffic Congress, 2015, pp. 203-209, doi: 10.1109/ITC.2015.31.

    @param t         Time variable (days)
    @param area_t    Area type: 'park', 'campus', 'cbd' - central business district,
                                'average' - default
    @retval thp_mean Throughput mean value

    ______________________________
    $Rev:: 2.5.1                 $
    $Author:: arimvydas          $
    $Date:: 2020-03-19 18:15:00  $
    $Notes:: Initial version     $
    ______________________________
    $Rev:: 2.5.2                 $
    $Author:: arimvydas          $
    $Date:: 2020-03-31 09:20:00  $
    $Notes:: t (days) instead of t(h) $
    ______________________________
    $Rev:: 2.5.3                 $
    $Author:: arimvydas          $
    $Date:: 2020-04-02 12:38:00  $
    $Notes:: Added thp_max       $
    ______________________________
    $Rev:: 2.6.3                 $
    $Author:: arimvydas          $
    $Date:: 2021-03-17 10:46:00  $
    $Notes:: Only mean throughput left   $
    """

    if area_t.lower() == 'park':
        # Park
        a = [351.06, 222.7, 96.24, 0.0]
        p = [3.11, 2.36, 0.0]
    elif area_t.lower() == 'campus':
        # Campus
        a = [323.04, 143.8, 109.4, 38.43]
        p = [2.98, 2.15, 1.0]
    elif area_t.lower() == 'cbd':
        # Central business district (CBD)
        a = [75.72, 47.52, 16.71, 0.0]
        p = [2.56, 1.45, 0.0]
    else:
        # Whole area (average)
        a = [173.9, 89.83, 52.6, 16.68]
        p = [3.08, 2.08, 1.13]

    thp_mean = a[0];
    for i in range(len(p)):
        thp_mean += a[i + 1] * np.sin(24 * (i + 1) * np.pi / 12 * t + p[i])

    thp_mean /= np.max(thp_mean)

    return thp_mean


def thp_add_randvar(df, sigma=0.1, thp_max=300, alpha=1.6, beta=1, seed=None):
    """
    @brief Add alpha-stable distributed variations to throughput mean with and without anomaly

    @param df      Pandas dataframe with traffic throughput data
    @param sigma   Standard deviation of random process
    @param thp_max Maximum throughput to limit long-tail random throughput values
    @param alpha   Parameter of alpha-stable distribution, 0 < alpha <= 2
    @param beta    Parameter of alpha-stable distribution, -1 <= beta <= 1
    @param seed    Random seed parameter

    ______________________________
    $Rev:: 2.5.1                 $
    $Author:: arimvydas          $
    $Date:: 2020-04-22 18:52:00  $
    $Notes:: Initial version     $
    ______________________________
    $Rev:: 2.5.2                 $
    $Author:: arimvydas          $
    $Date:: 2020-04-28 14:36:00  $
    $Notes:: Updated mean and deviation     $
    ______________________________
    $Rev:: 2.6.1                 $
    $Author:: arimvydas          $
    $Date:: 2020-09-24 10:39:00  $
    $Notes:: Option w/o thp_a_mbps    $
    ______________________________
    $Rev:: 2.6.2                 $
    $Author:: arimvydas          $
    $Date:: 2020-11-30 08:28:00  $
    $Notes:: Option for sigma=0  $
    ______________________________
    $Rev:: 2.6.3                 $
    $Author:: arimvydas          $
    $Date:: 2021-02-04 14:37:00  $
    $Notes:: Alpha-stable distribution  $
    ______________________________
    $Rev:: 2.7.1                 $
    $Author:: arimvydas          $
    $Date:: 2021-04-16 13:06:00  $
    $Notes:: Random seed option  $
    """

    np.random.seed(seed)

    thp_mean = df.thp_mbps
    thp_var = np.zeros(len(thp_mean))

    if 'thp_a_mbps' in df.columns:
        thp_a_mean = df.thp_a_mbps
        thp_a_var = np.zeros(len(thp_mean))

    for n in range(len(thp_mean)):

        # Normal traffic
        if sigma == 0:
            thp_var[n] = thp_mean[n]
        else:
            thp_var[n] = levy_stable.rvs(alpha=alpha, beta=beta,
                                         loc=thp_mean[n], scale=sigma,
                                         size=1)[0]

        # Anomalous traffic
        if 'thp_a_mbps' in df.columns:
            if sigma == 0:
                thp_a_var[n] = thp_a_mean[n]
            else:
                thp_a_var[n] = levy_stable.rvs(alpha=alpha, beta=beta,
                                               loc=thp_a_mean[n], scale=sigma,
                                               size=1)[0]

    # Limit min, max values
    thp_var[thp_var < 0] = 0
    thp_var[thp_var > thp_max] = thp_max

    df['thp_var_mbps'] = thp_var


    if 'thp_a_mbps' in df.columns:

        # Limit min, max values
        thp_a_var[thp_a_var < 0] = 0
        thp_a_var[thp_a_var > thp_max] = thp_max

        df['thp_a_var_mbps'] = thp_a_var

    return


def thp_add_anomaly(df, thp_adiff, astart_day, aend_day):
    """
    @brief Add traffic anomaly of specified amplitude at particular time

    @param df         Pandas dataframe with traffic throughput data
    @param thp_adiff  Amplitude of traffic anomaly
    @param astart_day Start of anomaly in day units
    @param aend_day   End of anomaly in day units

    ______________________________
    $Rev:: 2.5.1                 $
    $Author:: arimvydas          $
    $Date:: 2020-04-22 18:51:00  $
    $Notes:: Initial version     $
    ______________________________
    $Rev:: 2.6.1                 $
    $Author:: arimvydas          $
    $Date:: 2020-10-16 13:11:00  $
    $Notes:: Option with existing thp_a_mbps field for multiple anomalies     $
    """

    if 'thp_a_mbps' not in df.columns:
        df['thp_a_mbps'] = df.thp_mbps

    df.thp_a_mbps[(df.t_day > astart_day) & (df.t_day < aend_day)] += thp_adiff

    return




def gen_dataset_weekly(nweeks=4, sigma=20, thp_max=180, interp_daily=False,
                       std_scaler=True, inc_day=0.3 / 365, thp_mbps=90,
                       coeff_wknd=0.8, week_start=0, seed=None):
    """
    @brief Generates traffic datasets from weekly values for training neural network models

    @param nweeks       Number of weeks to be generated
    @param sigma        Standard deviation of random process
    @param thp_max      Maximum throughput to limit long-tail random throughput values
    @param interp_daily Flag (True/False) if daily interpolation required
    @param std_scaler   Flag (True/False) in order to apply standard scaler
    @param inc_day      Traffic trend increase (fraction per day)
    @param thp_mbps     Maximum normal throughput per day (Mbps)
    @param coeff_wknd   Weekend traffic multiplier
    @param week_start   Integer number indicating the first day of the week:
                        0 - Mon, 1 - Tue, 2 - Wed, etc. Default: 0 - Mon
    @param seed         Random seed parameter
    @retval             Traffic series of size [20, nweeks*7*24] if interp_daily set to True,
                        [20, nweeks*7*24*6] otherwise

    ______________________________
    $Rev:: 2.6.3                 $
    $Author:: arimvydas          $
    $Date:: 2021-03-12 16:11:00  $
    $Notes:: Initial version, copied from Jupyter notebook  $
    ______________________________
    $Rev:: 2.7.1                 $
    $Author:: arimvydas          $
    $Date:: 2021-04-16 13:06:00  $
    $Notes:: Random seed option  $
    """

    data_cols20 = ['xu17',
                   'xu17_residential',
                   'xu17_office',
                   'xu17_transport',
                   'xu17_entertainment',
                   'okic19_1_dl',
                   'okic19_1_ul',
                   'italy_jan',
                   'italy_mar',
                   'seoul_jan',
                   'seoul_mar',
                   'feldmann_isp_ce_mar',
                   'feldmann_isp_ce_apr',
                   'feldmann_isp_ce_jun',
                   'milan13_w1_sid4259',
                   'milan13_w2_sid4259',
                   'milan13_w1_sid4456',
                   'milan13_w2_sid4456',
                   'milan13_w1_sid5060',
                   'milan13_w2_sid5060'
                   ]

    df = pd.read_csv(file_traffic_csv)

    inc_day = nweeks * 7 * [inc_day]

    series = np.empty(
        (20, nweeks * 7 * 24 if interp_daily else nweeks * 7 * 24 * 6))
    tdays = np.linspace(0, nweeks * 7, nweeks * 7 * 24)

    for i, c in enumerate(data_cols20):

        df_gen = combine_traffic([(c, nweeks),  # weeks
                                  ], df,
                                 day_trend=inc_day,
                                 max_thp_mbps=thp_mbps,
                                 coeff_wknd=coeff_wknd,
                                 week_start=week_start,
                                 seed=seed)

        thp_add_randvar(df_gen, sigma, thp_max, seed=seed)

        if interp_daily:
            thp_day = interp1d(df_gen.t_day, df_gen.thp_var_mbps, kind='linear',
                               fill_value='extrapolate')(tdays)
        else:
            thp_day = df_gen.thp_var_mbps.values

        if std_scaler:
            scaler = StandardScaler()
            thp_day = scaler.fit_transform(thp_day.reshape(-1, 1)).reshape(1,
                                                                           -1)

        series[i, :] = thp_day

    return series


def gen_dataset_daily(nweeks=4, sigma=20, thp_max=180, interp_daily=False,
                      std_scaler=True, inc_day=0.3 / 365, thp_mbps=90,
                      coeff_wknd=0.8, week_start=0, seed=None):
    """
    @brief Generates traffic datasets from daily values for training neural network models

    @param nweeks       Number of weeks to be generated
    @param sigma        Standard deviation of random process
    @param thp_max      Maximum throughput to limit long-tail random throughput values
    @param interp_daily Flag (True/False) if daily interpolation required
    @param std_scaler   Flag (True/False) in order to apply standard scaler
    @param inc_day      Traffic trend increase (fraction per day)
    @param thp_mbps     Maximum normal throughput per day (Mbps)
    @param coeff_wknd   Weekend traffic multiplier
    @param week_start   Integer number indicating the first day of the week:
                        0 - Mon, 1 - Tue, 2 - Wed, etc. Default: 0 - Mon
    @param seed         Random seed parameter
    @retval             Traffic series of size [20, nweeks*7*24] if interp_daily set to True,
                        [20, nweeks*7*24*6] otherwise

    ______________________________
    $Rev:: 2.6.3                 $
    $Author:: arimvydas          $
    $Date:: 2021-03-15 13:00:00  $
    $Notes:: Initial version     $
    ______________________________
    $Rev:: 2.6.4                 $
    $Author:: arimvydas          $
    $Date:: 2021-03-22 17:15:00  $
    $Notes:: Added discretized version of [Wang'15] dataset     $
    ______________________________
    $Rev:: 2.7.1                 $
    $Author:: arimvydas          $
    $Date:: 2021-04-16 13:06:00  $
    $Notes:: Random seed option  $
    """


    # Daily data columns (#13)
    data_cols1 =  ['laner12',
                  'earth12',
                  'feknous14_orange_ds_fixed',
                  'feknous14_orange_ds_mobile',
                  'feknous14_orange_us_fixed',
                  'feknous14_orange_us_mobile',
                  'wang15_avg',
                  'wang15_park',
                  'wang15_campus',
                  'wang15_cbd',
                  'xu17_daily',
                  'okic19_2_hour',
                  'okic19_2_min'
                  ]

    # Daily -- weekday columns (#7)
    wkdy_data_cols = [
              'wkdy_trinh17_1',
              'wkdy_trinh17_2',
              'wkdy_feldmann_isp_ce_mar25',
              'wkdy_moreira_pre_lock_feb19',
              'wkdy_moreira_pre_lock_may19',
              'wkdy_moreira_pre_lock_jul19',
              'wkdy_moreira_pre_lock_oct19'
             ]

    # Daily -- weekend columns (#7)
    wknd_data_cols = [
             'wknd_trinh17_1',
             'wknd_trinh17_2',
             'wknd_feldmann_isp_ce_feb22',
             'wknd_moreira_pre_lock_feb19',
             'wknd_moreira_pre_lock_may19',
             'wknd_moreira_pre_lock_jul19',
             'wknd_moreira_pre_lock_oct19'
             ]

    df = pd.read_csv(file_traffic_csv)

    inc_day = nweeks * 7 * [inc_day]

    series = np.empty(
        (13+7, nweeks * 7 * 24 if interp_daily else nweeks * 7 * 24 * 6))
    tdays = np.linspace(0, nweeks * 7, nweeks * 7 * 24)

    for i in range(13+7):

        if i < 13:
            c = data_cols1[i]
            data_cols = nweeks * [(c, 7)]  # days
        else:
            a = wkdy_data_cols[i - 13]
            b = wknd_data_cols[i - 13]
            data_cols = [(a, 5)]  # days
            data_cols.append((b, 2)) # days
            data_cols = nweeks * data_cols

        df_gen = combine_traffic(data_cols,
                                 df,
                                 day_trend=inc_day,
                                 max_thp_mbps=thp_mbps,
                                 coeff_wknd=coeff_wknd,
                                 week_start=week_start,
                                 seed=seed)

        thp_add_randvar(df_gen, sigma, thp_max, seed=seed)

        if interp_daily:
            thp_day = interp1d(df_gen.t_day, df_gen.thp_var_mbps, kind='linear',
                               fill_value='extrapolate')(tdays)
        else:
            thp_day = df_gen.thp_var_mbps.values

        if std_scaler:
            scaler = StandardScaler()
            thp_day = scaler.fit_transform(
                thp_day.reshape(-1, 1)).reshape(1, -1)

        series[i, :] = thp_day

    return series


def gen_dataset(nweeks=4, sigma=20, thp_max=180, interp_daily=False,
                std_scaler=True, inc_day=0.3 / 365, thp_mbps=90,
                coeff_wknd=0.8, week_start=0, seed=None):
    """
    @brief Generates traffic datasets from daily and weekly values for training neural network models

    @param nweeks       Number of weeks to be generated
    @param sigma        Standard deviation of random process
    @param thp_max      Maximum throughput to limit long-tail random throughput values
    @param interp_daily Flag (True/False) if daily interpolation required
    @param std_scaler   Flag (True/False) in order to apply standard scaler
    @param inc_day      Traffic trend increase (fraction per day)
    @param thp_mbps     Maximum normal throughput per day (Mbps)
    @param coeff_wknd   Weekend traffic multiplier
    @param week_start   Integer number indicating the first day of the week:
                        0 - Mon, 1 - Tue, 2 - Wed, etc. Default: 0 - Mon
    @param seed         Random seed parameter
    @retval             Traffic series of size [40, nweeks*7*24] if interp_daily set to True,
                        [40, nweeks*7*24*6] otherwise

    ______________________________
    $Rev:: 2.6.3                 $
    $Author:: arimvydas          $
    $Date:: 2021-03-16 09:04:00  $
    $Notes:: Initial version     $
    ______________________________
    $Rev:: 2.6.4                 $
    $Author:: arimvydas          $
    $Date:: 2021-03-22 17:16:00  $
    $Notes:: Added discretized version of [Wang'15] dataset     $
    ______________________________
    $Rev:: 2.7.1                 $
    $Author:: arimvydas          $
    $Date:: 2021-04-16 13:06:00  $
    $Notes:: Random seed option  $
    """

    series1 = gen_dataset_weekly(nweeks=nweeks, sigma=sigma, thp_max=thp_max,
                                 interp_daily=interp_daily, std_scaler=std_scaler,
                                 inc_day=inc_day, thp_mbps=thp_mbps,
                                 coeff_wknd=coeff_wknd, week_start=week_start,
                                 seed=seed)

    series2 = gen_dataset_daily(nweeks=nweeks, sigma=sigma, thp_max=thp_max,
                                interp_daily=interp_daily, std_scaler=std_scaler,
                                inc_day=inc_day, thp_mbps=thp_mbps,
                                coeff_wknd=coeff_wknd, week_start=week_start,
                                seed=seed)

    series = np.vstack([series1, series2])

    np.random.seed(seed)

    # Apply random row index
    series_rnd = np.take(series, np.random.rand(series.shape[0]).argsort(),
                         axis=0, out=series)

    return series_rnd
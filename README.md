# mob-traffic-ds

Mobile network traffic datasets for anomaly and trend change detection.

## Data sources

The data represents daily and weekly mobile traffic evolutions recorded by network monitoring systems, per base station or mobile user. The traffic data has been digitized from publicly published papers. The following table contains list of traffic datasets and corresponding references.

| Name | Description | Units | Reference |
| ---- | ----------- | ----- | --------- |
| Laner'12 | Twenty-four hour daily data of mean throughput per cell of downlink HSDPA network in Vienna, Austria | Mean throughput per cell (bits/s) | [Laner'12](https://ieeexplore.ieee.org/document/6214330)  |
| EARTH'12 | Twenty-four hour daily traffic profile -- percentage of active users over time derived from average data usage in European countries | Traffic profile (\%) | [EARTH'12](https://cordis.europa.eu/docs/projects/cnect/3/247733/080/deliverables/001-EARTHWP2D23v2.pdf) |
| Milan'13 | Weekly traffic profiles extracted from public dataset provided by Telecom Italia for various places in Milan and Trentino, three weeks used from December 2013 internet usage data | Number of internet connections | [Milan'13](https://www.nature.com/articles/sdata201555) |
| Feknous'14 | Daily traffic profiles captured on Orange France fixed and mobile networks in 2013, upstream and downstream directions  | Normalized traffic volume | [Feknous'14](https://ieeexplore.ieee.org/document/6912519) |
| Polaganga'15 | Daily LTE network throughput in downlink on eNodeB level, averaged over four consecutive weeks excluding weekends |  Traffic volume (MB) | [Polaganga'15](https://www.sciencedirect.com/science/article/abs/pii/S0263224115003991) |
| Wang'15 |  Week-long daily mean traffic approximations using trigonometric series with 24, 12 and 8 hour cycles, for different areas: park, campus and central business district based on data from a mobile operator in China | Traffic volume (MB) | [Wang'15](https://ieeexplore.ieee.org/document/7277444) |
| Xu'17  | Twenty-four hour daily traffic for average day, weekly evolutions for each day of the week and weekly data for different areas: residential, office, transport and entertainment. 3G/LTE network data collected by Internet service provider in Shanghai, China | Throughput (Bytes/10min) and normalized units | [Xu'17](https://ieeexplore.ieee.org/document/7762185) |
| Trinh'17 | Twenty-four hour daily normalized traffic data for average day, separate sets for weekdays and weekends, two datasets from different cells, based on LTE scheduling data collected in one of European metropolitan cities | Normalized traffic units | [Trinh'17](https://ieeexplore.ieee.org/document/8292200) |
| Okic'19-1 | Weekly traffic signatures collected by Vodafone from about 125k customers in 2018, downlink and uplink directions | Normalized data traffic | [Okic19a](https://ieeexplore.ieee.org/document/8885902) |
| Okic'19-2 | Daily traffic signatures collected by Vodafone from about 125k customers in 2018, 10-minute resampling from hourly and 5-minute aggregations |   Normalized data traffic | [Okic19b](https://ieeexplore.ieee.org/document/8881824) |
| Italy'20  | Week-long Internet traffic datasets from Northern Italy, one from January and one from March 2020, corresponding to normal and COVID-19 affected traffic trends, respectively   | Arbitrary units  | [Cloudflare'20](https://blog.cloudflare.com/covid-19-impacts-on-internet-traffic-seattle-italy-and-south-korea/) |
| Seoul'20  | Week-long Internet traffic datasets from Seoul, South Korea, one from January and one from March 2020, corresponding to normal and COVID-19 affected traffic trends, respectively  | Arbitrary units  | [Cloudflare'20](https://blog.cloudflare.com/covid-19-impacts-on-internet-traffic-seattle-italy-and-south-korea/) |
| Feldmann'20  | Week-long time series of traffic volumes from a major Central European Internet service provider representing normal and COVID-19 affected traffic trends   | Normalized aggregated traffic volume per hour  | [Feldmann'20](https://dl.acm.org/doi/10.1145/3419394.3423658) |
| Moreira'20  | Internet download traffic data collected in United States by Federal Communications Commission’s (FCC) Measuring Broadband America (MBA) program before and during COVID-19 pandemic | Average volume of downloaded data per test unit (MB)  | [Moreira'20](https://arxiv.org/abs/2012.09850) |


## Traffic datasets

The following datasets are stored in /data folder:
- `traffic_normal.csv` - Traffic with normal annual trend  
- `traffic_covid.csv`  - COVID-19 pandemic affected traffic   
- `traffic.csv`        - Combined traffic dataset including normal and COVID-19 pandemic traffic. If dataset size doesn't matter, always use this file

The CSV data is stored in tabular format: header with field names and comma separated data values.

The data can be loaded into Python pandas as
```
import pandas as pd
df = pd.read_csv('data/traffic.csv')
```

Data fields: 
- `t_day`  
- `thp_<dataset name>`  
- `thp_<week day>_<dataset name>`  
- `thp_wkdy_<dataset name>`  
- `thp_wknd_<dataset name>`  

Here `<dataset name>` is unique dataset name corresponding to data columns in CSV dataset. `<week day>` indicates day of the week: mon, tue, wed, etc. `t_day` field contains values in range [0.0, ..., 0.99] indicating 10 minute intervals in which 24-hour daytime is divided. Fields beginning with `thp_` contain normalized traffic throughput values in range [0, ..., 1].


## Function library

Python functions for working with traffic data are included into Python file `trafficds.py`. Python version 3.6 or higher is required.

```
 def trafficds.concat_t_days(a,
                             b,
                             td = 2 
                             )       
  Concatenates daily traffic with smoothing erfc() function.

  Parameters
      a    First daily traffic data array
      b    Second daily traffic data array
      td   Time step width

  Return values
    Concatenated array with smoothing functions
  ```

  ```
  def trafficds.combine_traffic(data_seq,
                                df_traffic,
                                day_trend = None,
                                max_thp_mbps = 90,
                                coeff_wknd = 0.8,
                                week_start = None 
                                ) 		
  Combines daily/weekly traffic patterns into time-referenced datasets.

  Parameters
      data_seq     Tuple of column name and number of days or weeks
      df_traffic   Dataframe with daily traffic patterns
      day_trend    Daily trend – fraction of traffic increase per day, due to normal or anomalous traffic trend
      max_thp_mbps Maximum normal throughput (Mbps)
      coeff_wknd   Weekend traffic multiplier
      week_start   Integer number indicating the first day of the week: 0 Mon, 1 - Tue, 2 - Wed, etc. Default: None - Mon

  Return values
      Dataframe	consisting of daily time index and throughput column

  Usage example:
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

  ```


  ```
  def trafficds.thp_add_anomaly(df,
                                thp_adiff,
                                astart_day,
                                aend_day 
                                )       
  Add traffic anomaly of specified amplitude at particular time.

  Parameters
      df           Pandas dataframe with traffic throughput data
      thp_adiff    Amplitude of traffic anomaly
      astart_day   Start of anomaly in day units
      aend_day     End of anomaly in day units
```

```
 def trafficds.thp_add_lognormal(df,
                                 sigma = 0.1,
                                 thp_max = 300 
                                 )       
 Add lognormal variations to throughput mean with and without anomaly.

 Parameters
      df        Pandas dataframe with traffic throughput data
      sigma     Standard deviation of lognormal process
      thp_max   Maximum throughput to limit long-tail random throughput values
```

```
  def trafficds.thp_time_func(t,
                              area_t = '',
                              thp_max = 10
                              )       
  Generates througput temporal evolution: mean value and with lognormal
  random variations.

  Throughput adds constant value and up to three frequency components: 24,
  12 and 8 hour cycles.

  Based on publication: S. Wang, X. Zhang, J. Zhang, J. Feng, W. Wang, and
  K. Xin, ``An Approach for Spatial-Temporal Traffic Modeling in Mobile
  Cellular Networks," in 2015 27th International Teletraffic Congress,
  2015, pp. 203-209, doi: 10.1109/ITC.2015.31.

  Parameters
      t         Time variable (days)
      area_t    Area type: 'park', 'campus', 'cbd' - central business district, 'average' - default
      thp_max   Maximum throughput value

  Return values
      (thp_mean,thp_var)   Throughput mean value and with lognormal random variations
```

## Examples

Several examples for working with traffic data is available in Jupyter notebook `examples.ipynb`.

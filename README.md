# mob-traffic-ds

Mobile network traffic simulator for anomaly and trend change detection


| No. | Name | Description | Units | Reference |
| --- | ---- | ----------- | ----- | --------- |
| 1 | Laner'12 | Twenty-four hour daily data of mean throughput per cell  | Mean throughput per cell (bits/s) | [Laner'12](https://ieeexplore.ieee.org/document/6214330)  |
| 2 | EARTH'12 | Twenty-four hour daily traffic profile -- percentage of active users over time | Traffic profile (\%) | [EARTH'12](https://cordis.europa.eu/docs/projects/cnect/3/247733/080/deliverables/001-EARTHWP2D23v2.pdf) |
| 3 | Feknous'14 | Daily traffic profiles captured on Orange France fixed and mobile networks in 2013, upstream and downstream directions  | Normalized traffic volume | [Feknous'14](https://ieeexplore.ieee.org/document/6912519) |
| 4 | Wang'15 |  Week-long daily mean traffic approximations using trigonometric series with 24, 12 and 8 hour cycles, for different areas: park, campus and central business district | Traffic volume (MB) | [Wang'15](https://ieeexplore.ieee.org/document/7277444) |
| 5 | Xu'17  | Twenty-four hour daily traffic for average day, weekly evolutions for each day of the week and weekly data for different areas: residential, office, transport and entertainment | Throughput (Bytes/10~min) and normalized units | [Xu'17](https://ieeexplore.ieee.org/document/7762185) |
| 6 | Trinh'17 | Twenty-four hour daily normalized traffic data for average day, separate sets for weekdays and weekends, two datasets from different cells  | Normalized traffic units | [Trinh'17](https://ieeexplore.ieee.org/document/8292200) |
| 7 | Italy'20  | Week-long Internet traffic datasets from Northern Italy, one from January and one from March 2020, corresponding to normal and COVID-19 affected traffic trends, respectively   | Arbitrary units  | [Cloudflare'20](https://blog.cloudflare.com/covid-19-impacts-on-internet-traffic-seattle-italy-and-south-korea/) |
| 8 | Seoul'20  | Week-long Internet traffic datasets from Seoul, South Korea, one from January and one from March 2020, corresponding to normal and COVID-19 affected traffic trends, respectively  | Arbitrary units  | [Cloudflare'20](https://blog.cloudflare.com/covid-19-impacts-on-internet-traffic-seattle-italy-and-south-korea/) |
| 9 | Feldmann'20  | Week-long time series of traffic volumes from a major Central European Internet service provider representing normal and COVID-19 affected traffic trends   | Normalized aggregated traffic volume per hour  | [Feldmann'20](https://dl.acm.org/doi/10.1145/3419394.3423658) |
| 10 | Moreira'20  | Internet download traffic data collected in United States by Federal Communications Commission’s (FCC) Measuring Broadband America (MBA) program before and during COVID-19 pandemic | Average volume of downloaded data per test unit (MB)  | [Moreira'20](https://arxiv.org/abs/2012.09850) |


## Function library

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

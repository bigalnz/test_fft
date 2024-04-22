## Database

Kiwitracker uses [`SQLAlchemy`](https://www.sqlalchemy.org/) with `SQLite` backend for database handling.
For migrations [`alembic`](https://alembic.sqlalchemy.org/en/latest/) is used.

Default db name is `main.db`.

## Useful commands:

### SQL

Format date and numbers from SQL tables:

```sql
SELECT 
	strftime("%Y%m%d-%H%M%S", dt) as dt,
	channel,
	printf("%.2f", bpm) as bpm, 
	printf("%.2f", dbfs) as dbfs, 
	printf("%.2f", clipping) as clipping, 
	printf("%.2f", duration) as duration,
	printf("%.2f", snr) as snr, 
	printf("%.2f", lat) as lat, 
	printf("%.2f", lon) as lon
FROM bpm
ORDER BY dt, channel;
```

For exporting to JSON:

```sql
SELECT 
	json_group_array( 
		json_object(
			'dt', strftime("%Y%m%d-%H%M%S", dt),
			'channel', channel,
			'bpm', printf("%.2f", bpm), 
			'dbfs', printf("%.2f", dbfs), 
			'clipping', printf("%.2f", clipping), 
			'duration', printf("%.2f", duration),
			'snr', printf("%.2f", snr), 
			'lat', printf("%.2f", lat), 
			'lon', printf("%.2f", lon)
		)
	)
FROM bpm
ORDER BY dt, channel;
```

```sql
SELECT 
	strftime("%Y%m%d-%H%M%S", start_dt) as start_dt,
	strftime("%Y%m%d-%H%M%S", end_dt) as end_dt,

	channel,
	printf("%.2f", carrier_freq) as carrier_freq, 

	decoding_success,
	
	printf("%.2f", snr_min) as snr_min, 
	printf("%.2f", snr_max) as snr_max, 
	printf("%.2f", snr_mean) as snr_mean, 

	printf("%.2f", dbfs_min) as dbfs_min, 
	printf("%.2f", dbfs_max) as dbfs_max, 
	printf("%.2f", dbfs_mean) as dbfs_mean, 

    days_since_change_of_state,
    days_since_hatch,
    days_since_desertion_alert,
    time_of_emergence,
    weeks_batt_life_left,
    activity_yesterday,
    activity_two_days_ago,
    mean_activity_last_four_days

FROM chick_timer
ORDER BY start_dt, channel;
```

For exporting to JSON:

```sql
SELECT 
	json_group_array( 
		json_object(

			'start_dt', strftime("%Y%m%d-%H%M%S", start_dt),
			'end_dt', strftime("%Y%m%d-%H%M%S", end_dt),

			'channel', channel,
			'carrier_freq', printf("%.2f", carrier_freq), 

			'decoding_success', decoding_success,
			
			'snr_min', printf("%.2f", snr_min), 
			'snr_max', printf("%.2f", snr_max), 
			'snr_mean', printf("%.2f", snr_mean), 

			'dbfs_min', printf("%.2f", dbfs_min), 
			'dbfs_max', printf("%.2f", dbfs_max), 
			'dbfs_mean', printf("%.2f", dbfs_mean), 

		    'days_since_change_of_state', days_since_change_of_state,
		    'days_since_hatch', days_since_hatch,
		    'days_since_desertion_alert', days_since_desertion_alert,
		    'time_of_emergence', time_of_emergence,
		    'weeks_batt_life_left', weeks_batt_life_left,
		    'activity_yesterday', activity_yesterday,
		    'activity_two_days_ago', activity_two_days_ago,
		    'mean_activity_last_four_days', mean_activity_last_four_days
    	)
	)
FROM chick_timer
ORDER BY start_dt, channel;
```

### For developers

Creating new revision:

```
alembic revision --autogenerate -m "<message>"
```

Upgrade to latest DB revision:

```
alembic upgrade head
```

import ee
import datetime

# Инициализация GEE
ee.Initialize(project='interative-module')

# AOI
aoi = ee.Geometry.Polygon([
    [
        [63.075256, 52.918011],
        [62.75116, 52.901862],
        [62.700348, 52.830151],
        [62.819138, 52.747516],
        [63.080063, 52.880734],
        [63.075256, 52.918011]
    ]
])

# Список дат
sentinel_dates = [
    '2020-03-27', '2020-04-21', '2020-04-26', '2020-05-01', '2020-05-06',
    '2020-05-26', '2020-06-05', '2020-06-15', '2020-06-20', '2020-07-15',
    '2020-07-25', '2020-08-04', '2020-08-29', '2020-09-08', '2020-09-28',
    '2020-10-08', '2020-10-18', '2021-04-21', '2021-05-01', '2021-05-06',
    '2021-05-11', '2021-05-16', '2021-05-21', '2021-06-05', '2021-06-30',
    '2021-07-05', '2021-07-20', '2021-08-24', '2021-10-03', '2021-10-08',
    '2021-10-28', '2022-04-06', '2022-05-06', '2022-06-15', '2022-08-09',
    '2022-08-14', '2022-08-19', '2022-08-24', '2022-08-29', '2022-09-13',
    '2022-09-23', '2022-10-03', '2023-04-06', '2023-04-11', '2023-04-26',
    '2023-05-06', '2023-05-21', '2023-05-26', '2023-06-05', '2023-06-10',
    '2023-06-30', '2023-07-05', '2023-07-30', '2023-09-18', '2023-09-23',
    '2024-04-05', '2024-04-20', '2024-05-10', '2024-05-25', '2024-06-14',
    '2024-07-04', '2024-08-28', '2024-09-02', '2024-09-12', '2024-09-17',
    '2024-09-27', '2024-10-02', '2025-03-26', '2025-04-20', '2025-04-30',
    '2025-05-05', '2025-05-30', '2025-06-09', '2025-06-14', '2025-07-16'
]

# Параметры временного окна
time_window_days = 4

def get_nearest_image(ic, target_date):
    target_date = ee.Date(target_date)
    filtered = ic.filterDate(
        target_date.advance(-time_window_days, 'day'),
        target_date.advance(time_window_days + 1, 'day')
    )
    return ee.Image(filtered.sort('system:time_start').first()).clip(aoi)

mod16_ic = ee.ImageCollection('MODIS/006/MOD16A2').select('ET').map(lambda img: img.multiply(0.1))

def export_by_dates(dates, prefix, ic, scale):
    for date_str in dates:
        img = get_nearest_image(ic, date_str)
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=f'{prefix}_{date_str}',
            folder='GEE_Export',
            fileNamePrefix=f'{prefix}_{date_str}',
            scale=scale,
            region=aoi,
            fileFormat='GeoTIFF'
        )
        task.start()
        print(f'Exporting {prefix}_{date_str}...')

# Экспорт только MOD16A2 (500m)
export_by_dates(sentinel_dates, 'MOD16_ET', mod16_ic, 500)

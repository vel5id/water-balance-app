import geopandas as gpd
import pandas as pd
import numpy as np

# --- Настройки (изменять при необходимости) ---
INPUT_SHP = "shoreline.shp"    # Путь к входному shapefile с полигонами
OUTPUT_CSV = "edge_points.csv"  # Имя выходного CSV
NUM_POINTS = 500            # Количество точек по контуру каждого полигона


def generate_edge_points(shp_path: str, num_points: int) -> pd.DataFrame:
    """
    Читает shapefile, извлекает границы полигонов и генерирует по num_points
    равномерно распределённых точек на каждой границе.
    Возвращает DataFrame с колонками fid, id, Z, Y, X.
    """
    gdf = gpd.read_file(shp_path)
    records = []

    for fid, geom in enumerate(gdf.geometry):
        # Для Polygon.boundary возвращается LineString (или MultiLineString для сложных случаев)
        boundary = geom.boundary

        # Обеспечим работу и с MultiLineString
        if boundary.geom_type == 'MultiLineString':
            segments = list(boundary)
            total_length = sum(seg.length for seg in segments)

            # Равномерные расстояния вдоль всей границы
            distances = np.linspace(0, total_length, num_points, endpoint=False)

            # Функция для поиска точки по сквозному расстоянию
            def point_at(dist):
                acc = 0.0
                for seg in segments:
                    if acc + seg.length >= dist:
                        return seg.interpolate(dist - acc)
                    acc += seg.length
                return segments[-1].interpolate(segments[-1].length)

            points = [point_at(d) for d in distances]
        else:
            length = boundary.length
            distances = np.linspace(0, length, num_points, endpoint=False)
            points = [boundary.interpolate(d) for d in distances]

        # Сбор записей
        for idx, pt in enumerate(points):
            records.append({
                'fid': fid,
                'id': idx,
                'Z': 0,
                'Y': pt.y,
                'X': pt.x,
            })

    return pd.DataFrame(records)


def main():
    # Генерация точек
    df = generate_edge_points(INPUT_SHP, NUM_POINTS)
    # Сохранение в CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Сохранено {len(df)} точек в '{OUTPUT_CSV}'")


if __name__ == '__main__':
    main()

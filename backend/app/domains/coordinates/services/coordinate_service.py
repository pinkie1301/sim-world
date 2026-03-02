import logging
import math
from typing import Tuple, Dict, Any, Optional

from skyfield.api import load, wgs84, Distance
import numpy as np

from app.domains.coordinates.models.coordinate_model import (
    GeoCoordinate,
    CartesianCoordinate,
)
from app.domains.coordinates.interfaces.coordinate_service_interface import (
    CoordinateServiceInterface,
)

logger = logging.getLogger(__name__)

# --- Potou Scene Coordinate Conversion Constants ---
# Potou場景基準點: GPS (24.9255373543708, 120.97170270744304) 對應前端座標 (-1800, -3500)
ORIGIN_LATITUDE_POTOU = 24.9255373543708  # Potou場景GPS基準點緯度
ORIGIN_LONGITUDE_POTOU = 120.97170270744304  # Potou場景GPS基準點經度
ORIGIN_FRONTEND_X_POTOU = -1800  # 基準點對應的前端X座標 (米)
ORIGIN_FRONTEND_Y_POTOU = -3500  # 基準點對應的前端Y座標 (米)

# 座標比例因子 - 基於兩個實測點計算
# 點1: 前端(-1800, -3500) = GPS(24.9255373543708, 120.97170270744304)
# 點2: 前端(-2520, 4170) = GPS(24.990143538204382, 121.00856664531)
# ΔY=7670m, Δ緯度=0.0646度 -> 0.0646/7670 = 0.0000084235 度/米
# ΔX=-720m, Δ經度=0.03686度 -> 0.03686/(-720) = -0.0000512 度/米
LATITUDE_SCALE_PER_METER_Y = 0.0000084235  # 度 / 前端Y單位 (米)
LONGITUDE_SCALE_PER_METER_X = -0.0000512  # 度 / 前端X單位 (米) - 注意負號

# Poto場景基準點: GPS (24.923073528120895, 120.97982008858193) 對應前端座標 (-10, -60)
ORIGIN_LATITUDE_POTO = 24.923073528120895  # Poto場景GPS基準點緯度
ORIGIN_LONGITUDE_POTO = 120.97982008858193  # Poto場景GPS基準點經度
ORIGIN_FRONTEND_X_POTO = -10  # 基準點對應的前端X座標 (米)
ORIGIN_FRONTEND_Y_POTO = -60  # 基準點對應的前端Y座標 (米)

# Poto場景比例因子（基於實際測量點計算）
LATITUDE_SCALE_PER_METER_Y_POTO = -8.539671484888491e-06  # 度 / 前端Y單位 (米)
LONGITUDE_SCALE_PER_METER_X_POTO = 8.630300606175349e-06  # 度 / 前端X單位 (米)

# --- TestScene Coordinate Conversion Constants ---
# TestScene場景基準點: GPS (24.943834, 121.369192) 對應前端座標 (0, 0)
# 場景中心即原點，area_m=512，grid_res=128，pixel_size_m=4.0
ORIGIN_LATITUDE_TESTSCENE = 24.943834  # TestScene場景GPS基準點緯度
ORIGIN_LONGITUDE_TESTSCENE = 121.369192  # TestScene場景GPS基準點經度
ORIGIN_FRONTEND_X_TESTSCENE = 0.0  # 基準點對應的前端X座標 (米) - 場景以原點為中心
ORIGIN_FRONTEND_Y_TESTSCENE = 0.0  # 基準點對應的前端Y座標 (米) - 場景以原點為中心

# TestScene場景比例因子（基於WGS-84橢球體在緯度24.94°計算）
# 1度緯度 ≈ 110574 米 -> 1米 ≈ 9.044e-6 度
# 1度經度 ≈ 110574 * cos(24.94°) ≈ 100245 米 -> 1米 ≈ 9.976e-6 度
LATITUDE_SCALE_PER_METER_Y_TESTSCENE = 9.044e-06  # 度 / 前端Y單位 (米)
LONGITUDE_SCALE_PER_METER_X_TESTSCENE = 9.976e-06  # 度 / 前端X單位 (米)

# 保留舊的GLB常數作為備用
ORIGIN_LATITUDE_GLB = 24.786667  # NYCU基準點緯度
ORIGIN_LONGITUDE_GLB = 120.996944  # NYCU基準點經度
LATITUDE_SCALE_PER_GLB_Y = -0.000834 / 100  # 度 / GLB Y 單位
LONGITUDE_SCALE_PER_GLB_X = 0.000834 / 100  # 度 / GLB X 單位

# 地球參數
EARTH_RADIUS_KM = 6371.0  # 地球平均半徑 (公里)
WGS84_A = 6378137.0  # WGS-84 橢球體長半軸 (米)
WGS84_F = 1 / 298.257223563  # WGS-84 扁率
WGS84_B = WGS84_A * (1 - WGS84_F)  # WGS-84 橢球體短半軸 (米)

# Skyfield 的 timescale
try:
    ts = load.timescale(builtin=True)
except Exception as e:
    logger.error(
        f"Skyfield timescale failed to load: {e}. Satellite coordinate conversions will not be available."
    )
    ts = None


class CoordinateService(CoordinateServiceInterface):
    """座標轉換服務實現"""

    async def geo_to_cartesian(self, geo: GeoCoordinate) -> CartesianCoordinate:
        """將地理座標轉換為笛卡爾座標 (簡單投影)"""
        # 簡單球面投影，適合小區域
        lat_rad = math.radians(geo.latitude)
        lon_rad = math.radians(geo.longitude)

        x = EARTH_RADIUS_KM * math.cos(lat_rad) * math.cos(lon_rad)
        y = EARTH_RADIUS_KM * math.cos(lat_rad) * math.sin(lon_rad)
        z = EARTH_RADIUS_KM * math.sin(lat_rad)

        if geo.altitude is not None:
            # 添加高度 (單位轉換: 米 -> 公里)
            alt_km = geo.altitude / 1000.0
            scale = (EARTH_RADIUS_KM + alt_km) / EARTH_RADIUS_KM
            x *= scale
            y *= scale
            z *= scale

        return CartesianCoordinate(x=x, y=y, z=z)

    async def cartesian_to_geo(self, cartesian: CartesianCoordinate) -> GeoCoordinate:
        """將笛卡爾座標轉換為地理座標 (簡單投影)"""
        # 計算距離地心的距離
        r = math.sqrt(cartesian.x**2 + cartesian.y**2 + cartesian.z**2)

        # 計算地理座標
        lon_rad = math.atan2(cartesian.y, cartesian.x)
        lat_rad = math.asin(cartesian.z / r)

        # 轉換為度
        lat_deg = math.degrees(lat_rad)
        lon_deg = math.degrees(lon_rad)

        # 計算高度 (公里 -> 米)
        altitude = (r - EARTH_RADIUS_KM) * 1000.0

        return GeoCoordinate(
            latitude=lat_deg,
            longitude=lon_deg,
            altitude=altitude if altitude > 0.1 else None,  # 如果高度很小就設為 None
        )

    async def geo_to_ecef(self, geo: GeoCoordinate) -> CartesianCoordinate:
        """將地理座標轉換為地球中心地固座標 (ECEF)"""
        if ts is None:
            logger.error(
                "Skyfield timescale not initialized. Cannot perform geo_to_ecef conversion."
            )
            raise RuntimeError(
                "Skyfield timescale not initialized. Coordinate conversions unavailable."
            )

        # 使用 wgs84.latlon 建立地球表面上的點
        earth_location = wgs84.latlon(
            latitude_degrees=geo.latitude,
            longitude_degrees=geo.longitude,
            elevation_m=geo.altitude or 0.0,
        )
        # .itrs_xyz.m 屬性提供 ITRS 框架下的 ECEF 座標 (單位：米)
        x_m, y_m, z_m = earth_location.itrs_xyz.m

        return CartesianCoordinate(x=x_m, y=y_m, z=z_m)

    async def ecef_to_geo(self, ecef: CartesianCoordinate) -> GeoCoordinate:
        """將地球中心地固座標 (ECEF) 轉換為地理座標"""
        if ts is None:
            logger.error(
                "Skyfield timescale not initialized. Cannot perform ecef_to_geo conversion."
            )
            raise RuntimeError(
                "Skyfield timescale not initialized. Coordinate conversions unavailable."
            )

        # 建立一個 Skyfield Distance 物件，表示 ECEF 座標向量 (單位：米)
        position_vector_itrs_m = Distance(m=[ecef.x, ecef.y, ecef.z])

        # 使用 wgs84.geographic_position_of() 將此 ITRS 向量轉換為大地座標
        geopoint = wgs84.geographic_position_of(position_vector_itrs_m)

        return GeoCoordinate(
            latitude=geopoint.latitude.degrees,
            longitude=geopoint.longitude.degrees,
            altitude=geopoint.elevation.m,
        )

    async def bearing_distance(
        self, point1: GeoCoordinate, point2: GeoCoordinate
    ) -> Tuple[float, float]:
        """計算兩點間的方位角和距離"""
        # 轉換為弧度
        lat1 = math.radians(point1.latitude)
        lon1 = math.radians(point1.longitude)
        lat2 = math.radians(point2.latitude)
        lon2 = math.radians(point2.longitude)

        # 計算方位角
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
            lat2
        ) * math.cos(dlon)
        bearing_rad = math.atan2(y, x)
        bearing = math.degrees(bearing_rad)
        # 轉換為 0-360 度
        bearing = (bearing + 360) % 360

        # 使用 Haversine 公式計算距離
        a = (
            math.sin((lat2 - lat1) / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = EARTH_RADIUS_KM * c * 1000  # 轉換為米

        return bearing, distance

    async def destination_point(
        self, start: GeoCoordinate, bearing: float, distance: float
    ) -> GeoCoordinate:
        """根據起點、方位角和距離計算終點座標"""
        # 轉換為弧度
        lat1 = math.radians(start.latitude)
        lon1 = math.radians(start.longitude)
        bearing_rad = math.radians(bearing)

        # 距離轉換為公里
        distance_km = distance / 1000

        # 計算角距離
        angular_distance = distance_km / EARTH_RADIUS_KM

        # 計算終點座標
        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular_distance)
            + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing_rad)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(lat1),
            math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
        )

        # 轉換為度
        lat2_deg = math.degrees(lat2)
        lon2_deg = math.degrees(lon2)

        # 經度規範化到 -180 ~ 180
        lon2_deg = ((lon2_deg + 180) % 360) - 180

        return GeoCoordinate(
            latitude=lat2_deg,
            longitude=lon2_deg,
            altitude=start.altitude,  # 保持與起點相同的高度
        )

    async def utm_to_geo(
        self, easting: float, northing: float, zone_number: int, zone_letter: str
    ) -> GeoCoordinate:
        """將 UTM 座標轉換為地理座標"""
        # 簡單的 UTM 轉經緯度，實際應用中可能需要更複雜的庫
        k0 = 0.9996  # UTM 比例因子
        e = 0.00669438  # WGS-84 偏心率的平方
        e2 = e / (1 - e)

        # 確定半球
        northern = zone_letter >= "N"

        # 計算中央子午線
        cm = 6 * zone_number - 183

        # 調整 northing
        if not northern:
            northing = 10000000 - northing

        # 簡化的緯度計算
        lat_rad = northing / 6366197.724 / 0.9996

        # 實際應用中應使用專業庫如 pyproj 進行精確計算
        # UTM 轉換相當複雜，這裡僅作為簡化示例

        # 簡化實現 - 實際應該使用專業庫如 pyproj
        lat_deg = math.degrees(lat_rad)
        lon_deg = cm + math.degrees(
            math.atan(math.sinh(math.log(math.tan(math.pi / 4 + lat_rad / 2))))
        )

        return GeoCoordinate(latitude=lat_deg, longitude=lon_deg)

    async def geo_to_utm(self, geo: GeoCoordinate) -> Dict[str, Any]:
        """將地理座標轉換為 UTM 座標"""
        # 簡化的 UTM 區帶計算
        zone_number = int((geo.longitude + 180) / 6) + 1

        # 確定區帶字母
        if geo.latitude >= 72.0:
            zone_letter = "X"
        elif geo.latitude >= 64.0:
            zone_letter = "W"
        elif geo.latitude >= 56.0:
            zone_letter = "V"
        elif geo.latitude >= 48.0:
            zone_letter = "U"
        elif geo.latitude >= 40.0:
            zone_letter = "T"
        # ... 其他字母的計算
        else:
            # 簡化: 在赤道以南使用 M 到 A，以北使用 N 到 Z
            latitude_index = int((geo.latitude + 80) / 8)
            if latitude_index < 0:
                latitude_index = 0
            elif latitude_index > 20:
                latitude_index = 20
            zone_letter = "CDEFGHJKLMNPQRSTUVWX"[latitude_index]

        # UTM 轉換 - 這裡是簡化版，生產環境應該使用專業庫
        # 計算中央經線
        lon_origin = (zone_number - 1) * 6 - 180 + 3
        lon_rad = math.radians(geo.longitude)
        lat_rad = math.radians(geo.latitude)

        # 計算 UTM 參數
        N = WGS84_A / math.sqrt(1 - WGS84_F * (2 - WGS84_F) * (math.sin(lat_rad) ** 2))
        T = math.tan(lat_rad) ** 2
        C = (
            WGS84_F
            * (2 - WGS84_F)
            * (math.cos(lat_rad) ** 2)
            / (1 - WGS84_F * (2 - WGS84_F))
        )
        A = math.cos(lat_rad) * (lon_rad - math.radians(lon_origin))

        # 計算 easting 和 northing
        easting = 0.9996 * N * (A + (1 - T + C) * (A**3) / 6) + 500000
        northing = 0.9996 * (
            N
            * math.tan(lat_rad)
            * (
                1
                + math.cos(lat_rad) ** 2
                * (
                    (A**2) / 2
                    + (5 - 4 * T + 42 * C + 13 * C**2 - 28 * (WGS84_F * (2 - WGS84_F)))
                    * (A**4)
                    / 24
                )
            )
        )

        # 南半球調整
        if geo.latitude < 0:
            northing = northing + 10000000

        return {
            "easting": easting,
            "northing": northing,
            "zone_number": zone_number,
            "zone_letter": zone_letter,
        }

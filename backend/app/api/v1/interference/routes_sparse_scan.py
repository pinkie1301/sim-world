"""
UAV sparse ISS sampling API

Provides endpoints for UAV sparse sampling of interference signal strength (ISS) maps
"""

import logging
import numpy as np
import os
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_async_session
from app.domains.simulation.services.sionna_service import generate_iss_map
from app.core.config import ISS_MAP_IMAGE_PATH
from app.domains.coordinates.services.coordinate_service import CoordinateService
from app.domains.coordinates.models.coordinate_model import CartesianCoordinate, GeoCoordinate

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/interference", tags=["Sparse ISS Sampling"])

# 座標轉換服務實例
coordinate_service = CoordinateService()

def frontend_coords_to_gps(x_m: float, y_m: float, z_m: float = 0.0, scene: str = "potou") -> GeoCoordinate:
    """
    將前端座標系統轉換為GPS座標，支持不同場景
    
    支援的場景：
    - potou: 破斗山場景
    - poto: 坡頭漁港場景
    - testscene: TestScene場景
    """
    from app.domains.coordinates.services.coordinate_service import (
        ORIGIN_LATITUDE_POTOU, ORIGIN_LONGITUDE_POTOU,
        ORIGIN_FRONTEND_X_POTOU, ORIGIN_FRONTEND_Y_POTOU,
        LATITUDE_SCALE_PER_METER_Y, LONGITUDE_SCALE_PER_METER_X,
        ORIGIN_LATITUDE_POTO, ORIGIN_LONGITUDE_POTO,
        ORIGIN_FRONTEND_X_POTO, ORIGIN_FRONTEND_Y_POTO,
        LATITUDE_SCALE_PER_METER_Y_POTO, LONGITUDE_SCALE_PER_METER_X_POTO,
        ORIGIN_LATITUDE_TESTSCENE, ORIGIN_LONGITUDE_TESTSCENE,
        ORIGIN_FRONTEND_X_TESTSCENE, ORIGIN_FRONTEND_Y_TESTSCENE,
        LATITUDE_SCALE_PER_METER_Y_TESTSCENE, LONGITUDE_SCALE_PER_METER_X_TESTSCENE
    )
    
    # 根據場景選擇對應的參數
    if scene.lower() == "poto":
        origin_lat = ORIGIN_LATITUDE_POTO
        origin_lon = ORIGIN_LONGITUDE_POTO
        origin_x = ORIGIN_FRONTEND_X_POTO
        origin_y = ORIGIN_FRONTEND_Y_POTO
        lat_scale = LATITUDE_SCALE_PER_METER_Y_POTO
        lon_scale = LONGITUDE_SCALE_PER_METER_X_POTO
    elif scene.lower() == "testscene":
        origin_lat = ORIGIN_LATITUDE_TESTSCENE
        origin_lon = ORIGIN_LONGITUDE_TESTSCENE
        origin_x = ORIGIN_FRONTEND_X_TESTSCENE
        origin_y = ORIGIN_FRONTEND_Y_TESTSCENE
        lat_scale = LATITUDE_SCALE_PER_METER_Y_TESTSCENE
        lon_scale = LONGITUDE_SCALE_PER_METER_X_TESTSCENE
    else:  # 默認使用potou參數
        origin_lat = ORIGIN_LATITUDE_POTOU
        origin_lon = ORIGIN_LONGITUDE_POTOU
        origin_x = ORIGIN_FRONTEND_X_POTOU
        origin_y = ORIGIN_FRONTEND_Y_POTOU
        lat_scale = LATITUDE_SCALE_PER_METER_Y
        lon_scale = LONGITUDE_SCALE_PER_METER_X
    
    # 計算相對於基準點的偏移（前端座標米為單位）
    delta_x = x_m - origin_x  # 相對於基準點的X偏移
    delta_y = y_m - origin_y  # 相對於基準點的Y偏移
    
    # 轉換為GPS座標
    latitude = origin_lat + (delta_y * lat_scale)
    longitude = origin_lon + (delta_x * lon_scale)
    
    return GeoCoordinate(
        latitude=latitude,
        longitude=longitude,
        altitude=z_m if z_m > 0.1 else None
    )


def snake_indices(h: int, w: int, step_y: int = 4, step_x: int = 4):
    """Generate snake-path indices for sparse sampling"""
    for y in range(0, h, step_y):
        rng = range(0, w, step_x) if (y // step_y) % 2 == 0 else range(w - 1, -1, -step_x)
        for x in rng:
            yield y, x


@router.get("/sparse-scan")
async def get_sparse_scan(
    scene: str = Query(description="Scene name (e.g., 'Nanliao')"),
    step_y: int = Query(default=4, description="Y-axis sampling step size"),
    step_x: int = Query(default=4, description="X-axis sampling step size"),
    cell_size: Optional[float] = Query(None, gt=0.1, lt=20.0, description="Map resolution (meters/pixel)"),
    map_width: Optional[int] = Query(None, gt=64, lt=8192, description="Map width (pixels)"),
    map_height: Optional[int] = Query(None, gt=64, lt=8192, description="Map height (pixels)"),
    use_real_iss: bool = Query(default=False, description="Use real ISS map from database devices"),
    # 新增參數：根據設備位置調整掃描區域
    center_on_devices: bool = Query(default=True, description="Center scan area on device positions"),
    scan_radius: float = Query(default=200.0, description="Scan area radius around devices (meters)"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get sparse UAV sampling data for ISS visualization
    
    Loads the full ISS map and coordinate axes from .npy files and generates
    snake-path sampling indices based on the specified step sizes.
    
    Returns JSON with sampling points including coordinates and ISS values.
    """
    try:
        logger.info(f"Sparse scan request: scene={scene}, step_y={step_y}, step_x={step_x}, cell_size={cell_size}, map_size={map_width}x{map_height}")
        
        # Construct data file paths
        data_dir = f"/data/{scene}"
        iss_path = os.path.join(data_dir, "iss_map.npy")
        x_axis_path = os.path.join(data_dir, "x_axis.npy")
        y_axis_path = os.path.join(data_dir, "y_axis.npy")
        
        # Check if files exist
        if not all(os.path.exists(path) for path in [iss_path, x_axis_path, y_axis_path]):
            # For development - create sample data if files don't exist
            logger.warning(f"Data files not found for scene {scene}, creating sample data")
            return await create_sample_sparse_scan_data(
                step_y, step_x, 
                cell_size_override=cell_size,
                map_size_override=(map_width, map_height) if map_width and map_height else None,
                center_on_devices=center_on_devices,
                scan_radius=scan_radius,
                session=session
            )
        
        # Load data
        iss_map = np.load(iss_path)
        x_axis = np.load(x_axis_path)
        y_axis = np.load(y_axis_path)
        
        # Apply custom map parameters if provided
        if cell_size is not None or (map_width is not None and map_height is not None):
            original_height, original_width = iss_map.shape
            original_cell_size = float(x_axis[1] - x_axis[0]) if len(x_axis) > 1 else 1.0
            
            # Use override values or keep original
            new_cell_size = cell_size if cell_size is not None else original_cell_size
            new_width = map_width if map_width is not None else original_width
            new_height = map_height if map_height is not None else original_height
            
            logger.info(f"Resampling map from {original_width}x{original_height} (cell:{original_cell_size:.2f}) to {new_width}x{new_height} (cell:{new_cell_size:.2f})")
            
            # Calculate physical coverage area
            original_x_range = x_axis[-1] - x_axis[0]
            original_y_range = y_axis[-1] - y_axis[0]
            
            # Create new coordinate axes with requested resolution
            x_start = x_axis[0]
            y_start = y_axis[0]
            x_end = x_start + new_width * new_cell_size
            y_end = y_start + new_height * new_cell_size
            
            x_axis = np.linspace(x_start, x_end, new_width)
            y_axis = np.linspace(y_start, y_end, new_height)
            
            # Resample ISS map using nearest neighbor interpolation
            from scipy.interpolate import RegularGridInterpolator
            
            # Create interpolator from original data
            orig_x = np.load(x_axis_path)
            orig_y = np.load(y_axis_path)
            interpolator = RegularGridInterpolator(
                (orig_y, orig_x), iss_map, 
                method='nearest', bounds_error=False, fill_value=0
            )
            
            # Create new grid points
            new_y_grid, new_x_grid = np.meshgrid(y_axis, x_axis, indexing='ij')
            grid_points = np.stack([new_y_grid.ravel(), new_x_grid.ravel()], axis=-1)
            
            # Interpolate ISS values
            iss_map = interpolator(grid_points).reshape(new_height, new_width)
        
        height, width = iss_map.shape
        
        # Generate snake-path sampling points
        points = []
        for i, j in snake_indices(height, width, step_y, step_x):
            if i < height and j < width:  # Bounds check
                x_m = float(x_axis[j])
                y_m = float(y_axis[i])
                iss_dbm = float(iss_map[i, j])
                
                # Check if we need coordinate conversion based on file source
                # For .npy files generated by ISS map service, coordinates are in Sionna system
                from app.domains.simulation.services.sionna_service import to_frontend_coords
                frontend_coords = to_frontend_coords([x_m, y_m, 0])
                
                points.append({
                    "i": i,
                    "j": j,
                    "x_m": frontend_coords[0],  # x stays the same  
                    "y_m": frontend_coords[1],  # y gets converted back from Sionna coords
                    "iss_dbm": iss_dbm
                })
        
        # Add debug info for coordinate system verification
        debug_info = {
            "grid_shape": (height, width),
            "x_range": (float(x_axis[0]), float(x_axis[-1])),
            "y_range": (float(y_axis[0]), float(y_axis[-1])),
            "cell_size_inferred": float(x_axis[1] - x_axis[0]) if len(x_axis) > 1 else "unknown"
        }
        
        logger.info(f"DEBUG sparse-scan:")
        logger.info(f"  grid: {debug_info['grid_shape']}")
        logger.info(f"  x_range: {debug_info['x_range']}")  
        logger.info(f"  y_range: {debug_info['y_range']}")
        logger.info(f"  cell_size: {debug_info['cell_size_inferred']}")
        logger.info(f"  first 3 points: {[(p['x_m'], p['y_m']) for p in points[:3]]}")

        # Convert axis data back to frontend coordinates for the response
        from app.domains.simulation.services.sionna_service import to_frontend_coords
        frontend_x_axis = x_axis.tolist()  # x_axis stays the same
        frontend_y_axis = [-y for y in y_axis.tolist()]  # negate y_axis to convert back from Sionna coords
        
        # 獲取干擾源設備的GPS位置
        jammer_locations_gps = []
        if session is not None:
            try:
                from app.domains.device.services.device_service import DeviceService
                from app.domains.device.adapters.sqlmodel_device_repository import SQLModelDeviceRepository
                
                device_repository = SQLModelDeviceRepository(session)
                device_service = DeviceService(device_repository)
                devices = await device_service.get_devices(active_only=True)
                
                # 找出所有活躍的干擾源設備
                for device in devices:
                    if device.active and device.role == 'jammer':
                        gps_coord = frontend_coords_to_gps(
                            device.position_x, 
                            device.position_y, 
                            device.position_z,
                            scene
                        )
                        jammer_locations_gps.append({
                            "device_id": device.id,
                            "device_name": device.name,
                            "device_role": device.role,
                            "frontend_coords": {
                                "x": device.position_x,
                                "y": device.position_y, 
                                "z": device.position_z
                            },
                            "gps_coords": {
                                "latitude": gps_coord.latitude,
                                "longitude": gps_coord.longitude,
                                "altitude": gps_coord.altitude
                            }
                        })
                        logger.info(f"干擾源 {device.name}: Frontend({device.position_x:.1f}, {device.position_y:.1f}, {device.position_z:.1f}) -> GPS({gps_coord.latitude:.6f}, {gps_coord.longitude:.6f})")
                        
            except Exception as e:
                logger.error(f"獲取干擾源GPS位置失敗: {e}")
        
        # 對真實ISS地圖數據執行CFAR峰值檢測
        cfar_peaks_gps = []
        try:
            from scipy.ndimage import gaussian_filter
            
            # 對ISS地圖執行CFAR檢測
            iss_smooth = gaussian_filter(iss_map, sigma=1.0)
            
            # 統計閾值計算
            iss_mean = np.mean(iss_smooth)
            iss_std = np.std(iss_smooth)
            iss_max = np.max(iss_smooth)
            
            # 更嚴格的閾值設定
            percentile_threshold = np.percentile(iss_smooth, 99.8)  # 提高到99.8%
            statistical_threshold = iss_mean + 4 * iss_std  # 4-sigma規則
            max_threshold = iss_max * 0.7  # 提高到70%
            
            threshold = max(percentile_threshold, statistical_threshold, max_threshold)
            
            # 峰值檢測
            try:
                from skimage.feature import peak_local_maxima as peak_local_max
            except ImportError:
                try:
                    from skimage.feature import peak_local_max
                except ImportError:
                    peak_local_max = None
            
            if peak_local_max is not None:
                peak_coords = peak_local_max(
                    iss_smooth,
                    min_distance=15,  # 增加最小距離
                    threshold_abs=threshold
                )
                
                # 限制最多檢測5個最強峰值
                if len(peak_coords) > 5:
                    # 按強度排序，取前5個
                    peak_intensities = [iss_smooth[coord[0], coord[1]] for coord in peak_coords]
                    sorted_indices = np.argsort(peak_intensities)[::-1]  # 降序
                    peak_coords = peak_coords[sorted_indices[:5]]
                
                logger.info(f"Real data - CFAR檢測參數: threshold={threshold:.2f}, mean={iss_mean:.2f}, std={iss_std:.2f}, max={iss_max:.2f}")
                logger.info(f"Real data - 檢測到 {len(peak_coords)} 個CFAR峰值")
                
                # 轉換峰值座標為GPS
                for i, coord in enumerate(peak_coords):
                    row_idx, col_idx = coord[0], coord[1]
                    
                    if 0 <= row_idx < len(frontend_y_axis) and 0 <= col_idx < len(frontend_x_axis):
                        x_frontend = frontend_x_axis[col_idx]
                        y_frontend = frontend_y_axis[row_idx]
                        
                        gps_coord = frontend_coords_to_gps(x_frontend, y_frontend, 0.0, scene)
                        iss_value = float(iss_map[row_idx, col_idx])
                        
                        peak_info = {
                            "peak_id": i + 1,
                            "grid_coords": {"row": int(row_idx), "col": int(col_idx)},
                            "frontend_coords": {"x": x_frontend, "y": y_frontend},
                            "gps_coords": {
                                "latitude": gps_coord.latitude,
                                "longitude": gps_coord.longitude,
                                "altitude": gps_coord.altitude
                            },
                            "iss_strength_dbm": iss_value
                        }
                        cfar_peaks_gps.append(peak_info)
            
        except Exception as e:
            logger.warning(f"Real data - CFAR峰值檢測失敗: {e}")
            cfar_peaks_gps = []

        return {
            "success": True,
            "height": height,
            "width": width,
            "x_axis": frontend_x_axis,
            "y_axis": frontend_y_axis,
            "points": points,
            "total_points": len(points),
            "step_x": step_x,
            "step_y": step_y,
            "scene": scene,
            "debug_info": debug_info,
            "jammer_locations_gps": jammer_locations_gps,  # 設備干擾源GPS位置
            "cfar_peaks_gps": cfar_peaks_gps  # 新增：CFAR檢測峰值GPS位置
        }
        
    except Exception as e:
        logger.error(f"Sparse scan API failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sparse scan failed: {str(e)}")


async def create_sample_sparse_scan_data(
    step_y: int = 4, 
    step_x: int = 4,
    cell_size_override: Optional[float] = None,
    map_size_override: Optional[tuple[int, int]] = None,
    center_on_devices: bool = True,
    scan_radius: float = 200.0,
    session: Optional[AsyncSession] = None
) -> Dict[str, Any]:
    """Create sample data for development/testing - matching backend ISS map parameters"""
    # Use override parameters if provided, otherwise use defaults
    height, width = map_size_override if map_size_override else (512, 512)
    cell_size = cell_size_override if cell_size_override else 1.0  # Changed default to match ISS map
    
    # Calculate scan area based on device positions if requested
    if center_on_devices and session is not None:
        try:
            from app.domains.device.services.device_service import DeviceService
            from app.domains.device.adapters.sqlmodel_device_repository import SQLModelDeviceRepository
            from app.domains.simulation.services.sionna_service import to_sionna_coords
            
            device_repository = SQLModelDeviceRepository(session)
            device_service = DeviceService(device_repository)
            devices = await device_service.get_devices(active_only=True)
            
            if devices:
                # Find the first active RX device (receiver)
                rx_device = None
                for device in devices:
                    if device.active and device.role == 'receiver':
                        rx_device = device
                        break
                
                if rx_device:
                    # Use RX device position as scan center in frontend coordinates
                    # Store in Sionna coordinates for internal grid creation
                    sionna_pos = to_sionna_coords([rx_device.position_x, rx_device.position_y, rx_device.position_z])
                    center_x = sionna_pos[0]
                    center_y = sionna_pos[1]
                    
                    # Create scan area centered on RX device
                    x_start = center_x - scan_radius
                    x_end = center_x + scan_radius
                    y_start = center_y - scan_radius
                    y_end = center_y + scan_radius
                    
                    logger.info(f"Centering sparse scan on RX device: {rx_device.name} at Sionna coords ({center_x:.1f}, {center_y:.1f})")
                    logger.info(f"RX device frontend coords: ({rx_device.position_x:.1f}, {rx_device.position_y:.1f})")
                    logger.info(f"Scan area in Sionna coords: x=[{x_start:.1f}, {x_end:.1f}], y=[{y_start:.1f}, {y_end:.1f}]")
                    logger.info(f"Scan area in frontend coords: x=[{x_start:.1f}, {x_end:.1f}], y=[{-y_end:.1f}, {-y_start:.1f}]")
                    
                    logger.info(f"動態掃描區域: 中心=({center_x:.1f}, {center_y:.1f}), 半徑={scan_radius}m")
                    logger.info(f"掃描範圍: x=[{x_start:.1f}, {x_end:.1f}], y=[{y_start:.1f}, {y_end:.1f}]")
                else:
                    # No active devices - use default centered area
                    x_start = -width * cell_size / 2
                    x_end = width * cell_size / 2
                    y_start = -height * cell_size / 2
                    y_end = height * cell_size / 2
            else:
                # No devices - use default centered area
                x_start = -width * cell_size / 2
                x_end = width * cell_size / 2
                y_start = -height * cell_size / 2
                y_end = height * cell_size / 2
        except Exception as e:
            logger.warning(f"無法獲取設備位置，使用默認掃描區域: {e}")
            x_start = -width * cell_size / 2
            x_end = width * cell_size / 2
            y_start = -height * cell_size / 2
            y_end = height * cell_size / 2
    else:
        # Use default centered area
        x_start = -width * cell_size / 2   # -256m 
        x_end = width * cell_size / 2      # +256m
        y_start = -height * cell_size / 2  # -256m  
        y_end = height * cell_size / 2     # +256m
    
    x_axis = np.linspace(x_start, x_end, width)
    y_axis = np.linspace(y_start, y_end, height)
    
    # Create sample ISS map with some interesting patterns
    X, Y = np.meshgrid(x_axis, y_axis)
    
    # Create interference pattern with multiple sources
    iss_map = np.zeros((height, width))
    
    # Add interference sources at different locations (matching typical TX positions)
    # Convert real-world positions to grid indices
    # Example: jammer at (-50, 60) meters -> find corresponding grid indices  
    def world_to_grid(x_m, y_m):
        j = int((x_m - x_start) / cell_size)
        i = int((y_m - y_start) / cell_size) 
        return max(0, min(height-1, i)), max(0, min(width-1, j))
    
    # Get actual device positions from database if available
    device_positions_world = []
    if session is not None:
        try:
            from app.domains.device.services.device_service import DeviceService
            from app.domains.device.adapters.sqlmodel_device_repository import SQLModelDeviceRepository
            from app.domains.simulation.services.sionna_service import to_frontend_coords
            
            device_repository = SQLModelDeviceRepository(session)
            device_service = DeviceService(device_repository)
            devices = await device_service.get_devices(active_only=True)
            
            if devices:
                for device in devices:
                    if device.active:
                        # Use actual device position from database
                        # Convert from database coordinates to frontend coordinates for ISS map generation
                        frontend_pos = to_frontend_coords([device.position_x, device.position_y, device.position_z])
                        
                        # Determine power based on device role
                        power = 40 if device.role == 'jammer' else 30  # Higher power for jammers
                        device_positions_world.append((frontend_pos[0], frontend_pos[1], power))
                        logger.info(f"設備 {device.id} ({device.role}): db_pos=({device.position_x:.1f}, {device.position_y:.1f}, {device.position_z:.1f}) -> frontend_pos=({frontend_pos[0]:.1f}, {frontend_pos[1]:.1f}) -> ISS_power={power}")
        except Exception as e:
            logger.warning(f"無法從數據庫獲取設備位置，使用默認位置: {e}")
    
    # Fallback to default positions if no devices found
    if not device_positions_world:
        device_positions_world = [
            # Default device positions (fallback)
            (100, 60, 40),   # jam1: [100, 60, 40] - 右上角
            (-30, 53, 40),   # jam2: [-30, 53, 67] - 左上角 (使用40高度統一)
            (-105, -31, 40), # jam3: [-105, -31, 64] - 左下角
            # TX stations (發射器) - 較低功率
            (-110, -110, 30), # tx0: [-110, -110, 40] - 左下角遠處
            (-106, 56, 30),   # tx1: [-106, 56, 61] - 左上角
            (100, -30, 30),   # tx2: [100, -30, 40] - 右下角
        ]
    
    sources = []
    for x_m, y_m, power in device_positions_world:
        i, j = world_to_grid(x_m, y_m)
        sources.append((i, j, power))
    
    for src_i, src_j, power in sources:
        # Gaussian-like interference pattern
        dist_sq = (np.arange(height)[:, None] - src_i)**2 + (np.arange(width)[None, :] - src_j)**2
        iss_map += power * np.exp(-dist_sq / 200)
    
    # Add some noise
    iss_map += np.random.normal(0, 2, (height, width))
    
    # Generate snake-path sampling points
    points = []
    for i, j in snake_indices(height, width, step_y, step_x):
        if i < height and j < width:
            x_m = float(x_axis[j])
            y_m = float(y_axis[i])
            iss_dbm = float(iss_map[i, j])
            
            # For sample data: coordinates are already in Sionna system, convert to frontend
            from app.domains.simulation.services.sionna_service import to_frontend_coords
            frontend_coords = to_frontend_coords([x_m, y_m, 0])
            
            points.append({
                "i": i,
                "j": j,
                "x_m": frontend_coords[0],  # x stays the same
                "y_m": frontend_coords[1],  # y gets converted back from Sionna coords
                "iss_dbm": iss_dbm
            })
    
    # Add debug info for coordinate system verification  
    debug_info = {
        "grid_shape": (height, width),
        "x_range": (float(x_axis[0]), float(x_axis[-1])),
        "y_range": (float(y_axis[0]), float(y_axis[-1])),
        "cell_size_inferred": float(x_axis[1] - x_axis[0]) if len(x_axis) > 1 else "unknown",
        "sample_device_positions": device_positions_world
    }
    
    logger.info(f"DEBUG sample sparse-scan:")
    logger.info(f"  grid: {debug_info['grid_shape']}")
    logger.info(f"  x_range: {debug_info['x_range']}")  
    logger.info(f"  y_range: {debug_info['y_range']}")
    logger.info(f"  cell_size: {debug_info['cell_size_inferred']}")
    logger.info(f"  device positions: {device_positions_world}")
    logger.info(f"  first 3 points: {[(p['x_m'], p['y_m']) for p in points[:3]]}")
    
    # Convert axis data back to frontend coordinates for the response
    frontend_x_axis = x_axis.tolist()  # x_axis stays the same
    frontend_y_axis = [-y for y in y_axis.tolist()]  # negate y_axis to convert back from Sionna coords
    
    # 獲取干擾源設備的GPS位置（也適用於sample data）
    jammer_locations_gps = []
    if session is not None:
        try:
            from app.domains.device.services.device_service import DeviceService
            from app.domains.device.adapters.sqlmodel_device_repository import SQLModelDeviceRepository
            
            device_repository = SQLModelDeviceRepository(session)
            device_service = DeviceService(device_repository)
            devices = await device_service.get_devices(active_only=True)
            
            # 找出所有活躍的干擾源設備
            for device in devices:
                if device.active and device.role == 'jammer':
                    gps_coord = frontend_coords_to_gps(
                        device.position_x, 
                        device.position_y, 
                        device.position_z,
                        "sample"
                    )
                    jammer_locations_gps.append({
                        "device_id": device.id,
                        "device_name": device.name,
                        "device_role": device.role,
                        "frontend_coords": {
                            "x": device.position_x,
                            "y": device.position_y, 
                            "z": device.position_z
                        },
                        "gps_coords": {
                            "latitude": gps_coord.latitude,
                            "longitude": gps_coord.longitude,
                            "altitude": gps_coord.altitude
                        }
                    })
                    logger.info(f"Sample data - 干擾源 {device.name}: Frontend({device.position_x:.1f}, {device.position_y:.1f}, {device.position_z:.1f}) -> GPS({gps_coord.latitude:.6f}, {gps_coord.longitude:.6f})")
                    
        except Exception as e:
            logger.error(f"Sample data - 獲取干擾源GPS位置失敗: {e}")
    
    # 對於樣本數據，不使用快取的峰值數據，因為可能不匹配
    # 樣本數據應該根據實際生成的樣本ISS地圖進行CFAR檢測
    cfar_peaks_gps = []
    
    # 對樣本ISS地圖執行簡化的CFAR峰值檢測
    try:
        # 對iss_map執行CFAR檢測
        from scipy.ndimage import gaussian_filter, maximum_filter
        
        # 平滑處理
        iss_smooth = gaussian_filter(iss_map, sigma=1.0)
        
        # 統計閾值計算
        iss_mean = np.mean(iss_smooth)
        iss_std = np.std(iss_smooth)
        iss_max = np.max(iss_smooth)
        
        # 更嚴格的閾值設定
        percentile_threshold = np.percentile(iss_smooth, 99.8)  # 提高到99.8%
        statistical_threshold = iss_mean + 4 * iss_std  # 4-sigma規則
        max_threshold = iss_max * 0.7  # 提高到70%
        
        threshold = max(percentile_threshold, statistical_threshold, max_threshold)
        
        # 峰值檢測
        try:
            from skimage.feature import peak_local_maxima as peak_local_max
        except ImportError:
            try:
                from skimage.feature import peak_local_max
            except ImportError:
                peak_local_max = None
        
        if peak_local_max is not None:
            peak_coords = peak_local_max(
                iss_smooth,
                min_distance=15,  # 增加最小距離
                threshold_abs=threshold
            )
            
            # 限制最多檢測5個最強峰值
            if len(peak_coords) > 5:
                # 按強度排序，取前5個
                peak_intensities = [iss_smooth[coord[0], coord[1]] for coord in peak_coords]
                sorted_indices = np.argsort(peak_intensities)[::-1]  # 降序
                peak_coords = peak_coords[sorted_indices[:5]]
            
            logger.info(f"Sample data - CFAR檢測參數: threshold={threshold:.2f}, mean={iss_mean:.2f}, std={iss_std:.2f}, max={iss_max:.2f}")
            logger.info(f"Sample data - 檢測到 {len(peak_coords)} 個CFAR峰值")
            
            # 轉換峰值座標為GPS
            for i, coord in enumerate(peak_coords):
                row_idx, col_idx = coord[0], coord[1]
                
                if 0 <= row_idx < len(frontend_y_axis) and 0 <= col_idx < len(frontend_x_axis):
                    x_frontend = frontend_x_axis[col_idx]
                    y_frontend = frontend_y_axis[row_idx]
                    
                    gps_coord = frontend_coords_to_gps(x_frontend, y_frontend, 0.0, "sample")
                    iss_value = float(iss_map[row_idx, col_idx])
                    
                    peak_info = {
                        "peak_id": i + 1,
                        "grid_coords": {"row": int(row_idx), "col": int(col_idx)},
                        "frontend_coords": {"x": x_frontend, "y": y_frontend},
                        "gps_coords": {
                            "latitude": gps_coord.latitude,
                            "longitude": gps_coord.longitude,
                            "altitude": gps_coord.altitude
                        },
                        "iss_strength_dbm": iss_value
                    }
                    cfar_peaks_gps.append(peak_info)
        
    except Exception as e:
        logger.warning(f"Sample data - CFAR峰值檢測失敗: {e}")
        cfar_peaks_gps = []
    
    return {
        "success": True,
        "height": height,
        "width": width,
        "x_axis": frontend_x_axis,
        "y_axis": frontend_y_axis,
        "points": points,
        "total_points": len(points),
        "step_x": step_x,
        "step_y": step_y,
        "scene": "sample_data",
        "note": "Using sample data matching backend ISS map parameters",
        "debug_info": debug_info,
        "jammer_locations_gps": jammer_locations_gps,  # 設備干擾源GPS位置
        "cfar_peaks_gps": cfar_peaks_gps  # 新增：CFAR檢測峰值GPS位置
    }


# Export router
__all__ = ["router"]
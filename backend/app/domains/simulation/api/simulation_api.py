import logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.core.config import (
    CFR_PLOT_IMAGE_PATH,
    SINR_MAP_IMAGE_PATH,
    DOPPLER_IMAGE_PATH,
    CHANNEL_RESPONSE_IMAGE_PATH,
    ISS_MAP_IMAGE_PATH,
    TSS_MAP_IMAGE_PATH,
    UAV_SPARSE_MAP_IMAGE_PATH,
    get_scene_xml_path,
)
from app.domains.simulation.models.simulation_model import (
    SimulationParameters,
    SimulationImageRequest,
)
from app.domains.simulation.services.sionna_service import sionna_service

logger = logging.getLogger(__name__)
router = APIRouter()


# 通用的圖像回應函數
def create_image_response(image_path: str, filename: str):
    """建立統一的圖像檔案串流回應"""
    logger.info(f"返回圖像，文件路徑: {image_path}")

    def iterfile():
        with open(image_path, "rb") as f:
            chunk = f.read(4096)
            while chunk:
                yield chunk
                chunk = f.read(4096)

    return StreamingResponse(
        iterfile(),
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/scene-image", response_description="空場景圖像")
async def get_scene_image(
    scene: str = Query("nycu", description="場景名稱 (nycu, lotus, ntpu, nanliao, testscene)"),
):
    """產生並回傳只包含基本場景的圖像 (無設備)"""
    logger.info(f"--- API Request: /scene-image?scene={scene} (empty map) ---")

    try:
        output_path = "app/static/images/scene_empty.png"
        success = await sionna_service.generate_empty_scene_image(output_path, scene.upper())

        if not success:
            raise HTTPException(status_code=500, detail="無法產生空場景圖像")

        return create_image_response(output_path, "scene_empty.png")
    except Exception as e:
        logger.error(f"生成空場景圖像時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成場景圖像時出錯: {str(e)}")


@router.get("/cfr-plot", response_description="通道頻率響應圖")
async def get_cfr_plot(
    session: AsyncSession = Depends(get_session),
    scene: str = Query("nycu", description="場景名稱 (nycu, lotus, testscene)"),
):
    """產生並回傳通道頻率響應 (CFR) 圖"""
    logger.info(f"--- API Request: /cfr-plot?scene={scene} ---")

    try:
        success = await sionna_service.generate_cfr_plot(
            session=session, output_path=str(CFR_PLOT_IMAGE_PATH), scene_name=scene
        )

        if not success:
            raise HTTPException(status_code=500, detail="產生 CFR 圖失敗")

        return create_image_response(str(CFR_PLOT_IMAGE_PATH), "cfr_plot.png")
    except Exception as e:
        logger.error(f"生成 CFR 圖時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成 CFR 圖時出錯: {str(e)}")


@router.get("/sinr-map", response_description="SINR 地圖")
async def get_sinr_map(
    session: AsyncSession = Depends(get_session),
    scene: str = Query("nycu", description="場景名稱 (nycu, lotus, testscene)"),
    sinr_vmin: float = Query(-40.0, description="SINR 最小值 (dB)"),
    sinr_vmax: float = Query(0.0, description="SINR 最大值 (dB)"),
    cell_size: float = Query(1.0, description="Radio map 網格大小 (m)"),
    samples_per_tx: int = Query(10**7, description="每個發射器的採樣數量"),
):
    """產生並回傳 SINR 地圖"""
    logger.info(
        f"--- API Request: /sinr-map?scene={scene}&sinr_vmin={sinr_vmin}&sinr_vmax={sinr_vmax}&cell_size={cell_size}&samples_per_tx={samples_per_tx} ---"
    )

    try:
        success = await sionna_service.generate_sinr_map(
            session=session,
            output_path=str(SINR_MAP_IMAGE_PATH),
            scene_name=scene,
            sinr_vmin=sinr_vmin,
            sinr_vmax=sinr_vmax,
            cell_size=cell_size,
            samples_per_tx=samples_per_tx,
        )

        if not success:
            raise HTTPException(status_code=500, detail="產生 SINR 地圖失敗")

        return create_image_response(str(SINR_MAP_IMAGE_PATH), "sinr_map.png")
    except Exception as e:
        logger.error(f"生成 SINR 地圖時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成 SINR 地圖時出錯: {str(e)}")


@router.get("/radio-map", response_description="無線電地圖 (不含干擾源)")
async def get_radio_map(
    session: AsyncSession = Depends(get_session),
    scene: str = Query("nycu", description="場景名稱 (nycu, lotus, testscene)"),
    sinr_vmin: float = Query(-40.0, description="SINR 最小值 (dB)"),
    sinr_vmax: float = Query(0.0, description="SINR 最大值 (dB)"),
    cell_size: float = Query(1.0, description="Radio map 網格大小 (m)"),
    samples_per_tx: int = Query(10**7, description="每個發射器的採樣數量"),
    center_on_transmitter: bool = Query(True, description="是否以發射器為中心"),
):
    """產生並回傳無線電地圖 (不含干擾源，可選以發射器為中心)"""
    logger.info(
        f"--- API Request: /radio-map?scene={scene}&sinr_vmin={sinr_vmin}&sinr_vmax={sinr_vmax}&cell_size={cell_size}&samples_per_tx={samples_per_tx}&center_on_transmitter={center_on_transmitter} ---"
    )

    try:
        from app.core.config import RADIO_MAP_IMAGE_PATH
        
        success = await sionna_service.generate_radio_map(
            session=session,
            output_path=str(RADIO_MAP_IMAGE_PATH),
            scene_name=scene,
            sinr_vmin=sinr_vmin,
            sinr_vmax=sinr_vmax,
            cell_size=cell_size,
            samples_per_tx=samples_per_tx,
            exclude_jammers=True,
            center_on_transmitter=center_on_transmitter,
        )

        if not success:
            raise HTTPException(status_code=500, detail="產生無線電地圖失敗")

        return create_image_response(str(RADIO_MAP_IMAGE_PATH), "radio_map.png")
    except Exception as e:
        logger.error(f"生成無線電地圖時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成無線電地圖時出錯: {str(e)}")


@router.get("/doppler-plots", response_description="延遲多普勒圖")
async def get_doppler_plots(
    session: AsyncSession = Depends(get_session),
    scene: str = Query("nycu", description="場景名稱 (nycu, lotus, testscene)"),
):
    """產生並回傳延遲多普勒圖"""
    logger.info(f"--- API Request: /doppler-plots?scene={scene} ---")

    try:
        success = await sionna_service.generate_doppler_plots(
            session, str(DOPPLER_IMAGE_PATH), scene_name=scene
        )

        if not success:
            raise HTTPException(status_code=500, detail="產生延遲多普勒圖失敗")

        return create_image_response(str(DOPPLER_IMAGE_PATH), "delay_doppler.png")
    except Exception as e:
        logger.error(f"生成延遲多普勒圖時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成延遲多普勒圖時出錯: {str(e)}")


@router.get("/channel-response", response_description="通道響應圖")
async def get_channel_response(
    session: AsyncSession = Depends(get_session),
    scene: str = Query("nycu", description="場景名稱 (nycu, lotus, testscene)"),
):
    """產生並回傳通道響應圖，顯示 H_des、H_jam 和 H_all 的三維圖"""
    logger.info(f"--- API Request: /channel-response?scene={scene} ---")

    try:
        success = await sionna_service.generate_channel_response_plots(
            session,
            str(CHANNEL_RESPONSE_IMAGE_PATH),
            scene_name=scene,
        )

        if not success:
            raise HTTPException(status_code=500, detail="產生通道響應圖失敗")

        return create_image_response(
            str(CHANNEL_RESPONSE_IMAGE_PATH), "channel_response_plots.png"
        )
    except Exception as e:
        logger.error(f"生成通道響應圖時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成通道響應圖時出錯: {str(e)}")


@router.get("/iss-map", response_description="干擾信號檢測地圖")
async def get_iss_map(
    session: AsyncSession = Depends(get_session),
    scene: str = Query("potou", description="場景名稱 (potou, poto, nycu, lotus, testscene)"),
    tx_x: Optional[float] = Query(None, description="TX位置X座標 (米)"),
    tx_y: Optional[float] = Query(None, description="TX位置Y座標 (米)"),
    tx_z: Optional[float] = Query(None, description="TX位置Z座標 (米)"),
    rx_x: Optional[float] = Query(None, description="RX位置X座標 (米) - 用於實時計算"),
    rx_y: Optional[float] = Query(None, description="RX位置Y座標 (米) - 用於實時計算"),
    rx_z: Optional[float] = Query(None, description="RX位置Z座標 (米) - 用於實時計算"),
    jammer: List[str] = Query([], description="Jammer位置列表 (格式: x,y,z)"),
    force_refresh: bool = Query(False, description="強制重新生成地圖，忽略快取"),
    cell_size: Optional[float] = Query(None, gt=0.1, lt=20.0, description="地圖解析度 (米/像素)"),
    map_width: Optional[int] = Query(None, gt=64, lt=8192, description="地圖寬度 (像素)"),
    map_height: Optional[int] = Query(None, gt=64, lt=8192, description="地圖高度 (像素)"),
    center_on: str = Query("receiver", description="地圖中心選擇 (receiver/transmitter)"),
    cfar_threshold_percentile: float = Query(99.5, ge=90.0, le=99.9, description="CFAR 檢測門檻百分位數"),
    gaussian_sigma: float = Query(1.0, ge=0.1, le=5.0, description="高斯平滑參數"),
    min_distance: int = Query(3, ge=1, le=20, description="峰值檢測最小距離"),
    samples_per_tx: int = Query(10000000, ge=100000, le=100000000, description="每發射器採樣數量"),
    # --- 新增 UAV 稀疏取樣參數 ---
    uav_points: Optional[str] = Query(None, description="UAV取樣點座標串 (格式: x1,y1;x2,y2;...)"),
    num_random_samples: int = Query(0, ge=0, le=100, description="隨機取樣點數量 (若無UAV點)"),
    sparse_noise_std_db: float = Query(0.0, ge=0.0, le=10.0, description="稀疏量測雜訊標準差 (dB)"),
    sparse_first_then_full: bool = Query(False, description="先顯示稀疏取樣再顯示完整地圖"),
    return_json: bool = Query(False, description="返回包含CFAR峰值的JSON數據而不是圖片"),
):
    """產生並回傳干擾信號檢測地圖 (使用 2D-CFAR 技術)"""
    logger.info(f"--- API Request: /iss-map?scene={scene}, force_refresh={force_refresh}, center_on={center_on}, return_json={return_json} ---")
    if tx_x is not None and tx_y is not None:
        logger.info(f"TX位置參數: ({tx_x}, {tx_y}, {tx_z})")
    if rx_x is not None and rx_y is not None:
        logger.info(f"RX位置參數: ({rx_x}, {rx_y}, {rx_z})")
    if jammer:
        for i, jam_pos_str in enumerate(jammer):
            logger.info(f"Jammer {i+1} 位置參數: {jam_pos_str}")
    if cell_size is not None:
        logger.info(f"自定義解析度: {cell_size} 米/像素")
    if map_width is not None and map_height is not None:
        logger.info(f"自定義地圖大小: {map_width} x {map_height} 像素")

    try:
        # 建構位置覆蓋字典
        position_override = {}
        if tx_x is not None and tx_y is not None:
            position_override['tx'] = {
                'x': tx_x,
                'y': tx_y,  # 不在這裡轉換座標，讓 to_sionna_coords 統一處理
                'z': tx_z if tx_z is not None else 30.0
            }
        
        # 添加RX位置覆蓋支持（用於實時計算）
        if rx_x is not None and rx_y is not None:
            position_override['rx'] = {
                'x': rx_x,
                'y': rx_y,  # 不在這裡轉換座標，讓 to_sionna_coords 統一處理
                'z': rx_z if rx_z is not None else 30.0
            }
        
        # 處理多個 jammer 位置
        if jammer:
            jammer_positions = []
            for jam_pos_str in jammer:
                try:
                    x, y, z = map(float, jam_pos_str.split(','))
                    # 不在這裡轉換座標，讓 to_sionna_coords 統一處理
                    jammer_positions.append({'x': x, 'y': y, 'z': z})
                except (ValueError, IndexError) as e:
                    logger.warning(f"無效的 jammer 位置格式: {jam_pos_str}, 錯誤: {e}")
                    continue
            if jammer_positions:
                position_override['jammers'] = jammer_positions
            
        # 處理 UAV 點座標串解析 (保持前端座標系，讓Service層統一轉換)
        parsed_uav_points = None
        if uav_points:
            try:
                parsed_uav_points = []
                point_pairs = uav_points.split(';')
                for pair in point_pairs:
                    if pair.strip():  # 略過空字串
                        x, y = map(float, pair.split(','))
                        # 保持前端座標系，讓Service層統一轉換
                        parsed_uav_points.append((x, y))
                logger.info(f"解析到 {len(parsed_uav_points)} 個 UAV 取樣點 (前端座標系)")
            except (ValueError, IndexError) as e:
                logger.warning(f"無效的 UAV 點座標格式: {uav_points}, 錯誤: {e}")
                parsed_uav_points = None
        
        # 添加大小限制保護
        if map_width is not None and map_height is not None:
            if map_width * map_height > 16_000_000:  # 限制在1600萬像素以內 (約4000x4000)
                raise HTTPException(status_code=400, detail="地圖尺寸過大，請限制在1600萬像素以內")
                
        # 直接調用全域generate_iss_map函數以獲取峰值數據
        from app.domains.simulation.services.sionna_service import generate_iss_map
        import time
        
        result = await generate_iss_map(
            session=session, 
            output_path=str(ISS_MAP_IMAGE_PATH),
            scene_name=scene,
            position_override=position_override,
            force_refresh=force_refresh,
            cell_size_override=cell_size,
            map_size_override=(map_width, map_height) if map_width is not None and map_height is not None else None,
            center_on=center_on,
            cfar_threshold_percentile=cfar_threshold_percentile,
            gaussian_sigma=gaussian_sigma,
            min_distance=min_distance,
            samples_per_tx=samples_per_tx,
            # 新增稀疏採樣參數
            uav_points=parsed_uav_points,
            num_random_samples=num_random_samples,
            sparse_noise_std_db=sparse_noise_std_db,
            sparse_first_then_full=sparse_first_then_full,
        )

        if not result["success"]:
            error_msg = result.get("error", "產生干擾信號檢測地圖失敗")
            raise HTTPException(status_code=500, detail=error_msg)

        # 根據return_json參數決定回應格式
        if return_json:
            # 返回包含CFAR峰值的JSON數據
            return {
                "success": True,
                "scene": scene,
                "cfar_peaks_gps": result["cfar_peaks_gps"],
                "total_peaks": result["total_peaks"],
                "image_url": f"/static/images/iss_map.png?t={int(time.time())}"
            }
        else:
            # 返回圖片（原有行為）
            return create_image_response(str(ISS_MAP_IMAGE_PATH), "iss_map.png")
    except Exception as e:
        logger.error(f"生成干擾信號檢測地圖時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成干擾信號檢測地圖時出錯: {str(e)}")


@router.get("/iss-map-cfar-peaks", response_description="ISS地圖CFAR峰值GPS數據")
async def get_iss_map_cfar_peaks(
    session: AsyncSession = Depends(get_session),
    scene: str = Query("potou", description="場景名稱 (potou, poto, nycu, lotus, testscene)"),
    force_refresh: bool = Query(False, description="強制重新生成ISS地圖以獲取最新峰值")
):
    """獲取ISS地圖的CFAR峰值GPS座標"""
    try:
        from app.domains.simulation.services.sionna_service import SionnaSimulationService
        from app.api.v1.interference.routes_sparse_scan import frontend_coords_to_gps
        
        # 獲取SionnaSimulationService實例
        sionna_service = SionnaSimulationService()
        
        cfar_peaks_gps = []
        
        # 如果需要強制刷新或沒有快取數據，重新生成ISS地圖
        should_regenerate = force_refresh
        
        if not should_regenerate:
            # 檢查是否有有效的快取數據
            from app.domains.simulation.services.sionna_service import generate_iss_map
            if hasattr(generate_iss_map, '_iss_cache') and generate_iss_map._iss_cache:
                # 有快取數據，但檢查是否為當前場景相關
                found_valid_cache = False
                for cache_key, cached_data in generate_iss_map._iss_cache.items():
                    if 'peak_locations_gps' in cached_data and cached_data['peak_locations_gps']:
                        found_valid_cache = True
                        break
                
                if not found_valid_cache:
                    should_regenerate = True
                    logger.info("快取中沒有找到有效的峰值數據，將重新生成ISS地圖")
            else:
                should_regenerate = True
                logger.info("沒有ISS地圖快取數據，將重新生成")
        
        if should_regenerate:
            # 重新生成ISS地圖以確保獲取最新的峰值數據
            logger.info(f"重新生成ISS地圖以獲取最新CFAR峰值，場景: {scene}")
            from app.core.config import ISS_MAP_IMAGE_PATH
            
            # 使用默認參數重新生成ISS地圖
            success = await sionna_service.generate_iss_map(
                session=session,
                output_path=str(ISS_MAP_IMAGE_PATH),
                scene_name=scene
            )
            
            if not success:
                logger.warning("重新生成ISS地圖失敗，嘗試使用現有快取數據")
        
        # 從快取中獲取峰值數據並轉換GPS座標
        from app.domains.simulation.services.sionna_service import generate_iss_map
        if hasattr(generate_iss_map, '_iss_cache') and generate_iss_map._iss_cache:
            # 找到最新的快取條目（基於時間戳）
            latest_cache_data = None
            latest_timestamp = 0
            latest_cache_key = None
            
            for cache_key, cached_data in generate_iss_map._iss_cache.items():
                if 'peak_locations_gps' in cached_data and cached_data['peak_locations_gps']:
                    timestamp = cached_data.get('timestamp', 0)
                    if timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_cache_data = cached_data
                        latest_cache_key = cache_key
            
            if latest_cache_data:
                # 從最新快取的峰值數據中獲取前端座標，然後根據當前場景重新計算GPS座標
                cached_peaks = latest_cache_data['peak_locations_gps']
                logger.info(f"從最新ISS地圖快取獲取到 {len(cached_peaks)} 個CFAR峰值 (時間戳: {latest_timestamp:.0f}, key: {latest_cache_key[:16]}...)，重新計算GPS位置使用場景: {scene}")
                
                for peak_data in cached_peaks:
                    # 獲取前端座標
                    frontend_coords = peak_data.get('frontend_coords', {})
                    x_frontend = frontend_coords.get('x', 0)
                    y_frontend = frontend_coords.get('y', 0)
                    
                    # 使用當前場景參數重新計算GPS座標
                    gps_coord = frontend_coords_to_gps(x_frontend, y_frontend, 0.0, scene)
                    
                    # 構建新的峰值數據，保留其他信息但更新GPS座標
                    updated_peak = peak_data.copy()
                    updated_peak['gps_coords'] = {
                        "latitude": gps_coord.latitude,
                        "longitude": gps_coord.longitude,
                        "altitude": gps_coord.altitude
                    }
                    cfar_peaks_gps.append(updated_peak)
            else:
                logger.warning("快取中沒有找到有效的峰值數據")
        
        return {
            "success": True,
            "scene": scene,
            "cfar_peaks_gps": cfar_peaks_gps,
            "total_peaks": len(cfar_peaks_gps)
        }
        
    except Exception as e:
        logger.error(f"獲取CFAR峰值GPS數據時出錯: {e}", exc_info=True)
        return {
            "success": False,
            "scene": scene,
            "cfar_peaks_gps": [],
            "total_peaks": 0,
            "error": str(e)
        }


@router.get("/tss-map")
async def get_tss_map():
    """返回 TSS (Total Signal Strength) 地圖"""
    logger.info("--- API Request: /tss-map ---")
    try:
        return create_image_response(str(TSS_MAP_IMAGE_PATH), "tss_map.png")
    except Exception as e:
        logger.error(f"返回 TSS 地圖時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"返回 TSS 地圖時出錯: {str(e)}")


@router.get("/uav-sparse-map")
async def get_uav_sparse_map():
    """返回 UAV Sparse 地圖"""
    logger.info("--- API Request: /uav-sparse-map ---")
    try:
        return create_image_response(str(UAV_SPARSE_MAP_IMAGE_PATH), "uav_sparse_map.png")
    except Exception as e:
        logger.error(f"返回 UAV Sparse 地圖時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"返回 UAV Sparse 地圖時出錯: {str(e)}")


@router.post("/run", response_model=Dict[str, Any])
async def run_simulation(
    params: SimulationParameters, session: AsyncSession = Depends(get_session)
):
    """執行通用模擬"""
    logger.info(f"--- API Request: /run (type: {params.simulation_type}) ---")

    try:
        result = await sionna_service.run_simulation(session, params)

        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error_message", "模擬執行失敗"),
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"執行模擬時出錯: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"執行模擬時出錯: {str(e)}",
        )


@router.get("/scenes", response_description="獲取可用場景列表")
async def get_available_scenes():
    """獲取系統中所有可用場景的列表"""
    logger.info("--- API Request: /scenes (獲取可用場景列表) ---")

    try:
        from app.core.config import SCENE_DIR
        import os

        # 檢查場景目錄是否存在
        if not os.path.exists(SCENE_DIR):
            return {"scenes": [], "default": "NYCU"}

        # 獲取所有子目錄作為場景名稱
        scenes = []
        for item in os.listdir(SCENE_DIR):
            scene_path = os.path.join(SCENE_DIR, item)
            if os.path.isdir(scene_path):
                # 檢查是否有GLB模型文件
                if os.path.exists(os.path.join(scene_path, f"{item}.glb")):
                    scenes.append(
                        {
                            "name": item,
                            "has_model": True,
                            "has_xml": os.path.exists(
                                os.path.join(scene_path, f"{item}.xml")
                            ),
                        }
                    )

        # 當沒有場景時返回空列表
        if not scenes:
            return {"scenes": [], "default": "NYCU"}

        return {"scenes": scenes, "default": "NYCU"}
    except Exception as e:
        logger.error(f"獲取場景列表時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取場景列表時出錯: {str(e)}")


@router.get("/scenes/{scene_name}", response_description="獲取特定場景信息")
async def get_scene_info(scene_name: str):
    """獲取特定場景的詳細信息"""
    logger.info(f"--- API Request: /scenes/{scene_name} (獲取場景信息) ---")

    try:
        from app.core.config import (
            get_scene_dir,
            get_scene_model_path,
            get_scene_xml_path,
        )
        import os

        scene_dir = get_scene_dir(scene_name)
        if not os.path.exists(scene_dir):
            raise HTTPException(status_code=404, detail=f"場景 {scene_name} 不存在")

        # 檢查場景文件
        model_path = get_scene_model_path(scene_name)
        xml_path = get_scene_xml_path(scene_name)

        # 獲取場景中的紋理文件
        textures = []
        textures_dir = os.path.join(scene_dir, "textures")
        if os.path.exists(textures_dir):
            textures = [
                f
                for f in os.listdir(textures_dir)
                if os.path.isfile(os.path.join(textures_dir, f))
            ]

        return {
            "name": scene_name,
            "has_model": os.path.exists(model_path),
            "has_xml": os.path.exists(xml_path),
            "textures": textures,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"獲取場景 {scene_name} 信息時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取場景信息時出錯: {str(e)}")


@router.get("/scenes/{scene_name}/model", response_description="獲取場景模型文件")
async def get_scene_model(scene_name: str):
    """獲取特定場景的3D模型文件"""
    logger.info(f"--- API Request: /scenes/{scene_name}/model (獲取場景模型) ---")

    try:
        from app.core.config import get_scene_model_path
        import os

        model_path = get_scene_model_path(scene_name)
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, detail=f"場景 {scene_name} 的模型不存在"
            )

        return StreamingResponse(
            open(model_path, "rb"),
            media_type="model/gltf-binary",
            headers={"Content-Disposition": f"attachment; filename={scene_name}.glb"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"獲取場景 {scene_name} 模型時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"獲取場景模型時出錯: {str(e)}")

# backend/app/api/v1/router.py
from fastapi import APIRouter, Response, status, Query, Request, HTTPException
import os
from starlette.responses import FileResponse
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel

# Import new domain API routers
from app.domains.device.api.device_api import router as device_router

# 恢復領域API路由
from app.domains.coordinates.api.coordinate_api import router as coordinates_router
from app.domains.simulation.api.simulation_api import router as simulation_router

# Import wireless domain API router
from app.domains.wireless.api.wireless_api import router as wireless_router

# Import interference domain API router
from app.domains.interference.api.interference_api import router as interference_router
from app.api.v1.interference.routes_sparse_scan import router as sparse_scan_router

# Import sparse ISS map generation router
from app.api.v1.simulations.routes_sparse_iss_map import router as sparse_iss_map_router

# Import drone tracking domain API router
from app.domains.drone_tracking.api.drone_tracking_api import router as drone_tracking_router

api_router = APIRouter()

# Register domain API routers
api_router.include_router(device_router, prefix="/devices", tags=["Devices"])
# 恢復領域API路由
api_router.include_router(
    coordinates_router, prefix="/coordinates", tags=["Coordinates"]
)
api_router.include_router(
    simulation_router, prefix="/simulations", tags=["Simulations"]
)

# Register wireless domain API router
api_router.include_router(wireless_router, prefix="/wireless", tags=["Wireless"])

# Register interference domain API router
api_router.include_router(interference_router, tags=["Interference"])
api_router.include_router(sparse_scan_router, tags=["Sparse Scan"])

# Register sparse ISS map generation router
api_router.include_router(sparse_iss_map_router, tags=["Sparse ISS Map"])

# Register drone tracking domain API router
api_router.include_router(drone_tracking_router, tags=["Drone Tracking"])


# 添加模型資源路由
@api_router.get("/sionna/models/{model_name}", tags=["Models"])
async def get_model(model_name: str):
    """提供3D模型文件"""
    # 定義模型文件存儲路徑
    static_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "static",
    )
    models_dir = os.path.join(static_dir, "models")

    # 獲取對應的模型文件
    model_file = os.path.join(models_dir, f"{model_name}.glb")

    # 檢查文件是否存在
    if not os.path.exists(model_file):
        return Response(
            content=f"模型 {model_name} 不存在", status_code=status.HTTP_404_NOT_FOUND
        )

    # 返回模型文件
    return FileResponse(
        path=model_file, media_type="model/gltf-binary", filename=f"{model_name}.glb"
    )


# 添加場景資源路由
@api_router.get("/scenes/{scene_name}/model", tags=["Scenes"])
async def get_scene_model(scene_name: str):
    """提供3D場景模型文件"""
    # 定義場景文件存儲路徑
    static_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "static",
    )
    scenes_dir = os.path.join(static_dir, "scenes")
    scene_dir = os.path.join(scenes_dir, scene_name)

    # 獲取對應的場景模型文件
    model_file = os.path.join(scene_dir, f"{scene_name}.glb")

    # 檢查文件是否存在
    if not os.path.exists(model_file):
        return Response(
            content=f"場景 {scene_name} 的模型不存在",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    # 返回場景模型文件
    return FileResponse(
        path=model_file, media_type="model/gltf-binary", filename=f"{scene_name}.glb"
    )


# ===== UAV 位置追蹤端點 =====


class UAVPosition(BaseModel):
    """UAV 位置模型"""

    uav_id: str
    latitude: float
    longitude: float
    altitude: float
    timestamp: str
    speed: Optional[float] = None
    heading: Optional[float] = None


class UAVPositionResponse(BaseModel):
    """UAV 位置響應模型"""

    success: bool
    message: str
    uav_id: str
    received_at: str
    channel_update_triggered: bool = False


# UAV 位置儲存（簡單的記憶體儲存，生產環境應使用資料庫）
uav_positions = {}


@api_router.post("/uav/position", tags=["UAV Tracking"])
async def update_uav_position(position: UAVPosition):
    """
    更新 UAV 位置

    接收來自 NetStack 的 UAV 位置更新，並觸發 Sionna 信道模型重計算

    Args:
        position: UAV 位置資訊

    Returns:
        更新結果
    """
    try:
        # 儲存位置資訊
        uav_positions[position.uav_id] = {
            "latitude": position.latitude,
            "longitude": position.longitude,
            "altitude": position.altitude,
            "timestamp": position.timestamp,
            "speed": position.speed,
            "heading": position.heading,
            "last_updated": datetime.utcnow().isoformat(),
        }

        # 觸發信道模型更新（這裡可以添加實際的 Sionna 整合邏輯）
        channel_update_triggered = await trigger_channel_model_update(position)

        return UAVPositionResponse(
            success=True,
            message=f"UAV {position.uav_id} 位置更新成功",
            uav_id=position.uav_id,
            received_at=datetime.utcnow().isoformat(),
            channel_update_triggered=channel_update_triggered,
        )

    except Exception as e:
        return UAVPositionResponse(
            success=False,
            message=f"位置更新失敗: {str(e)}",
            uav_id=position.uav_id,
            received_at=datetime.utcnow().isoformat(),
        )


@api_router.get("/uav/{uav_id}/position", tags=["UAV Tracking"])
async def get_uav_position(uav_id: str):
    """
    獲取 UAV 當前位置

    Args:
        uav_id: UAV ID

    Returns:
        UAV 位置資訊
    """
    if uav_id not in uav_positions:
        return Response(
            content=f"找不到 UAV {uav_id} 的位置資訊",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    return {"success": True, "uav_id": uav_id, "position": uav_positions[uav_id]}


@api_router.get("/uav/positions", tags=["UAV Tracking"])
async def get_all_uav_positions():
    """
    獲取所有 UAV 位置

    Returns:
        所有 UAV 的位置資訊
    """
    return {
        "success": True,
        "total_uavs": len(uav_positions),
        "positions": uav_positions,
    }


@api_router.delete("/uav/{uav_id}/position", tags=["UAV Tracking"])
async def delete_uav_position(uav_id: str):
    """
    刪除 UAV 位置記錄

    Args:
        uav_id: UAV ID

    Returns:
        刪除結果
    """
    if uav_id in uav_positions:
        del uav_positions[uav_id]
        return {"success": True, "message": f"UAV {uav_id} 位置記錄已刪除"}
    else:
        return Response(
            content=f"找不到 UAV {uav_id} 的位置記錄",
            status_code=status.HTTP_404_NOT_FOUND,
        )


async def trigger_channel_model_update(position: UAVPosition) -> bool:
    """
    觸發 Sionna 信道模型更新

    Args:
        position: UAV 位置

    Returns:
        是否成功觸發更新
    """
    try:
        # 這裡可以添加實際的 Sionna 信道模型更新邏輯
        # 例如：
        # 1. 計算 UAV 與衛星的距離和角度
        # 2. 更新路徑損耗模型
        # 3. 計算都卜勒頻移
        # 4. 更新多路徑衰落參數

        # 現在只是模擬觸發
        print(
            f"觸發 Sionna 信道模型更新: UAV {position.uav_id} at ({position.latitude}, {position.longitude}, {position.altitude}m)"
        )

        # 模擬一些信道參數計算
        import math

        # 假設衛星在 600km 高度
        satellite_altitude = 600000  # 米
        uav_altitude = position.altitude

        # 計算直線距離（簡化計算）
        distance_to_satellite = math.sqrt(
            (satellite_altitude - uav_altitude) ** 2
            + (position.latitude * 111000) ** 2
            + (position.longitude * 111000) ** 2
        )

        # 計算路徑損耗（自由空間損耗）
        frequency_hz = 2.15e9  # 2.15 GHz
        c = 3e8  # 光速
        path_loss_db = (
            20 * math.log10(distance_to_satellite)
            + 20 * math.log10(frequency_hz)
            + 20 * math.log10(4 * math.pi / c)
        )

        print(
            f"計算結果: 距離={distance_to_satellite/1000:.1f}km, 路徑損耗={path_loss_db:.1f}dB"
        )

        return True

    except Exception as e:
        print(f"信道模型更新失敗: {e}")
        return False


# 添加新的 CQRS 衛星端點 — 已移除（衛星領域已刪除）

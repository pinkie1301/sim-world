# backend/app/services/sionna_simulation.py
import logging
import os
import traceback
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field as PydanticField  # Use Pydantic BaseModel
from sionna.rt import (
    load_scene,
    Transmitter as SionnaTransmitter,
    Receiver as SionnaReceiver,
    PlanarArray,
    PathSolver,
    subcarrier_frequencies,
    RadioMapSolver,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

# Import models and config from their new locations
from app.domains.device.models.device_model import Device, DeviceRole
from app.core.config import (
    NYCU_XML_PATH,
    CFR_PLOT_IMAGE_PATH,
    DOPPLER_IMAGE_PATH,
    CHANNEL_RESPONSE_IMAGE_PATH,
    SINR_MAP_IMAGE_PATH,
    ISS_MAP_IMAGE_PATH,
    TSS_MAP_IMAGE_PATH,
    UAV_SPARSE_MAP_IMAGE_PATH,
    get_scene_xml_path,
)

# 從設備領域中導入設備服務和儲存庫
from app.domains.device.services.device_service import DeviceService
from app.domains.device.adapters.sqlmodel_device_repository import (
    SQLModelDeviceRepository,
)

# Import interfaces and models
from app.domains.simulation.interfaces.simulation_service_interface import (
    SimulationServiceInterface,
)
from app.domains.simulation.models.simulation_model import SimulationParameters

# 新增導入 for GLB rendering
import trimesh
import pyrender
from PIL import Image
import io
import tensorflow as tf

# ISS Map 相關導入
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.interpolate import RegularGridInterpolator

# 從 config 導入
from app.core.config import NYCU_GLB_PATH, OUTPUT_DIR  # 確保導入 NYCU_GLB_PATH

logger = logging.getLogger(__name__)

# 嘗試導入 skimage，如果失敗則使用替代方案
try:
    from skimage.feature import peak_local_max
except ImportError:
    try:
        from skimage.feature import peak_local_max as peak_local_maxima
        peak_local_max = peak_local_maxima
    except ImportError:
        logger.warning("Warning: skimage not available, using custom implementation")
        peak_local_max = None

# --- 新增：場景背景顏色常數 ---
SCENE_BACKGROUND_COLOR_RGB = [0.5, 0.5, 0.5]
# --- End Constant ---

# --- 座標轉換工具函數 ---
def to_sionna_coords(p):
    """DB/前端座標 -> Sionna座標 (y 取負)"""
    return [p[0], -p[1], p[2]]

def to_frontend_coords(p):
    """Sionna座標 -> DB/前端座標 (y 還原)"""
    return [p[0], -p[1], p[2]]

def to_sionna_xy_from_frontend(xy: tuple[float, float]) -> tuple[float, float]:
    """(x, y) from DB/Frontend → Sionna (x, -y)"""
    return (xy[0], -xy[1])

def build_iss_interpolator(x_unique: np.ndarray, y_unique: np.ndarray, iss_dbm: np.ndarray):
    """
    建立 2D 內插器，注意 RegularGridInterpolator 的軸順序是 (y, x)
    iss_dbm shape 必須是 (len(y_unique), len(x_unique))
    """
    return RegularGridInterpolator(
        (y_unique, x_unique), iss_dbm, bounds_error=False, fill_value=np.nan
    )

def sample_iss_at_points(
    x_unique: np.ndarray, y_unique: np.ndarray, iss_dbm: np.ndarray,
    pts_frontend_xy: List[tuple[float, float]],
    noise_std_db: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在指定的前端/DB座標點取樣 ISS (dBm)。
    回傳: (xs, ys, vals_dbm) 皆為 Sionna 座標平面上的 x/y 與 dBm
    """
    if len(pts_frontend_xy) == 0:
        return np.array([]), np.array([]), np.array([])

    # 轉成 Sionna 平面座標
    pts_sionna = [to_sionna_xy_from_frontend(p) for p in pts_frontend_xy]
    interp = build_iss_interpolator(x_unique, y_unique, iss_dbm)
    # RegularGridInterpolator 要求點為 (y, x) 順序
    query = np.array([[py, px] for (px, py) in pts_sionna], dtype=np.float64)
    vals = interp(query)

    if noise_std_db > 0:
        vals = vals + np.random.normal(0.0, noise_std_db, size=vals.shape)

    xs = np.array([p[0] for p in pts_sionna])
    ys = np.array([p[1] for p in pts_sionna])
    return xs, ys, vals
# --- End 座標轉換工具函數 ---


# --- 輔助函數：XML 健康度檢查 ---
def check_scene_health(scene_name: str, xml_path: str) -> bool:
    """
    檢查場景的健康度，包括 XML 格式和幾何數據完整性
    返回 True 表示健康，False 表示有問題（將拋出異常）
    """
    try:
        # 檢查 1: XML 文件是否存在
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"場景 {scene_name} 的 XML 文件不存在: {xml_path}")

        # 檢查 2: XML 格式問題 - NTPU 有已知的 shape id 問題
        # if scene_name in ["NTPU"]:
        #     raise ValueError(
        #         f"場景 {scene_name} 的 XML 文件格式不相容於 Sionna"
        #         f"（shape 元素缺少 id 屬性）"
        #     )

        # 檢查 3: 幾何數據完整性 - 檢查 PLY 文件大小
        if scene_name == "Lotus":
            scene_dir = os.path.dirname(xml_path)
            meshes_dir = os.path.join(scene_dir, "meshes")

            if os.path.exists(meshes_dir):
                ply_files = [f for f in os.listdir(meshes_dir) if f.endswith(".ply")]
                total_size = 0
                small_files = 0

                for ply_file in ply_files:
                    ply_path = os.path.join(meshes_dir, ply_file)
                    if os.path.exists(ply_path):
                        size = os.path.getsize(ply_path)
                        total_size += size
                        if size < 2000:  # 小於 2KB 的文件視為不完整
                            small_files += 1

                # 如果總大小太小或太多小文件，視為不健康
                if total_size < 30000 or small_files > len(ply_files) * 0.8:
                    raise ValueError(
                        f"場景 {scene_name} 的幾何數據不完整"
                        f"（總大小: {total_size} bytes，{small_files}/{len(ply_files)} 個小文件）"
                    )

        return True

    except Exception as e:
        logger.error(f"檢查場景 {scene_name} 健康度時出錯: {e}")
        raise


# --- 輔助函數：獲取場景 XML 路徑 ---
def get_scene_xml_file_path(scene_name: str) -> str:
    """
    根據場景名稱獲取對應的 XML 文件路徑，直接返回錯誤而不回退
    """
    # 將前端路由參數映射到後端場景名稱
    scene_mapping = {
        "nycu": "NYCU",
        "lotus": "Lotus",
        "ntpu": "NTPU",
        "nanliao": "nnn",
        "potou": "potou",
        "poto": "poto",
        "testscene": "TestScene",
    }

    backend_scene_name = scene_mapping.get(scene_name.lower(), "NYCU")
    original_scene_name = backend_scene_name

    # 獲取 XML 路徑
    xml_path = get_scene_xml_path(backend_scene_name)

    # 健康度檢查 - 如果失敗會拋出異常
    check_scene_health(original_scene_name, xml_path)

    logger.info(f"場景 '{scene_name}' 映射到 XML 路徑: {xml_path}")
    return str(xml_path)


# --- 通用函數：GPU 設置 ---
def _setup_gpu():
    """設置 GPU 環境，啟用記憶體增長"""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info("GPU 記憶體成長已啟用")
        except Exception as e:
            logger.warning(f"無法啟用GPU記憶體增長: {e}")
    else:
        logger.info("未找到 GPU，使用 CPU")

    return gpus is not None


def _clean_output_file(output_path, file_desc="圖檔"):
    """清理舊的輸出文件"""
    if os.path.exists(output_path):
        logger.info(f"刪除舊的{file_desc}: {output_path}")
        os.remove(output_path)
        return True
    return False


def _ensure_output_dir(output_path):
    """確保輸出目錄存在"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return True


def verify_output_file(output_path):
    """檢查輸出文件是否成功生成，可被外部調用"""
    exists = os.path.exists(output_path)
    size = os.path.getsize(output_path) if exists else -1
    is_file = os.path.isfile(output_path) if exists else False

    if exists and is_file and size > 0:
        logger.info(
            f"SUCCESS: File verified. Path: {output_path}, Size: {size} bytes, IsFile: {is_file}"
        )
        return True
    else:
        logger.error(
            f"FAILURE: File verification failed. Path: {output_path}, Exists: {exists}, Size: {size} bytes, IsFile: {is_file}"
        )
        return False


# 統一的文件準備函數
def prepare_output_file(output_path, file_desc="圖檔"):
    """清理舊文件並準備目錄結構"""
    _clean_output_file(output_path, file_desc)
    _ensure_output_dir(output_path)
    return True


# --- 定義新的資料容器 ---
class DeviceData(BaseModel):
    """用於傳遞設備模型和其處理後的位置列表"""

    device_model: Device = PydanticField(...)  # Store the original SQLModel object
    position_list: List[float] = None  # Store the position as a list [x, y, z]
    orientation_list: List[float] = None  # Store the orientation as a list [x, y, z]
    transmitter_role: Optional[DeviceRole] = (
        None  # Store transmitter type if applicable
    )

    class Config:
        arbitrary_types_allowed = True  # Allow complex types like SQLModel objects


# --- Helper Function for Pyrender Scene Setup ---
def _setup_pyrender_scene_from_glb(scene_name: str = "NYCU") -> Optional[pyrender.Scene]:
    """Loads GLB, sets up pyrender scene, lights, camera. Returns Scene or None on error."""
    from app.core.config import get_scene_model_path
    
    # 根據場景名稱獲取對應的 GLB 文件路徑
    glb_path = get_scene_model_path(scene_name)
    logger.info(f"Setting up base pyrender scene from GLB: {glb_path}")
    try:
        # 1. Load GLB
        if not os.path.exists(glb_path) or os.path.getsize(glb_path) == 0:
            logger.error(f"GLB file not found or empty: {glb_path}")
            return None
        scene_tm = trimesh.load(glb_path, force="scenes")
        logger.info("GLB file loaded.")

        # 2. Create pyrender scene with background and ambient light
        pr_scene = pyrender.Scene(
            bg_color=[*SCENE_BACKGROUND_COLOR_RGB, 1.0],
            ambient_light=[0.6, 0.6, 0.6],
        )

        # 3. Add GLB geometry
        logger.info("Adding GLB geometry...")
        for name, geom in scene_tm.geometry.items():
            if geom.vertices is not None and len(geom.vertices) > 0:
                if (
                    not hasattr(geom, "vertex_normals")
                    or geom.vertex_normals is None
                    or len(geom.vertex_normals) != len(geom.vertices)
                ):
                    if (
                        hasattr(geom, "faces")
                        and geom.faces is not None
                        and len(geom.faces) > 0
                    ):
                        try:
                            geom.compute_vertex_normals()
                        except Exception as norm_err:
                            logger.error(
                                f"Failed compute normals for '{name}': {norm_err}",
                                exc_info=True,
                            )
                            continue
                    else:
                        logger.warning(f"Mesh '{name}' has no faces. Skipping.")
                        continue
                if not hasattr(geom, "visual") or (
                    not hasattr(geom.visual, "vertex_colors")
                    and not hasattr(geom.visual, "material")
                ):
                    geom.visual = trimesh.visual.ColorVisuals(
                        mesh=geom, vertex_colors=[255, 255, 255, 255]
                    )
                try:
                    mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
                    pr_scene.add(mesh)
                except Exception as mesh_err:
                    logger.error(
                        f"Failed convert mesh '{name}': {mesh_err}", exc_info=True
                    )
            else:
                logger.warning(f"Skipping empty mesh '{name}'.")
        logger.info("GLB geometry added.")

        # 4. Add lights
        warm_white = np.array([1.0, 0.98, 0.9])
        main_light = pyrender.DirectionalLight(color=warm_white, intensity=3.0)
        pr_scene.add(main_light, pose=np.eye(4))
        logger.info("Lights added.")

        # 5. Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, znear=0.1, zfar=10000.0)
        cam_pose = np.array(
            [
                [1.0, 0.0, 0.0, 17.0],
                [0.0, 0.0, 1.0, 940.0],
                [0.0, -1.0, 0.0, -19.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        pr_scene.add(camera, pose=cam_pose)
        logger.info("Camera added.")

        return pr_scene

    except Exception as e:
        logger.error(f"Error setting up pyrender scene from GLB: {e}", exc_info=True)
        return None


# --- NEW Helper Function for Rendering, Cropping, and Saving ---
def _render_crop_and_save(
    pr_scene: pyrender.Scene,
    output_path: str,
    bg_color_float: List[float] = SCENE_BACKGROUND_COLOR_RGB,
    render_width: int = 1200,
    render_height: int = 858,
    padding_y: int = 0,  # Default vertical padding
    padding_x: int = 0,  # Default horizontal padding
) -> bool:
    """Renders the scene, crops based on content, and saves the image."""
    logger.info("Starting offscreen rendering...")
    try:
        renderer = pyrender.OffscreenRenderer(render_width, render_height)
        color, _ = renderer.render(pr_scene)

        # 安全地釋放 renderer 資源，捕獲可能的 GLError
        try:
            renderer.delete()
        except Exception as delete_err:
            # 這是已知的 EGL 問題，不影響渲染結果，可以忽略
            pass

        logger.info("Rendering complete.")
    except Exception as render_err:
        logger.error(f"Pyrender OffscreenRenderer failed: {render_err}", exc_info=True)
        return False

    # --- Cropping Logic ---
    logger.info("Calculating bounding box for cropping...")
    image_to_save = color  # Default to original image
    try:
        bg_color_uint8 = (np.array(bg_color_float) * 255).astype(np.uint8)
        mask = ~np.all(color[:, :, :3] == bg_color_uint8, axis=2)
        rows, cols = np.where(mask)

        if rows.size > 0 and cols.size > 0:
            ymin, ymax = rows.min(), rows.max()
            xmin, xmax = cols.min(), cols.max()
            # Apply padding separately
            ymin = max(0, ymin - padding_y)
            xmin = max(0, xmin - padding_x)
            ymax = min(render_height - 1, ymax + padding_y)
            xmax = min(render_width - 1, xmax + padding_x)

            if xmin < xmax and ymin < ymax:
                cropped_color = color[ymin : ymax + 1, xmin : xmax + 1]
                logger.info(
                    f"Cropping image to bounds: (xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax})"
                )
                image_to_save = cropped_color
            else:
                logger.warning(f"Invalid crop bounds (min>=max). Saving original.")
        else:
            logger.warning("No non-background pixels found. Saving original image.")

    except Exception as crop_err:
        logger.error(f"Error during image cropping: {crop_err}", exc_info=True)
        # Fallback to original image

    # --- Save the Image using PIL ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Saving final image to: {output_path}")
    try:
        img = Image.fromarray(image_to_save)
        img.save(output_path, format="PNG")
    except Exception as save_err:
        logger.error(f"Failed to save rendered image: {save_err}", exc_info=True)
        return False

    # Final check
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(
            f"Successfully saved image to {output_path}, size: {os.path.getsize(output_path)} bytes"
        )
        return True
    else:
        logger.error(f"Failed to save image or image is empty: {output_path}")
        return False


# --- Refactor generate_empty_scene_image to use the helpers ---
def generate_empty_scene_image(output_path: str, scene_name: str = "NYCU"):
    """Generates a cropped scene image by rendering the GLB file (using helpers)."""
    logger.info(f"Entering generate_empty_scene_image function, calling helpers...")
    try:
        # 準備輸出檔案
        prepare_output_file(output_path, "空場景圖檔")

        # 1. Setup scene using helper
        pr_scene = _setup_pyrender_scene_from_glb(scene_name)  # Helper uses this bg color
        if pr_scene is None:
            return False

        # 2. Render, Crop, and Save using helper
        success = _render_crop_and_save(
            pr_scene,
            output_path,
            bg_color_float=SCENE_BACKGROUND_COLOR_RGB,
            padding_x=5,  # Set horizontal padding to 5
            padding_y=20,  # Keep vertical padding at 20 (or adjust if needed)
        )
        return success

    except ImportError as ie:
        logger.error(f"Import error in generate_empty_scene_image: {ie}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Error rendering empty scene via helpers: {e}", exc_info=True)
        return False


# 新增函數: generate_cfr_plot
async def generate_cfr_plot(
    session: AsyncSession,
    output_path: str = str(CFR_PLOT_IMAGE_PATH),
    scene_name: str = "nycu",
) -> bool:
    """
    生成 Channel Frequency Response (CFR) 圖，基於 Sionna 的模擬。
    這是從 cfr.py 整合的功能。

    從資料庫中獲取接收器 (receiver)、發射器 (desired) 和干擾器 (jammer) 參數。
    """
    logger.info("Entering generate_cfr_plot function...")

    try:
        # 準備輸出檔案
        prepare_output_file(output_path, "CFR 圖檔")

        # 創建設備服務和儲存庫
        device_repository = SQLModelDeviceRepository(session)
        device_service = DeviceService(device_repository)

        logger.info("Fetching active receivers from database...")
        active_receivers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.RECEIVER.value, active_only=True
        )

        if not active_receivers:
            logger.warning(
                "No active receivers found in database. Using default receiver parameters."
            )
            # 使用默認的接收器參數
            rx_name = "rx"
            rx_position = [0, 0, 20]
        else:
            # 使用第一個活動接收器的參數
            receiver = active_receivers[0]
            rx_name = receiver.name
            rx_position = [
                receiver.position_x,
                receiver.position_y,
                receiver.position_z,
            ]
            logger.info(f"Using receiver '{rx_name}' with position {rx_position}")

        # 從資料庫獲取活動的發射器 (desired)
        logger.info("Fetching active desired transmitters from database...")
        active_desired = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.DESIRED.value, active_only=True
        )

        # 從資料庫獲取活動的干擾器 (jammer)
        logger.info("Fetching active jammers from database...")
        active_jammers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.JAMMER.value, active_only=True
        )

        # 構建 TX_LIST (發射器和干擾器列表)
        TX_LIST = []

        # 添加發射器
        if not active_desired:
            logger.warning(
                "No active desired transmitters found in database. Simulation might not be meaningful."
            )
            # 添加默認的發射器參數 # REMOVED
        else:
            # 添加從資料庫獲取的發射器
            for i, tx in enumerate(active_desired):
                tx_name = tx.name
                tx_position = [tx.position_x, tx.position_y, tx.position_z]
                tx_orientation = [tx.orientation_x, tx.orientation_y, tx.orientation_z]
                tx_power = tx.power_dbm

                TX_LIST.append(
                    (tx_name, tx_position, tx_orientation, "desired", tx_power)
                )
                logger.info(
                    f"Added desired transmitter: {tx_name}, position: {tx_position}, orientation: {tx_orientation}, power: {tx_power} dBm"
                )

        # 添加干擾器
        if not active_jammers:
            logger.warning(
                "No active jammers found in database. Interference simulation will not run."
            )
            # 添加默認的干擾器參數 # REMOVED
        else:
            # 添加從資料庫獲取的干擾器
            for i, jammer in enumerate(active_jammers):
                jammer_name = jammer.name
                jammer_position = [
                    jammer.position_x,
                    jammer.position_y,
                    jammer.position_z,
                ]
                jammer_orientation = [
                    jammer.orientation_x,
                    jammer.orientation_y,
                    jammer.orientation_z,
                ]
                jammer_power = jammer.power_dbm

                TX_LIST.append(
                    (
                        jammer_name,
                        jammer_position,
                        jammer_orientation,
                        "jammer",
                        jammer_power,
                    )
                )
                logger.info(
                    f"Added jammer: {jammer_name}, position: {jammer_position}, orientation: {jammer_orientation}, power: {jammer_power} dBm"
                )

        # 檢查是否有足夠的發射器和干擾器
        if not TX_LIST:
            logger.error(
                "No transmitters or jammers available for simulation. Cannot proceed."
            )
            return False

        # 參數設置
        SCENE_NAME = get_scene_xml_file_path(scene_name)
        logger.info(f"Loading scene from: {SCENE_NAME}")

        TX_ARRAY_CONFIG = {
            "num_rows": 1,
            "num_cols": 1,
            "vertical_spacing": 0.5,
            "horizontal_spacing": 0.5,
            "pattern": "iso",
            "polarization": "V",
        }
        RX_ARRAY_CONFIG = TX_ARRAY_CONFIG

        # 使用從資料庫獲取的接收器參數
        RX_CONFIG = (rx_name, rx_position)

        PATHSOLVER_ARGS = {
            "max_depth": 10,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": False,
            "synthetic_array": False,
            "seed": 41,
        }

        N_SYMBOLS = 1
        N_SUBCARRIERS = 1024
        SUBCARRIER_SPACING = 30e3
        EBN0_dB = 20.0

        # 場景設置
        logger.info("Setting up scene")
        scene = load_scene(SCENE_NAME)
        scene.tx_array = PlanarArray(**TX_ARRAY_CONFIG)
        scene.rx_array = PlanarArray(**RX_ARRAY_CONFIG)

        # 清除現有的發射器和接收器
        for name in list(scene.transmitters.keys()) + list(scene.receivers.keys()):
            scene.remove(name)

        # 添加發射器
        logger.info("Adding transmitters")

        def add_tx(scene, name, pos, ori, role, power_dbm):
            tx = SionnaTransmitter(
                name=name, position=to_sionna_coords(pos), orientation=ori, power_dbm=power_dbm
            )
            tx.role = role
            scene.add(tx)
            return tx

        for name, pos, ori, role, p_dbm in TX_LIST:
            add_tx(scene, name, pos, ori, role, p_dbm)

        # 添加接收器
        logger.info(f"Adding receiver '{rx_name}' at position {rx_position}")
        rx_name, rx_pos = RX_CONFIG
        scene.add(SionnaReceiver(name=rx_name, position=to_sionna_coords(rx_pos)))

        # 分組發射器
        tx_names = list(scene.transmitters.keys())
        all_txs = [scene.get(n) for n in tx_names]
        idx_des = [i for i, tx in enumerate(all_txs) if tx.role == "desired"]
        idx_jam = [i for i, tx in enumerate(all_txs) if tx.role == "jammer"]

        # 檢查是否有發射器和干擾器
        if not idx_des:
            logger.warning(
                "No desired transmitters available in scene. CFR calculation may not be accurate."
            )
        if not idx_jam:
            logger.warning(
                "No jammers available in scene. Interference will not be present in plot."
            )

        # 計算 CFR
        logger.info("Computing CFR")
        freqs = subcarrier_frequencies(N_SUBCARRIERS, SUBCARRIER_SPACING)
        for name in tx_names:
            scene.get(name).velocity = [30, 0, 0]
        paths = PathSolver()(scene, **PATHSOLVER_ARGS)

        def dbm2w(dbm):
            return 10 ** (dbm / 10) / 1000

        tx_powers = [dbm2w(scene.get(n).power_dbm) for n in tx_names]
        ofdm_symbol_duration = 1 / SUBCARRIER_SPACING
        H_unit = paths.cfr(
            frequencies=freqs,
            sampling_frequency=1 / ofdm_symbol_duration,
            num_time_steps=N_SUBCARRIERS,
            normalize_delays=True,
            normalize=False,
            out_type="numpy",
        ).squeeze()  # shape: (num_tx, T, F)

        H_all = np.sqrt(np.array(tx_powers)[:, None, None]) * H_unit
        H = H_unit[:, 0, :]  # 取第一個時間步

        # 安全處理：確保有所需的發射器
        h_main = np.zeros(N_SUBCARRIERS, dtype=complex)
        if idx_des:
            h_main = sum(np.sqrt(tx_powers[i]) * H[i] for i in idx_des)

        h_intf = np.zeros(N_SUBCARRIERS, dtype=complex)
        if idx_jam:
            h_intf = sum(np.sqrt(tx_powers[i]) * H[i] for i in idx_jam)

        # 生成 QPSK+OFDM 符號
        logger.info("Generating QPSK+OFDM symbols")
        bits = np.random.randint(0, 2, (N_SYMBOLS, N_SUBCARRIERS, 2))
        bits_jam = np.random.randint(0, 2, (N_SYMBOLS, N_SUBCARRIERS, 2))
        X_sig = (1 - 2 * bits[..., 0] + 1j * (1 - 2 * bits[..., 1])) / np.sqrt(2)
        X_jam = (1 - 2 * bits_jam[..., 0] + 1j * (1 - 2 * bits_jam[..., 1])) / np.sqrt(
            2
        )

        Y_sig = X_sig * h_main[None, :]
        Y_int = X_jam * h_intf[None, :]
        p_sig = np.mean(np.abs(Y_sig) ** 2)
        N0 = p_sig / (10 ** (EBN0_dB / 10) * 2) if p_sig > 0 else 1e-10
        noise = np.sqrt(N0 / 2) * (
            np.random.randn(*Y_sig.shape) + 1j * np.random.randn(*Y_sig.shape)
        )
        Y_tot = Y_sig + Y_int + noise

        # 安全處理：避免除以零
        non_zero_mask = np.abs(h_main) > 1e-10
        y_eq_no_i = np.zeros_like(Y_sig)
        y_eq_with_i = np.zeros_like(Y_tot)

        if np.any(non_zero_mask):
            y_eq_no_i[:, non_zero_mask] = (Y_sig + noise)[:, non_zero_mask] / h_main[
                None, non_zero_mask
            ]
            y_eq_with_i[:, non_zero_mask] = (
                Y_tot[:, non_zero_mask] / h_main[None, non_zero_mask]
            )

        # 繪製星座圖和 CFR，然後保存到文件
        logger.info("Plotting constellation and CFR")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].scatter(y_eq_no_i.real, y_eq_no_i.imag, s=4, alpha=0.25)
        ax[0].set(title="No interference", xlabel="Real", ylabel="Imag")
        ax[0].grid(True)

        ax[1].scatter(y_eq_with_i.real, y_eq_with_i.imag, s=4, alpha=0.25)
        ax[1].set(title="With interferer", xlabel="Real", ylabel="Imag")
        ax[1].grid(True)

        ax[2].plot(np.abs(h_main), label="|H_main|")
        ax[2].plot(np.abs(h_intf), label="|H_intf|")
        ax[2].set(title="CFR Magnitude", xlabel="Subcarrier Index")
        ax[2].legend()
        ax[2].grid(True)

        plt.tight_layout()

        # 保存圖片
        logger.info(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 檢查文件是否成功生成
        return verify_output_file(output_path)

    except Exception as e:
        logger.exception(f"Error in generate_cfr_plot: {e}")
        # 確保關閉所有打開的圖表
        plt.close("all")
        return False


# 新增 SINR Map 生成函數
async def generate_radio_map(
    session: AsyncSession,
    output_path: str,
    scene_name: str = "nycu",
    sinr_vmin: float = -40,
    sinr_vmax: float = 0,
    cell_size: float = 1.0,
    samples_per_tx: int = 10**7,
    exclude_jammers: bool = True,
    center_on_transmitter: bool = True,
) -> bool:
    """
    生成無線電地圖 (可選擇排除干擾源並以發射器為中心)
    
    從數據庫獲取發射器設置，計算並生成無線電地圖
    """
    logger.info(f"開始生成無線電地圖... exclude_jammers={exclude_jammers}, center_on_transmitter={center_on_transmitter}")

    try:
        # 準備輸出檔案
        prepare_output_file(output_path, "無線電地圖圖檔")

        # GPU 設置
        gpus = _setup_gpu()

        # 創建設備服務和儲存庫
        device_repository = SQLModelDeviceRepository(session)
        device_service = DeviceService(device_repository)

        # 從數據庫獲取活動的發射器 (desired)
        logger.info("從數據庫獲取活動的發射器...")
        active_desired = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.DESIRED.value, active_only=True
        )

        # 從數據庫獲取活動的接收器
        logger.info("從數據庫獲取活動的接收器...")
        active_receivers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.RECEIVER.value, active_only=True
        )

        # 檢查是否有足夠的設備
        if not active_desired:
            logger.error("沒有活動的發射器，無法生成無線電地圖")
            return False

        if not active_receivers:
            logger.warning("沒有活動的接收器，將使用預設接收器位置")
            rx_config = ("rx", [-30, 50, 20])
        else:
            # 使用第一個活動接收器
            receiver = active_receivers[0]
            rx_config = (
                receiver.name,
                [receiver.position_x, receiver.position_y, receiver.position_z],
            )

        # 構建 TX_LIST (只包含發射器，不包含干擾器)
        tx_list = []

        # 添加發射器
        for tx in active_desired:
            tx_name = tx.name
            tx_position = [tx.position_x, tx.position_y, tx.position_z]
            tx_orientation = [tx.orientation_x, tx.orientation_y, tx.orientation_z]
            tx_power = tx.power_dbm

            tx_list.append((tx_name, tx_position, tx_orientation, "desired", tx_power))
            logger.info(
                f"添加發射器: {tx_name}, 位置: {tx_position}, 方向: {tx_orientation}, 功率: {tx_power} dBm"
            )

        # 如果不排除干擾器，則添加它們
        if not exclude_jammers:
            logger.info("從數據庫獲取活動的干擾器...")
            active_jammers = await device_service.get_devices(
                skip=0, limit=100, role=DeviceRole.JAMMER.value, active_only=True
            )
            
            # 添加干擾器
            for jammer in active_jammers:
                jammer_name = jammer.name
                jammer_position = [jammer.position_x, jammer.position_y, jammer.position_z]
                jammer_orientation = [
                    jammer.orientation_x,
                    jammer.orientation_y,
                    jammer.orientation_z,
                ]
                jammer_power = jammer.power_dbm

                tx_list.append(
                    (
                        jammer_name,
                        jammer_position,
                        jammer_orientation,
                        "jammer",
                        jammer_power,
                    )
                )
                logger.info(
                    f"添加干擾器: {jammer_name}, 位置: {jammer_position}, 方向: {jammer_orientation}, 功率: {jammer_power} dBm"
                )

        # 如果沒有足夠的發射器，返回錯誤
        if not tx_list:
            logger.error("沒有可用的發射器，無法生成無線電地圖")
            return False

        # 如果要以發射器為中心，調整地圖區域
        map_center = None
        if center_on_transmitter and active_desired:
            first_tx = active_desired[0]
            map_center = to_sionna_coords([first_tx.position_x, first_tx.position_y, first_tx.position_z])[:2]
            logger.info(f"地圖將以發射器為中心: {map_center}")

        # 參數設置
        scene_xml_path = get_scene_xml_file_path(scene_name)
        logger.info(f"從 {scene_xml_path} 加載場景")

        tx_array_config = {
            "num_rows": 1,
            "num_cols": 1,
            "vertical_spacing": 0.5,
            "horizontal_spacing": 0.5,
            "pattern": "iso",
            "polarization": "V",
        }
        rx_array_config = tx_array_config

        rmsolver_args = {
            "max_depth": 10,
            "cell_size": (cell_size, cell_size),
            "samples_per_tx": samples_per_tx,
        }

        # 場景設置
        logger.info("設置場景")
        scene = load_scene(scene_xml_path)
        scene.tx_array = PlanarArray(**tx_array_config)
        scene.rx_array = PlanarArray(**rx_array_config)

        # 清除現有的發射器和接收器
        for name in list(scene.transmitters.keys()) + list(scene.receivers.keys()):
            scene.remove(name)

        # 添加發射器
        logger.info("添加發射器")

        def add_tx(scene, name, pos, ori, role, power_dbm):
            tx = SionnaTransmitter(
                name=name, position=to_sionna_coords(pos), orientation=ori, power_dbm=power_dbm
            )
            tx.role = role
            scene.add(tx)
            return tx

        for name, pos, ori, role, p_dbm in tx_list:
            add_tx(scene, name, pos, ori, role, p_dbm)

        # 添加接收器
        rx_name, rx_pos = rx_config
        logger.info(f"添加接收器 '{rx_name}' 在位置 {rx_pos}")
        scene.add(SionnaReceiver(name=rx_name, position=to_sionna_coords(rx_pos)))

        # 按角色分組發射器
        all_txs = [scene.get(n) for n in scene.transmitters]
        idx_des = [
            i for i, tx in enumerate(all_txs) if getattr(tx, "role", None) == "desired"
        ]
        idx_jam = [
            i for i, tx in enumerate(all_txs) if getattr(tx, "role", None) == "jammer"
        ]

        if not idx_des:
            logger.error("場景中沒有有效的發射器")
            return False

        # 計算無線電地圖
        logger.info("計算無線電地圖")
        rm_solver = RadioMapSolver()
        rm = rm_solver(scene, **rmsolver_args)

        # 計算並繪製無線電地圖 (不含干擾的RSS)
        logger.info("計算無線電地圖 (不含干擾)")
        cc = rm.cell_centers.numpy()
        x_unique = cc[0, :, 0]
        y_unique = cc[:, 0, 1]
        rss_list = [rm.rss[i].numpy() for i in range(len(all_txs))]

        # 只使用目標發射器的RSS
        rss_clean = sum(rss_list[i] for i in idx_des)

        # 轉換為 dB
        rss_clean_db = 10 * np.log10(rss_clean + 1e-12)

        # 生成圖表
        logger.info("生成無線電地圖圖表")
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 繪製無線電地圖
        im = ax.contourf(
            x_unique,
            y_unique,
            rss_clean_db,
            levels=np.linspace(sinr_vmin, sinr_vmax, 20),
            cmap="viridis",
            extend="both",
        )

        # 標記發射器和接收器位置
        for tx in active_desired:
            ax.scatter(
                tx.position_x,
                tx.position_y,
                color="red",
                s=100,
                marker="^",
                edgecolors="white",
                label=f"TX: {tx.name}",
            )

        if active_receivers:
            for rx in active_receivers:
                ax.scatter(
                    rx.position_x,
                    rx.position_y,
                    color="blue",
                    s=100,
                    marker="o",
                    edgecolors="white",
                    label=f"RX: {rx.name}",
                )

        # 設置地圖範圍
        if center_on_transmitter and map_center:
            # 以發射器為中心設置範圍
            range_size = 100  # 100m範圍
            ax.set_xlim([map_center[0] - range_size, map_center[0] + range_size])
            ax.set_ylim([map_center[1] - range_size, map_center[1] + range_size])

        ax.set_xlabel("X座標 (m)")
        ax.set_ylabel("Y座標 (m)")
        ax.set_title(f"無線電地圖 (不含干擾源) - {scene_name.upper()}")
        
        # 添加顏色條
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("接收信號強度 (dB)")

        # 添加圖例
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"無線電地圖已保存到: {output_path}")
        return True

    except Exception as e:
        logger.error(f"生成無線電地圖時發生錯誤: {e}")
        logger.error(traceback.format_exc())
        return False


async def generate_sinr_map(
    session: AsyncSession,
    output_path: str = str(SINR_MAP_IMAGE_PATH),
    scene_name: str = "nycu",
    sinr_vmin: float = -40,
    sinr_vmax: float = 0,
    cell_size: float = 1.0,
    samples_per_tx: int = 10**7,
) -> bool:
    """
    生成 SINR (Signal-to-Interference-plus-Noise Ratio) 地圖

    從數據庫獲取發射器和接收器設置，計算並生成 SINR 地圖
    """
    logger.info("開始生成 SINR 地圖...")

    try:
        # 準備輸出檔案
        prepare_output_file(output_path, "SINR 地圖圖檔")

        # GPU 設置
        gpus = _setup_gpu()

        # 創建設備服務和儲存庫
        device_repository = SQLModelDeviceRepository(session)
        device_service = DeviceService(device_repository)

        # 從數據庫獲取活動的發射器 (desired)
        logger.info("從數據庫獲取活動的發射器...")
        active_desired = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.DESIRED.value, active_only=True
        )

        # 從數據庫獲取活動的干擾器 (jammer)
        logger.info("從數據庫獲取活動的干擾器...")
        active_jammers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.JAMMER.value, active_only=True
        )

        # 從數據庫獲取活動的接收器
        logger.info("從數據庫獲取活動的接收器...")
        active_receivers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.RECEIVER.value, active_only=True
        )

        # 檢查是否有足夠的設備
        if not active_desired and not active_jammers:
            logger.error("沒有活動的發射器或干擾器，無法生成 SINR 地圖")
            return False

        if not active_receivers:
            logger.warning("沒有活動的接收器，將使用預設接收器位置")
            rx_config = ("rx", [-30, 50, 20])
        else:
            # 使用第一個活動接收器
            receiver = active_receivers[0]
            rx_config = (
                receiver.name,
                [receiver.position_x, receiver.position_y, receiver.position_z],
            )

        # 構建 TX_LIST
        tx_list = []

        # 添加發射器
        for tx in active_desired:
            tx_name = tx.name
            tx_position = [tx.position_x, tx.position_y, tx.position_z]
            tx_orientation = [tx.orientation_x, tx.orientation_y, tx.orientation_z]
            tx_power = tx.power_dbm

            tx_list.append((tx_name, tx_position, tx_orientation, "desired", tx_power))
            logger.info(
                f"添加發射器: {tx_name}, 位置: {tx_position}, 方向: {tx_orientation}, 功率: {tx_power} dBm"
            )

        # 添加干擾器
        for jammer in active_jammers:
            jammer_name = jammer.name
            jammer_position = [jammer.position_x, jammer.position_y, jammer.position_z]
            jammer_orientation = [
                jammer.orientation_x,
                jammer.orientation_y,
                jammer.orientation_z,
            ]
            jammer_power = jammer.power_dbm

            tx_list.append(
                (
                    jammer_name,
                    jammer_position,
                    jammer_orientation,
                    "jammer",
                    jammer_power,
                )
            )
            logger.info(
                f"添加干擾器: {jammer_name}, 位置: {jammer_position}, 方向: {jammer_orientation}, 功率: {jammer_power} dBm"
            )

        # 如果沒有足夠的發射器，返回錯誤
        if not tx_list:
            logger.error("沒有可用的發射器或干擾器，無法生成 SINR 地圖")
            return False

        # 參數設置
        scene_xml_path = get_scene_xml_file_path(scene_name)
        logger.info(f"從 {scene_xml_path} 加載場景")

        tx_array_config = {
            "num_rows": 1,
            "num_cols": 1,
            "vertical_spacing": 0.5,
            "horizontal_spacing": 0.5,
            "pattern": "iso",
            "polarization": "V",
        }
        rx_array_config = tx_array_config

        rmsolver_args = {
            "max_depth": 10,
            "cell_size": (cell_size, cell_size),
            "samples_per_tx": samples_per_tx,
        }

        # 場景設置
        logger.info("設置場景")
        scene = load_scene(scene_xml_path)
        scene.tx_array = PlanarArray(**tx_array_config)
        scene.rx_array = PlanarArray(**rx_array_config)

        # 清除現有的發射器和接收器
        for name in list(scene.transmitters.keys()) + list(scene.receivers.keys()):
            scene.remove(name)

        # 添加發射器
        logger.info("添加發射器")

        def add_tx(scene, name, pos, ori, role, power_dbm):
            tx = SionnaTransmitter(
                name=name, position=to_sionna_coords(pos), orientation=ori, power_dbm=power_dbm
            )
            tx.role = role
            scene.add(tx)
            return tx

        for name, pos, ori, role, p_dbm in tx_list:
            add_tx(scene, name, pos, ori, role, p_dbm)

        # 添加接收器
        rx_name, rx_pos = rx_config
        logger.info(f"添加接收器 '{rx_name}' 在位置 {rx_pos}")
        scene.add(SionnaReceiver(name=rx_name, position=to_sionna_coords(rx_pos)))

        # 按角色分組發射器
        all_txs = [scene.get(n) for n in scene.transmitters]
        idx_des = [
            i for i, tx in enumerate(all_txs) if getattr(tx, "role", None) == "desired"
        ]
        idx_jam = [
            i for i, tx in enumerate(all_txs) if getattr(tx, "role", None) == "jammer"
        ]

        if not idx_des and not idx_jam:
            logger.error("場景中沒有有效的發射器或干擾器")
            return False

        # 計算無線電地圖
        logger.info("計算無線電地圖")
        rm_solver = RadioMapSolver()
        rm = rm_solver(scene, **rmsolver_args)

        # 計算並繪製 SINR 地圖
        logger.info("計算 SINR 地圖")
        cc = rm.cell_centers.numpy()
        x_unique = cc[0, :, 0]
        y_unique = cc[:, 0, 1]
        rss_list = [rm.rss[i].numpy() for i in range(len(all_txs))]

        # 計算 SINR
        N0_map = 1e-12  # 噪聲功率

        # 檢查是否有目標發射器和干擾器
        if idx_des:
            rss_des = sum(rss_list[i] for i in idx_des)
        else:
            logger.warning("沒有目標發射器，將假設沒有信號")
            rss_des = (
                np.zeros_like(rss_list[0])
                if rss_list
                else np.zeros((len(y_unique), len(x_unique)))
            )

        if idx_jam:
            rss_jam = sum(rss_list[i] for i in idx_jam)
        else:
            logger.warning("沒有干擾器，將假設沒有干擾")
            rss_jam = (
                np.zeros_like(rss_list[0])
                if rss_list
                else np.zeros((len(y_unique), len(x_unique)))
            )

        # 計算 SINR (dB)，確保公式與原始 sinr.py 一致
        sinr_db = 10 * np.log10(
            np.clip(rss_des / (rss_des + rss_jam + N0_map), 1e-12, None)
        )

        # 繪製地圖
        logger.info("繪製 SINR 地圖")
        fig, ax = plt.subplots(figsize=(7, 5))
        X, Y = np.meshgrid(x_unique, y_unique)
        pcm = ax.pcolormesh(
            X, Y, sinr_db, shading="nearest", vmin=sinr_vmin + 10, vmax=sinr_vmax
        )
        fig.colorbar(pcm, ax=ax, label="SINR (dB)")

        # 繪製發射器和接收器
        ax.scatter(
            [t.position[0] for t in all_txs if getattr(t, "role", None) == "desired"],
            [t.position[1] for t in all_txs if getattr(t, "role", None) == "desired"],
            c="red",
            marker="^",
            s=100,
            label="Tx",
        )
        ax.scatter(
            [t.position[0] for t in all_txs if getattr(t, "role", None) == "jammer"],
            [t.position[1] for t in all_txs if getattr(t, "role", None) == "jammer"],
            c="red",
            marker="x",
            s=100,
            label="Jam",
        )

        # 獲取接收器
        rx_object = scene.get(rx_name)
        if rx_object:
            ax.scatter(
                rx_object.position[0],
                rx_object.position[1],
                c="green",
                marker="o",
                s=50,
                label="Rx",
            )

        ax.legend()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("SINR Map")
        # Removed ax.invert_yaxis() to match frontend coordinate system
        plt.tight_layout()

        # 保存圖片
        logger.info(f"保存 SINR 地圖到 {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 檢查文件是否生成成功
        return verify_output_file(output_path)

    except Exception as e:
        logger.exception(f"生成 SINR 地圖時發生錯誤: {e}")
        # 確保關閉所有打開的圖表
        plt.close("all")
        return False


# 新增 Doppler 圖生成函數
async def generate_doppler_plots(
    session: AsyncSession,
    output_path: str = str(DOPPLER_IMAGE_PATH),
    scene_name: str = "nycu",
) -> bool:
    """
    生成延遲多普勒圖 (Delay-Doppler)，基於 delay-doppler-v2.py 的功能

    從數據庫中獲取發射器、接收器和干擾器參數，生成統一的延遲多普勒圖
    """
    logger.info("開始生成延遲多普勒圖...")

    try:
        # 準備輸出檔案
        prepare_output_file(output_path, "延遲多普勒圖檔")

        # 設置 GPU
        _setup_gpu()

        # 創建設備服務和儲存庫
        device_repository = SQLModelDeviceRepository(session)
        device_service = DeviceService(device_repository)

        # 從資料庫獲取活動的發射器 (desired)
        logger.info("從數據庫獲取活動的發射器...")
        active_desired = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.DESIRED.value, active_only=True
        )

        # 從資料庫獲取活動的干擾器 (jammer)
        logger.info("從數據庫獲取活動的干擾器...")
        active_jammers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.JAMMER.value, active_only=True
        )

        # 從資料庫獲取活動的接收器
        logger.info("從數據庫獲取活動的接收器...")
        active_receivers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.RECEIVER.value, active_only=True
        )

        # 構建 TX_LIST
        tx_list = []

        # 添加發射器
        for tx in active_desired:
            tx_name = tx.name
            tx_position = [tx.position_x, tx.position_y, tx.position_z]
            tx_orientation = [tx.orientation_x, tx.orientation_y, tx.orientation_z]
            tx_power = tx.power_dbm

            tx_list.append((tx_name, tx_position, tx_orientation, "desired", tx_power))
            logger.info(
                f"添加發射器: {tx_name}, 位置: {tx_position}, 方向: {tx_orientation}, 功率: {tx_power} dBm"
            )

        # 添加干擾器
        for jammer in active_jammers:
            jammer_name = jammer.name
            jammer_position = [jammer.position_x, jammer.position_y, jammer.position_z]
            jammer_orientation = [
                jammer.orientation_x,
                jammer.orientation_y,
                jammer.orientation_z,
            ]
            jammer_power = jammer.power_dbm

            tx_list.append(
                (
                    jammer_name,
                    jammer_position,
                    jammer_orientation,
                    "jammer",
                    jammer_power,
                )
            )
            logger.info(
                f"添加干擾器: {jammer_name}, 位置: {jammer_position}, 方向: {jammer_orientation}, 功率: {jammer_power} dBm"
            )

        # 設置接收器
        if not active_receivers:
            logger.warning("沒有找到活動的接收器，使用默認位置")
            rx_config = ("rx", [0, 0, 40])
        else:
            # 使用第一個活動接收器
            receiver = active_receivers[0]
            rx_config = (
                receiver.name,
                [receiver.position_x, receiver.position_y, receiver.position_z],
            )
            logger.info(f"使用接收器 '{rx_config[0]}' 在位置 {rx_config[1]}")

        # -------- 以下為參考 delay-doppler-v2.py 的邏輯 --------

        # 參數設定
        TX_ARRAY_CONFIG = dict(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        RX_ARRAY_CONFIG = TX_ARRAY_CONFIG

        # 如果沒有設備，回傳錯誤
        if not tx_list:
            logger.error("沒有活動的發射器或干擾器，無法生成延遲多普勒圖")
            return False

        PATHSOLVER_ARGS = dict(
            max_depth=3,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            synthetic_array=False,
            seed=41,
        )

        # OFDM 參數
        N_SUBCARRIERS = 1024
        SUBCARRIER_SPACING = 30e3
        num_ofdm_symbols = 1024

        # 建立場景與天線配置
        scene_xml_path = get_scene_xml_file_path(scene_name)
        logger.info(f"從 {scene_xml_path} 加載場景")
        scene = load_scene(scene_xml_path)
        scene.tx_array = PlanarArray(**TX_ARRAY_CONFIG)
        scene.rx_array = PlanarArray(**RX_ARRAY_CONFIG)

        # 移除現有發射機和接收機
        for tx_name in list(scene.transmitters.keys()):
            scene.remove(tx_name)
        for rx_name in list(scene.receivers.keys()):
            scene.remove(rx_name)

        # 確認清空
        if len(scene.transmitters) > 0 or len(scene.receivers) > 0:
            logger.warning("無法完全清空場景中的發射機和接收機")

        # 新增發射機
        def add_tx(scene, name, pos, ori, role, power_dbm):
            tx = SionnaTransmitter(
                name=name, position=to_sionna_coords(pos), orientation=ori, power_dbm=power_dbm
            )
            tx.role = role
            scene.add(tx)
            return tx

        for name, pos, ori, role, p_dbm in tx_list:
            add_tx(scene, name, pos, ori, role, p_dbm)

        # 新增接收機
        rx_name, rx_pos = rx_config
        logger.info(f"添加接收器 '{rx_name}' 在位置 {rx_pos}")
        scene.add(SionnaReceiver(name=rx_name, position=to_sionna_coords(rx_pos)))

        # 分組索引
        tx_names = list(scene.transmitters.keys())
        all_txs = [scene.get(n) for n in tx_names]
        idx_des = [
            i for i, tx in enumerate(all_txs) if getattr(tx, "role", None) == "desired"
        ]
        idx_jam = [
            i for i, tx in enumerate(all_txs) if getattr(tx, "role", None) == "jammer"
        ]

        # 計算 CFR
        logger.info("計算 CFR")
        freqs = subcarrier_frequencies(N_SUBCARRIERS, SUBCARRIER_SPACING)
        for name in scene.transmitters:
            scene.get(name).velocity = [30, 0, 0]

        # 使用 PathSolver
        solver = PathSolver()
        try:
            paths = solver(scene, **PATHSOLVER_ARGS)
        except RuntimeError as e:
            logger.error(f"PathSolver 错误: {e}")
            logger.error(
                "嘗試減少 max_depth, max_num_paths_per_src, 或使用更簡單的場景。"
            )
            return False

        ofdm_symbol_duration = 1 / SUBCARRIER_SPACING
        delay_resolution = ofdm_symbol_duration / N_SUBCARRIERS
        doppler_resolution = SUBCARRIER_SPACING / num_ofdm_symbols

        # 計算 CFR
        H_unit = paths.cfr(
            frequencies=freqs,
            sampling_frequency=1 / ofdm_symbol_duration,
            num_time_steps=num_ofdm_symbols,
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        ).squeeze()

        # 處理功率加權
        tx_p_lin = 10 ** (np.array([tx.power_dbm for tx in all_txs]) / 10) / 1e3
        tx_p_lin = np.squeeze(tx_p_lin)
        sqrtP = np.sqrt(tx_p_lin)[:, None, None]
        H_unit = H_unit * sqrtP

        # 計算 Delay-Doppler 圖
        def to_delay_doppler(H_tf):
            Hf = np.fft.fftshift(H_tf, axes=1)
            h_delay = np.fft.ifft(Hf, axis=1, norm="ortho")
            h_dd = np.fft.fft(h_delay, axis=0, norm="ortho")
            h_dd = np.fft.fftshift(h_dd, axes=0)
            return h_dd

        # 計算每個發射機的延遲多普勒圖
        Hdd_list = [np.abs(to_delay_doppler(H_unit[i])) for i in range(H_unit.shape[0])]

        # 動態組合網格
        grids = []
        labels = []
        doppler_bins = np.arange(
            -num_ofdm_symbols / 2 * doppler_resolution,
            num_ofdm_symbols / 2 * doppler_resolution,
            doppler_resolution,
        )
        delay_bins = (
            np.arange(0, N_SUBCARRIERS * delay_resolution, delay_resolution) / 1e-9
        )
        x, y = np.meshgrid(delay_bins, doppler_bins)

        offset = 20
        x_start = int(N_SUBCARRIERS / 2) - offset
        x_end = int(N_SUBCARRIERS / 2) + offset
        y_start = 0
        y_end = offset
        x_grid = x[x_start:x_end, y_start:y_end]
        y_grid = y[x_start:x_end, y_start:y_end]

        # Desired 個別 - 使用原始索引 i 而非 k+1
        for k, i in enumerate(idx_des):
            Zi = Hdd_list[i][x_start:x_end, y_start:y_end]
            grids.append(Zi)
            labels.append(f"Des Tx{i}")  # 使用 i 而非 k+1

        # Jammer 個別 - 使用原始索引 i 而非 k+1
        for k, i in enumerate(idx_jam):
            Zi = Hdd_list[i][x_start:x_end, y_start:y_end]
            grids.append(Zi)
            labels.append(f"Jam Tx{i}")  # 使用 i 而非 k+1

        # Desired All
        if idx_des:
            Z_des_all = np.sum([Hdd_list[i] for i in idx_des], axis=0)
            grids.append(Z_des_all[x_start:x_end, y_start:y_end])
            labels.append("Des ALL")

        # Jammer All
        if idx_jam:
            Z_jam_all = np.sum([Hdd_list[i] for i in idx_jam], axis=0)
            grids.append(Z_jam_all[x_start:x_end, y_start:y_end])
            labels.append("Jam ALL")

        # All Tx
        Z_all = np.sum(Hdd_list, axis=0)
        grids.append(Z_all[x_start:x_end, y_start:y_end])
        labels.append("ALL Tx")

        # 統一 Z 軸
        z_min = 0
        z_max = max(g.max() for g in grids) * 1.05

        # 自動排版
        n_plots = len(grids)
        cols = 3
        rows = int(np.ceil(n_plots / cols))

        # 調整圖像大小使其擴展到容器寬度 - 使用與原始相同的圖像大小計算
        figsize = (cols * 4.5, rows * 4.5)

        # 繪製單一的統一圖
        logger.info(f"繪製統一的延遲多普勒圖")
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Delay-Doppler Plots")  # 標題使用原始設置

        for idx, (Z, label) in enumerate(zip(grids, labels), start=1):
            ax = fig.add_subplot(rows, cols, idx, projection="3d")
            # 使用與原始相同的色彩映射 viridis
            ax.plot_surface(x_grid, y_grid, Z, cmap="viridis", edgecolor="none")
            ax.set_title(f"Delay–Doppler |{label}|", pad=8)
            ax.set_xlabel("Delay (ns)")
            ax.set_ylabel("Doppler (Hz)")
            ax.set_zlabel("|H|")
            ax.set_zlim(z_min, z_max)
            # 移除自定義視角設置，使用默認視角

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 檢查文件是否生成成功
        return verify_output_file(output_path)

    except Exception as e:
        logger.exception(f"生成延遲多普勒圖時發生錯誤: {e}")
        # 確保關閉所有打開的圖表
        plt.close("all")
        return False


# 新增函數: 整合 tf.py 的通道響應圖功能
async def generate_channel_response_plots(
    session: AsyncSession,
    output_path: str = str(CHANNEL_RESPONSE_IMAGE_PATH),
    scene_name: str = "nycu",
) -> bool:
    """
    生成通道響應圖 (H_des, H_jam, H_all)，基於 tf.py 中的功能。
    從資料庫獲取接收器、發射器和干擾器參數。
    """
    logger.info("開始生成通道響應圖...")

    try:
        # 準備輸出檔案
        prepare_output_file(output_path, "通道響應圖檔")

        # 創建設備服務和儲存庫
        device_repository = SQLModelDeviceRepository(session)
        device_service = DeviceService(device_repository)

        # 從資料庫獲取活動的發射器 (desired)
        logger.info("從數據庫獲取活動的發射器...")
        active_desired = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.DESIRED.value, active_only=True
        )

        # 從資料庫獲取活動的干擾器 (jammer)
        logger.info("從數據庫獲取活動的干擾器...")
        active_jammers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.JAMMER.value, active_only=True
        )

        # 從資料庫獲取活動的接收器
        logger.info("從數據庫獲取活動的接收器...")
        active_receivers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.RECEIVER.value, active_only=True
        )

        # 檢查是否有足夠的設備進行模擬
        if not active_desired:
            logger.error("沒有活動的發射器，無法生成通道響應圖")
            return False

        if not active_receivers:
            logger.error("沒有活動的接收器，無法生成通道響應圖")
            return False

        # 構建 TX_LIST
        tx_list = []

        # 添加從資料庫獲取的發射器
        for tx in active_desired:
            tx_name = tx.name
            tx_position = [tx.position_x, tx.position_y, tx.position_z]
            tx_orientation = [tx.orientation_x, tx.orientation_y, tx.orientation_z]
            tx_power = tx.power_dbm

            tx_list.append((tx_name, tx_position, tx_orientation, "desired", tx_power))
            logger.info(
                f"添加發射器: {tx_name}, 位置: {tx_position}, 方向: {tx_orientation}, 功率: {tx_power} dBm"
            )

        # 添加從資料庫獲取的干擾器 (如果有)
        for jammer in active_jammers:
            jammer_name = jammer.name
            jammer_position = [
                jammer.position_x,
                jammer.position_y,
                jammer.position_z,
            ]
            jammer_orientation = [
                jammer.orientation_x,
                jammer.orientation_y,
                jammer.orientation_z,
            ]
            jammer_power = jammer.power_dbm

            tx_list.append(
                (
                    jammer_name,
                    jammer_position,
                    jammer_orientation,
                    "jammer",
                    jammer_power,
                )
            )
            logger.info(
                f"添加干擾器: {jammer_name}, 位置: {jammer_position}, 方向: {jammer_orientation}, 功率: {jammer_power} dBm"
            )

        # 接收器設置
        receiver = active_receivers[0]  # 已確認有接收器
        rx_config = (
            receiver.name,
            [receiver.position_x, receiver.position_y, receiver.position_z],
        )
        logger.info(f"使用接收器 '{rx_config[0]}' 在位置 {rx_config[1]}")

        # 從 config.py 取得場景路徑
        scene_xml_path = get_scene_xml_file_path(scene_name)
        logger.info(f"從 {scene_xml_path} 加載場景")

        # 參數設置 (從 tf.py 移植)
        tx_array_config = {
            "num_rows": 1,
            "num_cols": 1,
            "vertical_spacing": 0.5,
            "horizontal_spacing": 0.5,
            "pattern": "iso",
            "polarization": "V",
        }
        rx_array_config = tx_array_config

        pathsolver_args = {
            "max_depth": 10,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": False,
            "synthetic_array": False,
            "seed": 41,
        }

        n_subcarriers = 1024
        subcarrier_spacing = 30e3
        num_ofdm_symbols = 1024

        # 場景設置
        logger.info("設置場景")
        scene = load_scene(scene_xml_path)
        scene.tx_array = PlanarArray(**tx_array_config)
        scene.rx_array = PlanarArray(**rx_array_config)

        # 清除現有的發射器和接收器
        for name in list(scene.transmitters.keys()) + list(scene.receivers.keys()):
            scene.remove(name)

        # 添加發射器
        logger.info("添加發射器和干擾器")

        def add_tx(scene, name, pos, ori, role, power_dbm):
            tx = SionnaTransmitter(
                name=name, position=to_sionna_coords(pos), orientation=ori, power_dbm=power_dbm
            )
            tx.role = role
            scene.add(tx)
            return tx

        for name, pos, ori, role, p_dbm in tx_list:
            add_tx(scene, name, pos, ori, role, p_dbm)

        # 添加接收器
        rx_name, rx_pos = rx_config
        logger.info(f"添加接收器 '{rx_name}' 在位置 {rx_pos}")
        scene.add(SionnaReceiver(name=rx_name, position=to_sionna_coords(rx_pos)))

        # 為所有發射器分配速度
        for name, tx in scene.transmitters.items():
            tx.velocity = [30, 0, 0]

        # 按角色分組發射器
        tx_names = list(scene.transmitters.keys())
        all_txs = [scene.get(n) for n in tx_names]
        idx_des = [
            i for i, tx in enumerate(all_txs) if getattr(tx, "role", None) == "desired"
        ]
        idx_jam = [
            i for i, tx in enumerate(all_txs) if getattr(tx, "role", None) == "jammer"
        ]

        # 計算路徑
        logger.info("計算路徑")
        solver = PathSolver()
        try:
            paths = solver(scene, **pathsolver_args)
        except RuntimeError as e:
            logger.error(f"PathSolver 錯誤: {e}")
            logger.error(
                "嘗試減少 max_depth, max_num_paths_per_src, 或使用更簡單的場景。"
            )
            return False

        # 計算 CFR
        logger.info("計算 CFR")
        freqs = subcarrier_frequencies(n_subcarriers, subcarrier_spacing)
        ofdm_symbol_duration = 1 / subcarrier_spacing

        H_unit = paths.cfr(
            frequencies=freqs,
            sampling_frequency=1 / ofdm_symbol_duration,
            num_time_steps=num_ofdm_symbols,
            normalize_delays=True,
            normalize=False,
            out_type="numpy",
        ).squeeze()  # shape: (num_tx, T, F)

        # 計算 H_all, H_des, H_jam
        logger.info("計算 H_all, H_des, H_jam")
        H_all = H_unit.sum(axis=0)

        # 安全檢查：確保有所需的發射器和干擾器
        H_des = np.zeros_like(H_all)
        if idx_des:
            H_des = H_unit[idx_des].sum(axis=0)

        H_jam = np.zeros_like(H_all)
        if idx_jam:
            H_jam = H_unit[idx_jam].sum(axis=0)

        # 準備繪圖網格
        logger.info("準備繪圖")
        T, F = H_des.shape
        t_axis = np.arange(T)
        f_axis = np.arange(F)
        T_mesh, F_mesh = np.meshgrid(t_axis, f_axis, indexing="ij")

        # 創建圖片並保存
        logger.info("繪製通道響應圖")
        fig = plt.figure(figsize=(18, 5))

        # 子圖 1: H_des
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.plot_surface(
            F_mesh, T_mesh, np.abs(H_des), cmap="viridis", edgecolor="none"
        )
        ax1.set_xlabel("子載波")
        ax1.set_ylabel("OFDM 符號")
        ax1.set_title("‖H_des‖")

        # 子圖 2: H_jam
        ax2 = fig.add_subplot(132, projection="3d")
        ax2.plot_surface(
            F_mesh, T_mesh, np.abs(H_jam), cmap="viridis", edgecolor="none"
        )
        ax2.set_xlabel("子載波")
        ax2.set_ylabel("OFDM 符號")
        ax2.set_title("‖H_jam‖")

        # 子圖 3: H_all
        ax3 = fig.add_subplot(133, projection="3d")
        ax3.plot_surface(
            F_mesh, T_mesh, np.abs(H_all), cmap="viridis", edgecolor="none"
        )
        ax3.set_xlabel("子載波")
        ax3.set_ylabel("OFDM 符號")
        ax3.set_title("‖H_all‖")

        plt.tight_layout()

        # 保存圖片
        logger.info(f"保存通道響應圖到 {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 檢查文件是否生成成功
        return verify_output_file(output_path)

    except Exception as e:
        logger.exception(f"生成通道響應圖時發生錯誤: {e}")
        # 確保關閉所有打開的圖表
        plt.close("all")
        return False


# 新增 ISS Map 生成函數
async def generate_iss_map(
    session: AsyncSession,
    output_path: str = str(ISS_MAP_IMAGE_PATH),
    scene_name: str = "nycu",
    scene_size: float = 128.0,
    altitude: float = 30.0,
    resolution: float = 4.0,
    cfar_threshold_percentile: float = 99.5,
    gaussian_sigma: float = 1.0,
    min_distance: int = 3,
    cell_size: float = 4.0,
    samples_per_tx: int = 10**7,
    position_override: dict = None,
    force_refresh: bool = False,
    cell_size_override: Optional[float] = None,
    map_size_override: Optional[tuple[int, int]] = None,
    center_on: str = "receiver",
    # --- 新增 UAV 稀疏取樣參數 ---
    uav_points: Optional[List[tuple[float, float]]] = None,  # 前端/DB座標 (x,y)
    num_random_samples: int = 0,      # 若未提供 uav_points，可隨機抽樣N點
    sparse_noise_std_db: float = 0.0, # 給稀疏量測加高斯雜訊(分貝)
    sparse_first_then_full: bool = True,  # 先顯示稀疏點，再顯示完整圖
    sparse_output_path: Optional[str] = None,  # 若要另外輸出稀疏圖
) -> bool:
    """
    生成干擾信號強度 (ISS) 地圖並進行 2D-CFAR 檢測
    
    從數據庫獲取發射器和干擾器設置，計算並生成 ISS 地圖
    基於更新後的 ISS_MAP.py 實現
    """
    logger.info("開始生成 ISS 地圖...")

    try:
        # 準備輸出檔案
        prepare_output_file(output_path, "ISS 地圖圖檔")

        # GPU 設置
        gpus = _setup_gpu()

        # 創建設備服務和儲存庫
        device_repository = SQLModelDeviceRepository(session)
        device_service = DeviceService(device_repository)

        # 從數據庫獲取活動的發射器 (desired)
        logger.info("從數據庫獲取活動的發射器...")
        active_desired = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.DESIRED.value, active_only=True
        )

        # 從數據庫獲取活動的干擾器 (jammer)
        logger.info("從數據庫獲取活動的干擾器...")
        active_jammers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.JAMMER.value, active_only=True
        )
        
        # 過濾掉隱藏的干擾器
        visible_jammers = [j for j in active_jammers if getattr(j, 'visible', True)]

        # 從數據庫獲取活動的接收器
        logger.info("從數據庫獲取活動的接收器...")
        active_receivers = await device_service.get_devices(
            skip=0, limit=100, role=DeviceRole.RECEIVER.value, active_only=True
        )

        # 生成包含設備座標的快取 key
        import hashlib
        import json
        import time
        
        # 確定實際使用的參數值（包括覆蓋參數）
        actual_cell_size = cell_size_override if cell_size_override is not None else cell_size
        actual_map_size = list(map_size_override) if map_size_override is not None else [512, 512]
        
        # 提取所有設備的位置和功率信息來生成快取 key
        cache_data = {
            "scene_name": scene_name,
            "scene_size": scene_size,
            "altitude": altitude,
            "cell_size": actual_cell_size,
            "map_size": actual_map_size,
            "samples_per_tx": samples_per_tx,
            "cfar_threshold_percentile": cfar_threshold_percentile,
            "gaussian_sigma": gaussian_sigma,
            "min_distance": min_distance,
            "desired_devices": [
                {
                    "name": tx.name,
                    "position": [tx.position_x, tx.position_y, tx.position_z],
                    "power_dbm": tx.power_dbm
                } for tx in active_desired
            ],
            "jammer_devices": [
                {
                    "name": jammer.name,
                    "position": [jammer.position_x, jammer.position_y, jammer.position_z],
                    "power_dbm": jammer.power_dbm
                } for jammer in visible_jammers
            ],
            "receiver_devices": [
                {
                    "name": rx.name,
                    "position": [rx.position_x, rx.position_y, rx.position_z]
                } for rx in active_receivers
            ]
        }
        
        # 應用位置覆蓋參數來修改快取 key
        if position_override:
            logger.info(f"應用位置覆蓋參數: {position_override}")
            
            # 覆蓋 TX 位置
            if 'tx' in position_override and cache_data["desired_devices"]:
                for device in cache_data["desired_devices"]:
                    device["position"] = [
                        position_override['tx']['x'],
                        position_override['tx']['y'], 
                        position_override['tx']['z']
                    ]
                    logger.info(f"覆蓋 TX 設備 {device['name']} 位置: {device['position']}")
            
            # 覆蓋 Jammer 位置  
            if 'jammers' in position_override and cache_data["jammer_devices"]:
                jammer_positions = position_override['jammers']
                for i, device in enumerate(cache_data["jammer_devices"]):
                    if i < len(jammer_positions):
                        # 使用對應索引的位置
                        pos = jammer_positions[i]
                        device["position"] = [pos['x'], pos['y'], pos['z']]
                        logger.info(f"覆蓋 Jammer 設備 {device['name']} 位置: {device['position']}")
                    else:
                        logger.warning(f"Jammer 設備 {device['name']} 沒有對應的覆蓋位置，使用原始位置")
            
            # 加入位置覆蓋資訊到快取 key 中
            cache_data["position_override"] = position_override
        
        # 如果 force_refresh，在快取 key 中加入時間戳避免快取命中
        if force_refresh:
            cache_data["force_refresh_timestamp"] = time.time()
            logger.info("強制重新生成地圖 - 跳過快取")
        
        # 生成快取 key
        cache_key_str = json.dumps(cache_data, sort_keys=True)
        cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
        logger.info(f"生成快取 key: {cache_key[:16]}... (基於 {len(active_desired)} 發射器, {len(active_jammers)} 干擾器, {len(active_receivers)} 接收器位置)")
        
        # 檢查全域快取 (這裡假設有一個全域的快取字典)
        if not hasattr(generate_iss_map, '_iss_cache'):
            generate_iss_map._iss_cache = {}
            
        # 檢查快取 (force_refresh 會因為時間戳導致快取未命中)
        cache_hit = cache_key in generate_iss_map._iss_cache
        if cache_hit:
            cached_data = generate_iss_map._iss_cache[cache_key]
            logger.info("✓ 使用快取的 ISS 地圖數據 - 設備位置未變更")
        else:
            logger.info("✗ 無快取數據或設備位置已變更，開始計算新的無線電地圖...")

        # 檢查是否有足夠的設備
        if not visible_jammers:
            logger.info("沒有可見的干擾器，生成無干擾的 ISS 地圖")
            # 可以繼續運行，只是沒有干擾器影響
        else:
            logger.info(f"找到 {len(visible_jammers)} 個可見的干擾器")

        if not active_receivers:
            logger.warning("沒有活動的接收器，將使用預設接收器位置")
            return False
        else:
            # 使用第一個活動接收器
            receiver = active_receivers[0]
            
            # 檢查是否有RX位置覆蓋（用於實時計算）
            if position_override and 'rx' in position_override:
                rx_position = [
                    position_override['rx']['x'], 
                    position_override['rx']['y'], 
                    position_override['rx']['z']
                ]
                logger.info(f"使用覆蓋位置 RX: {rx_position}")
                rx_config = (receiver.name, rx_position)
            else:
                rx_config = (
                    receiver.name,
                    [receiver.position_x, receiver.position_y, receiver.position_z],
                )

        # 快取邏輯分支
        if cache_hit:
            # 使用快取數據
            cached_data = generate_iss_map._iss_cache[cache_key]
            iss_dbm = cached_data['iss_dbm']
            x_unique = cached_data['x_unique']
            y_unique = cached_data['y_unique']
            peak_coords = cached_data['peak_coords']
            all_txs_info = cached_data['all_txs_info']
            # 檢查是否有GPS峰值數據，如果沒有則計算
            if 'peak_locations_gps' in cached_data:
                peak_locations_gps = cached_data['peak_locations_gps']
            else:
                peak_locations_gps = []
            
            logger.info(f"從快取載入 ISS 地圖數據: {iss_dbm.shape}")
        else:
            # 需要重新計算
            # 構建 TX_LIST 使用更新的格式
            tx_list = []

            # 添加發射器 (desired)
            for tx in active_desired:
                # 檢查是否有位置覆蓋
                if position_override and 'tx' in position_override:
                    position = [position_override['tx']['x'], position_override['tx']['y'], position_override['tx']['z']]
                    logger.info(f"使用覆蓋位置 TX: {position}")
                else:
                    position = [tx.position_x, tx.position_y, tx.position_z]
                
                tx_info = {
                    "name": tx.name,
                    "position": position,
                    "orientation": [tx.orientation_x, tx.orientation_y, tx.orientation_z],
                    "role": "desired",
                    "power_dbm": tx.power_dbm
                }
                tx_list.append(tx_info)
                logger.info(
                    f"添加發射器: {tx_info['name']}, 位置: {tx_info['position']}, 方向: {tx_info['orientation']}, 功率: {tx_info['power_dbm']} dBm"
                )

            # 添加干擾器 (jammer)
            for i, jammer in enumerate(visible_jammers):
                # 檢查是否有位置覆蓋
                if position_override and 'jammers' in position_override:
                    jammer_positions = position_override['jammers']
                    if i < len(jammer_positions):
                        pos = jammer_positions[i]
                        position = [pos['x'], pos['y'], pos['z']]
                        logger.info(f"使用覆蓋位置 Jammer {i+1}: {position}")
                    else:
                        position = [jammer.position_x, jammer.position_y, jammer.position_z]
                        logger.info(f"Jammer {i+1} 沒有覆蓋位置，使用原始位置: {position}")
                else:
                    position = [jammer.position_x, jammer.position_y, jammer.position_z]
                
                jammer_info = {
                    "name": jammer.name,
                    "position": position,
                    "orientation": [jammer.orientation_x, jammer.orientation_y, jammer.orientation_z],
                    "role": "jammer",
                    "power_dbm": jammer.power_dbm
                }
                tx_list.append(jammer_info)
                logger.info(
                    f"添加干擾器: {jammer_info['name']}, 位置: {jammer_info['position']}, 方向: {jammer_info['orientation']}, 功率: {jammer_info['power_dbm']} dBm"
                )

            # 天線配置
            TX_ARRAY_CONFIG = {
                "num_rows": 1,
                "num_cols": 1,
                "vertical_spacing": 0.5,
                "horizontal_spacing": 0.5,
                "pattern": "iso",
                "polarization": "V"
            }
            RX_ARRAY_CONFIG = TX_ARRAY_CONFIG

            # 載入場景
            scene_xml_path = get_scene_xml_file_path(scene_name)
            logger.info(f"從 {scene_xml_path} 加載場景")
            
            try:
                scene = load_scene(scene_xml_path)
            except Exception as e:
                logger.warning(f"無法載入自定義場景 {scene_xml_path}: {e}")
                logger.info("使用內建場景...")
                scene = None

            if scene is None:
                logger.error("無法建立場景")
                return False

            # 設置天線陣列
            scene.tx_array = PlanarArray(**TX_ARRAY_CONFIG)
            scene.rx_array = PlanarArray(**RX_ARRAY_CONFIG)

            # 清除現有的發射器和接收器
            for tx_name in list(scene.transmitters):
                scene.remove(tx_name)
            for rx_name in list(scene.receivers):
                scene.remove(rx_name)

            # 添加發射器
            transmitters = []
            for tx_info in tx_list:
                tx_position = to_sionna_coords(tx_info["position"])
                tx = SionnaTransmitter(
                    name=tx_info["name"],
                    position=tx_position,
                    orientation=tx_info["orientation"],
                    power_dbm=tx_info["power_dbm"]
                )
                # 設置角色屬性
                tx.role = tx_info["role"]
                scene.add(tx)
                transmitters.append(tx)
            scene.frequency= 1.5e9
            # 添加接收器
            rx_name, rx_pos = rx_config
            rx_position = to_sionna_coords(rx_pos)
            rx = SionnaReceiver(name=rx_name, position=rx_position)
            scene.add(rx)

            # 座標一致性檢查日誌
            logger.info("=== 座標一致性檢查 ===")
            for name in scene.transmitters:
                tx = scene.get(name)
                logger.info(f"[Sionna] TX {tx.name} pos={tx.position} role={getattr(tx, 'role', None)}")
            rx_obj = scene.get(rx_name)
            logger.info(f"[Sionna] RX {rx_obj.name} pos={rx_obj.position}")
            logger.info("=== 座標檢查完成 ===")

            # 計算無線電地圖
            logger.info("計算無線電地圖...")
            logger.info(f"使用解析度: {actual_cell_size} 米/像素")
            logger.info(f"使用地圖大小: {actual_map_size[0]} x {actual_map_size[1]} 像素")
            
            # 根據center_on參數設定地圖中心
            if center_on == "transmitter" and tx_list:
                # 尋找第一個發射機 (desired) 作為中心
                first_tx = None
                for tx in tx_list:
                    if tx["role"] == "desired":
                        first_tx = tx
                        break
                
                if first_tx:
                    # 確保座標系統一致
                    tx_pos = first_tx["position"]
                    map_center = [tx_pos[0], -tx_pos[1],1.5]
                    logger.info(f"使用發射機({first_tx['name']})位置作為地圖中心: {map_center}")
                else:
                    # 如果沒有發射機，回退到接收機
                    map_center = [rx_position[0], rx_position[1],1.5]
                    logger.info(f"未找到發射機，使用接收機位置作為地圖中心: {map_center}")
            else:
                # 預設使用接收機位置作為中心
                map_center = [rx_position[0], rx_position[1], 1.5]
                logger.info(f"使用接收機位置作為地圖中心: {map_center}")
            
            rm_solver = RadioMapSolver()
            rm = rm_solver(scene,
                           max_depth=5,           # Maximum number of ray scene interactions
                           samples_per_tx=samples_per_tx, 
                           cell_size=(actual_cell_size, actual_cell_size),      # Resolution of the radio map
                           center=map_center,       # Center of the radio map at receiver position
                           size=actual_map_size,       # Total size of the radio map
                           orientation=[0, 0, 0],
                           refraction=False,
                           specular_reflection=True,
                           diffuse_reflection=True)

            # 提取資料
            logger.info("提取無線電地圖數據...")
            # 獲取cell中心座標
            cc = rm.cell_centers.numpy()
            x_unique = cc[0, :, 0]
            y_unique = cc[:, 0, 1]

            # 獲取所有發射器
            all_txs = [scene.get(name) for name in scene.transmitters]

            # 分組：期望發射器和干擾器
            idx_des = [i for i, tx in enumerate(all_txs) if tx.role == 'desired']
            idx_jam = [i for i, tx in enumerate(all_txs) if tx.role == 'jammer']
            logger.info(f"干擾器索引: {idx_jam}")

            # 獲取RSS（接收信號強度）
            WSS = rm.rss[:].numpy()
            TSS = np.sum(WSS, axis=0)  # 將所有發射器的RSS加總
            logger.info(f"RSS形狀: {TSS.shape}")
            DSS = np.sum(WSS[idx_des,:,:], axis=0) if idx_des else np.zeros_like(TSS)
            ISS = np.sum(WSS[idx_jam,:,:], axis=0) if idx_jam else np.zeros_like(TSS)

            # 使用改進的2D CFAR檢測干擾源位置
            # 避免除零錯誤，設置最小值
            ISS_safe = np.maximum(ISS, 1e-12)  # 設置最小值為 1e-12 (避免 log10(0))
            iss_dbm = 10 * np.log10(ISS_safe / 1e-3)
            logger.info(f"ISS 原始數據統計: min={np.min(ISS):.2e}, max={np.max(ISS):.2e}, 零值數量={np.sum(ISS == 0)}")
            
            # 計算 TSS (Total Signal Strength) - 所有發射器的信號強度加總
            TSS_safe = np.maximum(TSS, 1e-12)
            TSS_dbm = 10 * np.log10(TSS_safe / 1e-3)

            DSS_safe = np.maximum(DSS, 1e-12)
            DSS_dbm = 10 * np.log10(DSS_safe / 1e-3)

            logger.info(f"TSS 原始數據統計: min={np.min(TSS):.2e}, max={np.max(TSS):.2e}, 零值數量={np.sum(TSS == 0)}")

            # 準備 ISS 地圖數據和 CFAR 檢測
            iss_smooth = gaussian_filter(iss_dbm, sigma=gaussian_sigma)
            logger.info(f"生成 ISS 地圖 - 執行 2D-CFAR 檢測")
            
            # 2D-CFAR 偵測 (僅對 ISS 執行)
            # 先做最大值過濾 (局部最大值)
            local_max = maximum_filter(iss_smooth, size=5)
            peaks = (iss_smooth == local_max)

            # 簡化的CFAR檢測：直接找最大值峰值
            iss_max = np.max(iss_smooth)
            iss_mean = np.mean(iss_smooth)
            
            # 只有當最大值明顯高於平均值時才認為有峰值
            # 使用動態範圍的閾值：最大值需要超過平均值 + 2*標準差
            iss_std = np.std(iss_smooth)
            threshold = iss_mean + 0.1 * iss_std
            
            logger.info(f"CFAR檢測統計: max={iss_max:.2f}, mean={iss_mean:.2f}, std={iss_std:.2f}")
            logger.info(f"CFAR閾值計算: {iss_mean:.2f} + 2×{iss_std:.2f} = {threshold:.2f}")
            
            peak_coords = []
            if iss_max > threshold:
                # 找到最大值的位置
                max_indices = np.where(iss_smooth == iss_max)
                if len(max_indices[0]) > 0:
                    # 取第一個最大值位置 - 需要轉換為numpy陣列格式
                    peak_coords = np.array([[max_indices[0][0], max_indices[1][0]]])
                    logger.info(f"✓ 檢測到CFAR峰值: 位置({max_indices[0][0]}, {max_indices[1][0]}), 強度{iss_max:.2f}dBm > 閾值{threshold:.2f}dBm")
                else:
                    logger.info("✗ 無法定位最大值位置")
            else:
                logger.info(f"✗ 無CFAR峰值: 最大值{iss_max:.2f}dBm ≤ 閾值{threshold:.2f}dBm")

            # 準備 TSS 地圖數據 (不需要 CFAR 檢測)
            tss_smooth = gaussian_filter(TSS_dbm, sigma=gaussian_sigma)
            logger.info(f"生成 TSS 地圖 - 不執行 CFAR 檢測")

            # 保存發射器信息用於可視化
            all_txs_info = [
                {
                    "name": tx.name,
                    "position": tx.position,
                    "role": tx.role
                }
                for tx in all_txs
            ]
            
            # 計算峰值的GPS座標
            peak_locations_gps = []
            if len(peak_coords) > 0:
                logger.info(f"計算 {len(peak_coords)} 個CFAR峰值的GPS座標...")
                from app.api.v1.interference.routes_sparse_scan import frontend_coords_to_gps
                
                for i, coord in enumerate(peak_coords):
                    # coord = [row, col] in the ISS map grid
                    row_idx, col_idx = coord[0], coord[1]
                    
                    # 確保索引在有效範圍內
                    if 0 <= row_idx < len(y_unique) and 0 <= col_idx < len(x_unique):
                        # 從grid座標轉換為實際座標（Sionna座標系）
                        x_sionna = float(x_unique[col_idx])
                        y_sionna = float(y_unique[row_idx])
                        
                        # 從Sionna座標轉換為前端座標
                        frontend_coords = to_frontend_coords([x_sionna, y_sionna, 0])
                        x_frontend = frontend_coords[0]
                        y_frontend = frontend_coords[1]
                        
                        # 轉換為GPS座標
                        gps_coord = frontend_coords_to_gps(x_frontend, y_frontend, 0.0, scene_name)
                        
                        # 獲取該位置的ISS強度值
                        iss_value = float(iss_dbm[row_idx, col_idx]) if row_idx < iss_dbm.shape[0] and col_idx < iss_dbm.shape[1] else 0.0
                        
                        peak_info = {
                            "peak_id": i + 1,
                            "grid_coords": {"row": int(row_idx), "col": int(col_idx)},
                            "sionna_coords": {"x": x_sionna, "y": y_sionna},
                            "frontend_coords": {"x": x_frontend, "y": y_frontend},
                            "gps_coords": {
                                "latitude": gps_coord.latitude,
                                "longitude": gps_coord.longitude,
                                "altitude": gps_coord.altitude
                            },
                            "iss_strength_dbm": iss_value
                        }
                        peak_locations_gps.append(peak_info)
                        
                        logger.info(f"CFAR峰值 {i+1}: Grid({row_idx}, {col_idx}) -> Frontend({x_frontend:.1f}, {y_frontend:.1f}) -> GPS({gps_coord.latitude:.6f}, {gps_coord.longitude:.6f}), ISS: {iss_value:.1f} dBm")
                    else:
                        logger.warning(f"峰值索引 {coord} 超出grid範圍 {iss_dbm.shape}")
            
            # 保存計算結果到快取
            logger.info("保存計算結果到快取...")
            generate_iss_map._iss_cache[cache_key] = {
                'iss_dbm': iss_dbm,
                'x_unique': x_unique,
                'y_unique': y_unique,
                'peak_coords': peak_coords,
                'peak_locations_gps': peak_locations_gps,
                'all_txs_info': all_txs_info,
                'timestamp': time.time()
            }
            logger.info(f"快取已更新 - 快取大小: {len(generate_iss_map._iss_cache)}")

        # 在此處，不管是從快取還是新計算的數據都已準備好

        # ====== [新增] UAV 稀疏點抽樣與預覽 ======
        sparse_done = False
        if sparse_first_then_full and (uav_points or num_random_samples > 0):
            # 1) 準備 UAV 取樣點（前端/DB座標系）
            if uav_points:
                pts_frontend = uav_points
            else:
                # 在完整圖的範圍內隨機抽樣
                # 注意：x_unique/y_unique 已是 Sionna 座標，前端是 y 取負
                # 這裡我們直接在「前端座標空間」抽樣，再轉 Sionna 做取樣
                xmin, xmax = float(np.min(x_unique)), float(np.max(x_unique))
                ymin, ymax = float(np.min(y_unique)), float(np.max(y_unique))
                # 轉回前端範圍（y 軸反號）：y_front = -y_sionna
                y_front_min, y_front_max = -ymax, -ymin
                rng = np.random.default_rng(1234)
                xs_rand = rng.uniform(xmin, xmax, size=num_random_samples)
                ys_rand_front = rng.uniform(y_front_min, y_front_max, size=num_random_samples)
                pts_frontend = list(zip(xs_rand, ys_rand_front))

            # 2) 在這些點上取樣 ISS(dBm)
            sparse_x_sionna, sparse_y_sionna, sparse_vals_dbm = sample_iss_at_points(
                x_unique, y_unique, iss_dbm, pts_frontend, noise_std_db=sparse_noise_std_db
            )

            # 3) 繪製稀疏預覽圖（只顯示量測點）
            fig_s, ax_s = plt.subplots(figsize=(7, 5))
            sc = ax_s.scatter(
                sparse_x_sionna, sparse_y_sionna,
                c=sparse_vals_dbm, s=22, marker='o'
            )
            cbar_s = plt.colorbar(sc, ax=ax_s, label="Measured ISS (dBm)")
            ax_s.set_xlabel("x (m)")
            ax_s.set_ylabel("y (m)")
            ax_s.set_title("UAV Sparse ISS Samples")

            # 畫設備位置（用 Sionna 座標）
            for tx_info in all_txs_info:
                if tx_info['role'] == 'desired':
                    ax_s.scatter(tx_info['position'][0], tx_info['position'][1],
                                 c='blue', marker='^', s=80, label='Desired Tx')
            for tx_info in all_txs_info:
                if tx_info['role'] == 'jammer':
                    ax_s.scatter(tx_info['position'][0], tx_info['position'][1],
                                 c='red', marker='x', s=80, label='Jammer')
            # 獲取接收器位置
            rx_name, rx_pos = rx_config
            rx_obj = scene.get(rx_name)
            if rx_obj:
                ax_s.scatter(rx_obj.position[0], rx_obj.position[1],
                             c='green', marker='o', s=50, label='Rx')

            # 去重 legend
            handles, labels = ax_s.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            if len(uniq) > 0:
                ax_s.legend(uniq.values(), uniq.keys(), loc="best")

            plt.tight_layout()

            # 是否另存檔（可與完整圖同資料夾，檔名自動加 _sparse）
            sparse_path = sparse_output_path or (
                os.path.splitext(output_path)[0] + "_sparse.png"
            )
            prepare_output_file(sparse_path, "ISS 稀疏預覽圖檔")
            plt.savefig(sparse_path, dpi=300, bbox_inches="tight")
            plt.close(fig_s)
            logger.info(f"UAV 稀疏 ISS 預覽已保存: {sparse_path}")
            sparse_done = True
        # ====== [新增結束] ======

        # 同時生成 ISS 和 TSS 兩張地圖
        logger.info("同時生成 ISS 和 TSS 地圖可視化")
        logger.info(f"ISS 地圖數據統計: min={np.min(iss_dbm):.2f} dBm, max={np.max(iss_dbm):.2f} dBm, mean={np.mean(iss_dbm):.2f} dBm")
        logger.info(f"TSS 地圖數據統計: min={np.min(TSS_dbm):.2f} dBm, max={np.max(TSS_dbm):.2f} dBm, mean={np.mean(TSS_dbm):.2f} dBm")
        logger.info(f"地圖形狀: {iss_dbm.shape}, 座標範圍: x=[{np.min(x_unique):.1f}, {np.max(x_unique):.1f}], y=[{np.min(y_unique):.1f}, {np.max(y_unique):.1f}]")
        
        # Use meshgrid like SINR map for consistent coordinate handling
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # 生成 ISS 地圖
        def generate_map_visualization(data_dbm, map_title, output_path, include_peaks=False, peak_coords_data=None):
            plt.figure(figsize=(8, 6))
            
            # 檢查是否有有效數據
            if np.all(np.isnan(data_dbm)) or np.all(data_dbm == -np.inf):
                logger.warning(f"{map_title} 地圖數據全為 NaN 或 -inf，將使用全零數據")
                data_dbm = np.zeros_like(data_dbm)
            
            # 設置顏色範圍來改善可視化效果
            
            vmin = -100
            vmax = -30
            if include_peaks and visible_jammers:  # ISS 地圖有干擾器
                plt.pcolormesh(X, Y, data_dbm, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
                plt.colorbar(label=f"{map_title} (dBm)")
                plt.title(map_title)
            elif include_peaks:  # ISS 地圖無干擾器
                plt.pcolormesh(X, Y, data_dbm, shading='nearest', cmap='viridis', vmin=-100, vmax=-50)
                plt.colorbar(label=f"{map_title} (dBm)")
                plt.title(map_title)
            else:  # TSS 地圖
                plt.pcolormesh(X, Y, data_dbm, shading='nearest', cmap='viridis', vmin=-100, vmax=-50)
                plt.colorbar(label=f"{map_title} (dBm)")
                plt.title(map_title)

            
            # 標記檢測到的峰值 (僅限 ISS 模式)
            if include_peaks and peak_coords_data is not None and len(peak_coords_data) > 0:
                peak_x = x_unique[peak_coords_data[:, 1]]
                peak_y = y_unique[peak_coords_data[:, 0]]
                plt.scatter(peak_x, peak_y, color='r', marker='+', s=100, label='2D-CFAR Peaks')
                
            return plt
            
        # 添加設備位置繪製的共用函數
        def add_device_positions(rx_config, all_txs_info, scene):
            # 期望發射器（藍色三角形）
            for tx_info in all_txs_info:
                if tx_info['role'] == 'desired':
                    plt.scatter(tx_info['position'][0], tx_info['position'][1], c='blue', marker='^', s=100, label='Desired Tx')
            
            # 干擾器（紅色X）
            for tx_info in all_txs_info:
                if tx_info['role'] == 'jammer':
                    plt.scatter(tx_info['position'][0], tx_info['position'][1], c='red', marker='x', s=100, label='Jammer')
            
            # 接收器（綠色圓圈）
            rx_name, rx_pos = rx_config
            rx_obj = scene.get(rx_name)
            if rx_obj:
                plt.scatter(rx_obj.position[0], rx_obj.position[1], c='green', marker='o', s=50, label='Rx')

        # 1. 生成 ISS 地圖 
        generate_map_visualization(TSS_dbm, "ISS Map with 2D-CFAR Peak Detection", str(ISS_MAP_IMAGE_PATH), 
                                   include_peaks=True, peak_coords_data=peak_coords)
        add_device_positions(rx_config, all_txs_info, scene)
        plt.tight_layout()
        plt.legend()
        logger.info(f"保存 ISS 地圖到 {ISS_MAP_IMAGE_PATH}")
        plt.savefig(str(ISS_MAP_IMAGE_PATH), dpi=300, bbox_inches="tight")
        plt.close()

        # 2. 生成 TSS 地圖
        generate_map_visualization(DSS_dbm, "TSS Map - Total Signal Strength", str(TSS_MAP_IMAGE_PATH), 
                                   include_peaks=False)
        add_device_positions(rx_config, all_txs_info, scene)
        plt.tight_layout()
        plt.legend()
        logger.info(f"保存 TSS 地圖到 {TSS_MAP_IMAGE_PATH}")
        plt.savefig(str(TSS_MAP_IMAGE_PATH), dpi=300, bbox_inches="tight")
        plt.close()

        # 3. 生成 UAV Sparse 地圖 (如果有 UAV 點資料)
        uav_sparse_success = True
        if uav_points and len(uav_points) > 0:
            logger.info(f"生成 UAV Sparse 地圖 - 使用 {len(uav_points)} 個 UAV 掃描點")
            
            # 從 TSS 地圖在 UAV 點位置取樣
            sparse_x_sionna, sparse_y_sionna, sparse_vals_dbm = sample_iss_at_points(
                x_unique, y_unique, TSS_dbm, uav_points, noise_std_db=sparse_noise_std_db
            )
            
            # 創建 UAV Sparse 地圖可視化
            plt.figure(figsize=(8, 6))
            
            # 使用和 TSS 相同的顏色範圍
            vmin = np.percentile(TSS_dbm[np.isfinite(TSS_dbm)], 5) if np.any(np.isfinite(TSS_dbm)) else -80
            vmax = np.percentile(TSS_dbm[np.isfinite(TSS_dbm)], 95) if np.any(np.isfinite(TSS_dbm)) else -20
            
            # 繪製稀疏點
            sc = plt.scatter(
                sparse_x_sionna, sparse_y_sionna,
                c=sparse_vals_dbm, s=50, marker='o', cmap='viridis', 
                vmin=vmin, vmax=vmax, alpha=0.8
            )
            
            plt.colorbar(sc, label="UAV Sparse TSS (dBm)")
            plt.title("UAV Sparse Map - UAV Trajectory TSS Sampling")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            
            # 添加設備位置
            add_device_positions(rx_config, all_txs_info, scene)
            
            # 添加 UAV 軌跡線（連接稀疏點）
            if len(sparse_x_sionna) > 1:
                plt.plot(sparse_x_sionna, sparse_y_sionna, 'k--', alpha=0.3, linewidth=1, label='UAV Trajectory')
            
            plt.tight_layout()
            plt.legend()
            logger.info(f"保存 UAV Sparse 地圖到 {UAV_SPARSE_MAP_IMAGE_PATH}")
            plt.savefig(str(UAV_SPARSE_MAP_IMAGE_PATH), dpi=300, bbox_inches="tight")
            plt.close()
            
            # 檢查 UAV Sparse 地圖文件是否生成成功
            uav_sparse_success = verify_output_file(str(UAV_SPARSE_MAP_IMAGE_PATH))
            logger.info(f"UAV Sparse 地圖生成 {'成功' if uav_sparse_success else '失敗'}")
        else:
            logger.info("未提供 UAV 點資料，跳過 UAV Sparse 地圖生成")

        # 記錄檢測結果
        logger.info(f"檢測到 {len(peak_coords)} 個干擾源峰值")
        for i, coord in enumerate(peak_coords):
            logger.info(f"峰值 {i+1}: 行={coord[0]}, 列={coord[1]}")
        logger.info("ISS, TSS 和 UAV Sparse 地圖都已生成完成")

        # 檢查所有文件是否都生成成功
        iss_success = verify_output_file(str(ISS_MAP_IMAGE_PATH))
        tss_success = verify_output_file(str(TSS_MAP_IMAGE_PATH))
        
        # 將峰值數據轉換為GPS座標（用於直接返回）
        cfar_peaks_gps = []
        if len(peak_coords) > 0:
            logger.info(f"轉換 {len(peak_coords)} 個CFAR峰值為GPS座標...")
            from app.api.v1.interference.routes_sparse_scan import frontend_coords_to_gps
            
            for i, coord in enumerate(peak_coords):
                # coord = [row, col] in the ISS map grid
                row_idx, col_idx = coord[0], coord[1]
                
                # 確保索引在有效範圍內
                if 0 <= row_idx < len(y_unique) and 0 <= col_idx < len(x_unique):
                    # 從grid座標轉換為實際座標（Sionna座標系）
                    x_sionna = float(x_unique[col_idx])
                    y_sionna = float(y_unique[row_idx])
                    
                    # 從Sionna座標轉換為前端座標
                    frontend_coords = to_frontend_coords([x_sionna, y_sionna, 0])
                    x_frontend = frontend_coords[0]
                    y_frontend = frontend_coords[1]
                    
                    # 轉換為GPS座標
                    gps_coord = frontend_coords_to_gps(x_frontend, y_frontend, 0.0, scene_name)
                    
                    # 獲取該位置的ISS強度值
                    iss_value = float(iss_dbm[row_idx, col_idx]) if row_idx < iss_dbm.shape[0] and col_idx < iss_dbm.shape[1] else 0.0
                    
                    peak_info = {
                        "peak_id": i + 1,
                        "grid_coords": {"row": int(row_idx), "col": int(col_idx)},
                        "sionna_coords": {"x": x_sionna, "y": y_sionna},
                        "frontend_coords": {"x": x_frontend, "y": y_frontend},
                        "gps_coords": {
                            "latitude": gps_coord.latitude,
                            "longitude": gps_coord.longitude,
                            "altitude": gps_coord.altitude
                        },
                        "iss_strength_dbm": iss_value
                    }
                    cfar_peaks_gps.append(peak_info)
                    
                    logger.info(f"CFAR峰值 {i+1}: Grid({row_idx}, {col_idx}) -> Frontend({x_frontend:.1f}, {y_frontend:.1f}) -> GPS({gps_coord.latitude:.6f}, {gps_coord.longitude:.6f}), ISS: {iss_value:.1f} dBm")
                else:
                    logger.warning(f"峰值索引 {coord} 超出grid範圍 {iss_dbm.shape}")
        
        # 返回成功狀態和峰值數據
        overall_success = iss_success and tss_success and uav_sparse_success
        return {
            "success": overall_success,
            "cfar_peaks_gps": cfar_peaks_gps,
            "total_peaks": len(cfar_peaks_gps)
        }

    except Exception as e:
        logger.exception(f"生成 ISS 地圖時發生錯誤: {e}")
        # 確保關閉所有打開的圖表
        plt.close("all")
        return {
            "success": False,
            "cfar_peaks_gps": [],
            "total_peaks": 0,
            "error": str(e)
        }


# --- 主服務類 ---
class SionnaSimulationService(SimulationServiceInterface):
    """Sionna模擬服務實現"""

    def __init__(self):
        """初始化服務"""
        # 可根據需要在這裡添加初始化邏輯
        pass

    # --- 實現接口定義的方法 ---

    async def generate_empty_scene_image(self, output_path: str, scene_name: str = "NYCU") -> bool:
        """生成空場景圖像"""
        logger.info(
            f"SionnaSimulationService: Generating empty scene image at {output_path}"
        )

        # 準備輸出文件
        # For empty scene, the output_path passed from API is relative,
        # and prepare_output_file and subsequent functions handle it.
        prepare_output_file(output_path, "空場景圖像")

        # 嘗試設置 GPU
        _setup_gpu()

        # 設置 pyrender 場景
        pr_scene = _setup_pyrender_scene_from_glb(scene_name)
        if not pr_scene:
            logger.error("無法設置 pyrender 場景")
            return False

        # 渲染並保存場景
        result = _render_crop_and_save(
            pr_scene,
            output_path,  # Uses the output_path as received
            bg_color_float=SCENE_BACKGROUND_COLOR_RGB,
            render_width=1200,
            render_height=858,
            padding_y=20,
            padding_x=20,
        )

        return verify_output_file(output_path) if result else False

    async def generate_cfr_plot(
        self, session: AsyncSession, output_path: str, scene_name: str = "nycu"
    ) -> bool:
        """生成通道頻率響應(CFR)圖像"""
        logger.info(
            f"SionnaSimulationService: Calling global generate_cfr_plot, output_path: {output_path}, scene: {scene_name}"
        )
        return await generate_cfr_plot(
            session=session, output_path=output_path, scene_name=scene_name
        )

    async def generate_sinr_map(
        self,
        session: AsyncSession,
        output_path: str,  # This is an absolute path from config
        scene_name: str = "nycu",
        sinr_vmin: float = -40.0,
        sinr_vmax: float = 0.0,
        cell_size: float = 1.0,
        samples_per_tx: int = 10**7,
    ) -> bool:
        """生成SINR地圖"""
        logger.info(
            f"SionnaSimulationService: Calling global generate_sinr_map, output_path: {output_path}, scene: {scene_name}"
        )
        return await generate_sinr_map(
            session=session,
            output_path=output_path,
            scene_name=scene_name,
            sinr_vmin=sinr_vmin,
            sinr_vmax=sinr_vmax,
            cell_size=cell_size,
            samples_per_tx=samples_per_tx,
        )

    async def generate_radio_map(
        self,
        session: AsyncSession,
        output_path: str,
        scene_name: str = "nycu",
        sinr_vmin: float = -40.0,
        sinr_vmax: float = 0.0,
        cell_size: float = 1.0,
        samples_per_tx: int = 10**7,
        exclude_jammers: bool = True,
        center_on_transmitter: bool = True,
    ) -> bool:
        """生成無線電地圖 (可選擇排除干擾源)"""
        logger.info(
            f"SionnaSimulationService: Calling global generate_radio_map, output_path: {output_path}, scene: {scene_name}"
        )
        return await generate_radio_map(
            session=session,
            output_path=output_path,
            scene_name=scene_name,
            sinr_vmin=sinr_vmin,
            sinr_vmax=sinr_vmax,
            cell_size=cell_size,
            samples_per_tx=samples_per_tx,
            exclude_jammers=exclude_jammers,
            center_on_transmitter=center_on_transmitter,
        )

    async def generate_doppler_plots(
        self, session: AsyncSession, output_path: str, scene_name: str = "nycu"
    ) -> bool:
        """生成延遲多普勒圖"""
        logger.info(
            f"SionnaSimulationService: Calling global generate_doppler_plots, output_path: {output_path}, scene: {scene_name}"
        )
        return await generate_doppler_plots(
            session=session, output_path=output_path, scene_name=scene_name
        )

    async def generate_channel_response_plots(
        self, session: AsyncSession, output_path: str, scene_name: str = "nycu"
    ) -> bool:
        """生成通道響應圖"""
        logger.info(
            f"SionnaSimulationService: Calling global generate_channel_response_plots, output_path: {output_path}, scene: {scene_name}"
        )
        return await generate_channel_response_plots(
            session=session, output_path=output_path, scene_name=scene_name
        )

    async def generate_iss_map(
        self,
        session: AsyncSession,
        output_path: str,
        scene_name: str = "nycu",
        scene_size: float = 128.0,
        altitude: float = 30.0,
        resolution: float = 4.0,
        cfar_threshold_percentile: float = 99.5,
        gaussian_sigma: float = 1.0,
        min_distance: int = 3,
        cell_size: float = 1.0,
        samples_per_tx: int = 10**7,
        position_override: dict = None,
        force_refresh: bool = False,
        cell_size_override: Optional[float] = None,
        map_size_override: Optional[tuple[int, int]] = None,
        center_on: str = "receiver",
        # --- 新增 UAV 稀疏取樣參數 ---
        uav_points: Optional[List[tuple[float, float]]] = None,
        num_random_samples: int = 0,
        sparse_noise_std_db: float = 0.0,
        sparse_first_then_full: bool = True,
        sparse_output_path: Optional[str] = None,
    ) -> bool:
        """生成干擾信號強度 (ISS) 地圖並進行 2D-CFAR 檢測"""
        logger.info(
            f"SionnaSimulationService: Calling global generate_iss_map, output_path: {output_path}, scene: {scene_name}"
        )
        return await generate_iss_map(
            session=session,
            output_path=output_path,
            scene_name=scene_name,
            scene_size=scene_size,
            altitude=altitude,
            resolution=resolution,
            cfar_threshold_percentile=cfar_threshold_percentile,
            gaussian_sigma=gaussian_sigma,
            min_distance=min_distance,
            cell_size=cell_size,
            samples_per_tx=samples_per_tx,
            position_override=position_override,
            force_refresh=force_refresh,
            cell_size_override=cell_size_override,
            map_size_override=map_size_override,
            center_on=center_on,
            uav_points=uav_points,
            num_random_samples=num_random_samples,
            sparse_noise_std_db=sparse_noise_std_db,
            sparse_first_then_full=sparse_first_then_full,
            sparse_output_path=sparse_output_path,
        )

    async def run_simulation(
        self, session: AsyncSession, params: SimulationParameters
    ) -> Dict[str, Any]:
        """執行通用模擬"""
        logger.info(f"Running simulation of type: {params.simulation_type}")

        result = {"success": False, "result_path": None, "error_message": None}

        try:
            # 根據模擬類型執行不同的模擬
            if params.simulation_type == "cfr":
                output_path = str(CFR_PLOT_IMAGE_PATH)
                success = await self.generate_cfr_plot(session, output_path)
                result["result_path"] = output_path
                result["success"] = success

            elif params.simulation_type == "sinr_map":
                output_path = str(SINR_MAP_IMAGE_PATH)
                success = await self.generate_sinr_map(
                    session,
                    output_path,
                    params.sinr_vmin or -40.0,
                    params.sinr_vmax or 0.0,
                    params.cell_size or 1.0,
                    params.samples_per_tx or 10**7,
                )
                result["result_path"] = output_path
                result["success"] = success

            elif params.simulation_type == "doppler":
                output_path = str(DOPPLER_IMAGE_PATH)
                success = await self.generate_doppler_plots(session, output_path)
                result["result_path"] = output_path
                result["success"] = success

            elif params.simulation_type == "channel_response":
                output_path = str(CHANNEL_RESPONSE_IMAGE_PATH)
                success = await self.generate_channel_response_plots(
                    session, output_path
                )
                result["result_path"] = output_path
                result["success"] = success

            elif params.simulation_type == "iss_map":
                output_path = str(ISS_MAP_IMAGE_PATH)
                success = await self.generate_iss_map(
                    session,
                    output_path,
                    scene_name="nycu",
                    scene_size=params.scene_size or 128.0,
                    altitude=params.altitude or 30.0,
                    resolution=params.resolution or 4.0,
                    cfar_threshold_percentile=params.cfar_threshold_percentile or 99.5,
                    gaussian_sigma=params.gaussian_sigma or 1.0,
                    min_distance=params.min_distance or 3,
                    cell_size=params.cell_size or 1.0,
                    samples_per_tx=params.samples_per_tx or 10**7,
                )
                result["result_path"] = output_path
                result["success"] = success

            else:
                logger.error(f"不支援的模擬類型: {params.simulation_type}")
                result["error_message"] = f"不支援的模擬類型: {params.simulation_type}"

        except Exception as e:
            logger.error(f"執行模擬時發生錯誤: {str(e)}", exc_info=True)
            result["error_message"] = f"執行模擬時發生錯誤: {str(e)}"

        return result


# 創建服務實例
sionna_service = SionnaSimulationService()

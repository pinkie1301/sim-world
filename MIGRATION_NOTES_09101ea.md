# 遷移紀錄：從 1a91b39 起的所有改動

> **基準節點**：`1a91b39` (first commit0905)  
> **範圍**：`09101ea` → `c518a35` (HEAD)  
> **日期**：2026-03-02  
> **總計**：109 files changed, 532 insertions(+), 3,524 deletions(-)

---

## 目錄

1. [Commit 總覽](#commit-總覽)
2. [重大架構變更：移除 Satellite 領域](#重大架構變更移除-satellite-領域)
3. [新增 TestScene 場景支援](#新增-testscene-場景支援)
4. [TestScene 模型座標軸問題與解決方案](#testscene-模型座標軸問題與解決方案)
5. [前端渲染與光照改善](#前端渲染與光照改善)
6. [場景資源檔案變更](#場景資源檔案變更)
7. [逐檔改動詳述](#逐檔改動詳述)

---

## Commit 總覽

| Commit | 時間 | 訊息 |
|--------|------|------|
| `09101ea` | 2026-03-02 14:45:42 +0800 | fix: TestScene Z-up to Y-up rotation fix, separate scene model and device rotation |
| `c518a35` | 2026-03-02 15:57:50 +0800 | fix: 調整場景光照參數以改善渲染效果 |

---

## 重大架構變更：移除 Satellite 領域

### 刪除範圍

整個 `backend/app/domains/satellite/` 目錄被完整移除（**12 個檔案，共 2,747 行**）：

| 檔案 | 行數 | 說明 |
|------|------|------|
| `__init__.py` | 25 | 領域初始化 |
| `adapters/sqlmodel_satellite_repository.py` | 137 | SQLModel 資料庫 Adapter |
| `api/satellite_api.py` | 372 | REST API 端點 |
| `interfaces/orbit_service_interface.py` | 64 | 軌道服務抽象介面 |
| `interfaces/satellite_repository.py` | 67 | Repository 抽象介面 |
| `interfaces/tle_service_interface.py` | 40 | TLE 服務抽象介面 |
| `models/dto.py` | 17 | 資料傳輸物件 |
| `models/ground_station_model.py` | 37 | 地面站模型 |
| `models/satellite_model.py` | 121 | 衛星模型 |
| `services/cqrs_satellite_service.py` | 954 | CQRS 衛星指令/查詢服務 |
| `services/orbit_service.py` | 446 | 軌道計算服務 |
| `services/tle_service.py` | 467 | TLE 資料同步服務 |

### 關聯清理

- **`backend/app/api/v1/router.py`**：移除 `satellite_router` 註冊、Skyfield TLE 全域載入邏輯、`VisibleSatelliteInfo` 模型、`/satellite-ops/visible_satellites` 端點、所有 CQRS 衛星端點（~245 行）
- **`backend/app/db/lifespan.py`**：移除 ground station import、`seed_default_ground_station()` 函數、TLE 同步邏輯
- **`backend/app/domains/__init__.py`**：更新領域文檔字串，移除 satellite 描述
- **`backend/app/domains/context_maps.py`**：從 `CONTEXT_MAP` 和 `BOUNDED_CONTEXTS` 中移除 satellite 定義，更新 simulation 和 coordinates 的依賴關係
- **`.env` / `.env.example`**：移除 `SAT_TLE_LINE1`、`SAT_TLE_LINE2` 環境變數
- **`backend/requirements.txt`**：`skyfield` 保留但註釋從「新增 skyfield 套件」改為「用於座標服務中的 ECEF/WGS84 座標轉換」

---

## 新增 TestScene 場景支援

### 後端新增

#### 1. 座標轉換常數 — `backend/app/domains/coordinates/services/coordinate_service.py`

```python
# --- TestScene Coordinate Conversion Constants ---
ORIGIN_LATITUDE_TESTSCENE = 24.943834
ORIGIN_LONGITUDE_TESTSCENE = 121.369192
ORIGIN_FRONTEND_X_TESTSCENE = 0.0
ORIGIN_FRONTEND_Y_TESTSCENE = 0.0
LATITUDE_SCALE_PER_METER_Y_TESTSCENE = 9.044e-06
LONGITUDE_SCALE_PER_METER_X_TESTSCENE = 9.976e-06
```

#### 2. GPS 座標轉換路由 — `backend/app/api/v1/interference/routes_sparse_scan.py`

在 `frontend_coords_to_gps` 函數新增 `testscene` 場景分支：

```python
elif scene.lower() == "testscene":
    origin_lat = ORIGIN_LATITUDE_TESTSCENE
    origin_lon = ORIGIN_LONGITUDE_TESTSCENE
    origin_x = ORIGIN_FRONTEND_X_TESTSCENE
    origin_y = ORIGIN_FRONTEND_Y_TESTSCENE
    lat_scale = LATITUDE_SCALE_PER_METER_Y_TESTSCENE
    lon_scale = LONGITUDE_SCALE_PER_METER_X_TESTSCENE
```

#### 3. 無人機追蹤場景配置 — `backend/app/domains/drone_tracking/services/drone_tracking_service.py`

```python
"testscene": {
    "bounds": {"min_x": -256, "max_x": 256, "min_y": -256, "max_y": 256},
    "resolution": 4.0,
    "matrix_size": 128,
    "offset_x": 64,
    "offset_y": 64,
    "scale": 1.0
}
```

#### 4. Simulation API — `backend/app/domains/simulation/api/simulation_api.py`

所有 API 端點（`/scene-image`, `/cfr-plot`, `/sinr-map`, `/radio-map`, `/doppler-plots`, `/channel-response`, `/iss-map`, `/iss-map-cfar-peaks`）的場景 Query 參數描述從：
```
場景名稱 (nycu, lotus, ntpu, nanliao)
```
改為：
```
場景名稱 (nycu, lotus, ntpu, nanliao, testscene)
```

#### 5. Sionna 場景映射 — `backend/app/domains/simulation/services/sionna_service.py`

```python
# 註解掉 NTPU 格式不相容阻擋（NTPU XML 已修復可使用）
# if scene_name in ["NTPU"]:
#     raise ValueError(...)

# scene_mapping 新增：
"testscene": "TestScene",
```

### 前端新增

#### `frontend/src/utils/sceneUtils.ts`

```typescript
// SCENE_MAPPING
testscene: 'TestScene',

// SCENE_DISPLAY_NAMES
testscene: '測試場景',

// SCENE_COORDINATE_TRANSFORMS
testscene: {
    offsetX: 64,
    offsetY: 64,
    scale: 0.25,
    rotationX: -Math.PI / 2,  // Z-up → Y-up
    rotationY: 0,
},

// getSceneTextureName
case 'TestScene':
    return 'EXPORT_GOOGLE_SAT_WM.png'
```

回傳型別擴展：`getSceneCoordinateTransform()` 新增 `rotationX`、`rotationY` 欄位。

---

## TestScene 模型座標軸問題與解決方案

### 問題描述

TestScene 的 GLB 模型檔（`TestScene.glb`）使用 **Z-up** 座標系（常見於 Blender 等建模軟體的匯出），而 Three.js 使用 **Y-up** 座標系。直接載入模型會導致場景「躺平」（地面朝向攝影機而非水平展開）。

### 核心挑戰

修正座標軸時必須考慮三種不同的物件類別，它們各自需要不同的旋轉處理：

| 物件類別 | 旋轉需求 | 原因 |
|----------|----------|------|
| **場景模型** (GLB) | X 軸 -90° + Y 軸旋轉 | 修正 Z-up → Y-up，加上可能的場景朝向修正 |
| **設備** (TX/RX/Jammer) | 僅 Y 軸旋轉 | 設備座標已經是 Y-up，只需匹配場景朝向 |
| **衛星** | 無旋轉 | 使用天頂座標系 (azimuth/elevation)，獨立於場景 |

### 解決方案實作

#### 1. 場景工具層 — `sceneUtils.ts`

每個場景新增 `rotationX`、`rotationY` 屬性。TestScene 設定 `rotationX: -Math.PI / 2`（繞 X 軸旋轉 -90°），其餘場景均為 `0`。

#### 2. 渲染層分離 — `MainScene.tsx`

```tsx
// 取得場景旋轉參數
const { rotationX: sceneRotationX, rotationY: sceneRotationY } =
    getSceneCoordinateTransform(sceneName)

return (
    <>
        {/* 場景模型：X 軸 + Y 軸旋轉 */}
        <group rotation={[sceneRotationX, sceneRotationY, 0]}>
            <primitive object={prepared} castShadow receiveShadow />
        </group>
        {/* 設備：僅 Y 軸旋轉 */}
        <group rotation={[0, sceneRotationY, 0]}>
            {deviceMeshes}
        </group>
        {/* 衛星：不受場景旋轉 */}
        <SatelliteManager satellites={satellites} />
    </>
)
```

#### 3. Z-up 場景偵測 — 地面面積計算修正

```typescript
const isZUpScene = Math.abs(sceneRotationX + Math.PI / 2) < 0.01

// Z-up 模型：地面在 XY 平面展開，用 size.x * size.y 偵測
// Y-up 模型：地面在 XZ 平面展開，用 size.x * size.z 偵測
const area = isZUpScene ? size.x * size.y : size.x * size.z
```

因為 Z-up GLB 在尚未旋轉前，地面的 bounding box 在幾何空間中是 XY 平面展開的，所以要用 `x * y`（而非 Y-up 場景的 `x * z`）來正確找出面積最大的 mesh 作為地面。

### 仍需注意的潛在問題

- **設備位置座標**：目前設備使用 `[position_x, position_z, position_y]` 的 swizzle 對應（Y-up 場景慣例），若 TestScene 設備座標來源也是 Z-up 系統，可能需要額外轉換
- **地面材質 DoubleSide**：Z-up GLB 旋轉後法線可能朝下，使用 `THREE.DoubleSide` 作為 workaround 確保雙面可見

---

## 前端渲染與光照改善

以下改動在 commit `c518a35` 中實施：

### `MainScene.tsx` 材質與渲染修正

| 項目 | 修改前 | 修改後 | 原因 |
|------|--------|--------|------|
| 紋理 anisotropy | `16` | `2` | 降低 GPU 負擔 |
| 頂點法線 | 無處理 | `computeVertexNormals()` | TestScene GLB 缺少法線導致光照異常 |
| 頂點顏色 | 不支援 | 自動偵測 `hasVertexColors` | 支援 vertex color 材質 |
| 材質陣列替換 | `forEach` + 賦值 | `.map()` 替換 | **修正 bug**：forEach 中 `mat = newMat` 不會修改原陣列 |
| 地面 roughness | `0.8` | `0.6` | 降低粗糙度改善視覺 |
| 地面 emissive | `0x555555` / `0.4` | 移除 | 移除自發光，依賴場景光源 |
| 地面 normalScale | `(0.5, 0.5)` | 移除 | 簡化材質參數 |
| 地面 side | 預設 (FrontSide) | `THREE.DoubleSide` | 修正 Z-up 翻轉後法線反向 |
| useMemo deps | `[mainScene, SATELLITE_TEXTURE_URL]` | `+isZUpScene` | 場景類型變化時重新處理 |

### `StereogramView.tsx` 光照參數調整

| 項目 | 修改前 | 修改後 | 原因 |
|------|--------|--------|------|
| toneMappingExposure | `1.2` | `1.6` | 提高整體曝光亮度 |
| ambientLight intensity | `0.2` | `0.5` | 提高環境光減少暗部過暗 |
| shadow-mapSize | `4096×4096` | `1024×1024` | 降低 VRAM 使用，提升效能 |

---

## 場景資源檔案變更

### 新增檔案

| 路徑 | 說明 |
|------|------|
| `backend/app/static/scenes/TestScene/TestScene.glb` | TestScene 3D 模型 (4.88 MB) |
| `backend/app/static/scenes/TestScene/TestScene.xml` | Sionna 場景描述（183 行） |
| `backend/app/static/scenes/TestScene/textures/EXPORT_GOOGLE_SAT_WM.png` | 衛星底圖紋理 (4.83 MB) |
| `backend/app/static/scenes/TestScene/mesh/ground.ply` | 地面 mesh |
| `backend/app/static/scenes/TestScene/mesh/building_0.ply` ~ `building_23.ply` | 24 棟建築物 mesh |
| `backend/app/static/scenes/TestScene/misc/*.png` | CFAR/ISS/Coverage 分析圖（6 張） |
| `backend/app/static/scenes/NTPU/scene.glb` | NTPU 簡化場景模型 (48 KB) |
| `backend/app/static/scenes/NTPU/scene_meta.json` | NTPU 場景元資料 |
| `backend/app/static/scenes/NTPU/mesh/ground.ply` + `building_*.ply` | NTPU 建築物 mesh（25 個） |

### 更新檔案

| 路徑 | 說明 |
|------|------|
| `backend/app/static/scenes/NTPU/NTPU.glb` | 5.75 MB → 4.88 MB（重新匯出） |
| `backend/app/static/scenes/NTPU/NTPU.xml` | 完整重寫為 ITU 材質格式（與 TestScene 一致） |
| `backend/app/static/scenes/NTPU/textures/EXPORT_GOOGLE_SAT_WM.png` | 5.48 MB → 4.83 MB |
| `backend/app/static/images/channel_response_plots.png` | 2.63 MB → 1.58 MB |
| `backend/app/static/images/sinr_map.png` | 80 KB → 1.52 MB |
| `backend/app/static/images/iss_map.png` | 341 KB → 547 KB |
| `backend/app/static/images/tss_map.png` | 344 KB → 398 KB |

### 刪除檔案

| 路徑 | 說明 |
|------|------|
| `backend/app/static/images/cfr_plot.png` | 416 KB |
| `backend/app/static/scenes/NTPU/meshes/*.ply` (12 個) | 舊版 OSM 道路/建築 mesh |

### NTPU XML 改動說明

NTPU.xml 從舊版 OSM 地圖格式（包含 wall/roof/road 等 PLY mesh 和 shapegroup instance）完整重寫為程式生成場景格式，包含：
- `ground` + `building_0` ~ `building_23` 的 PLY mesh
- 使用 ITU 標準材質：`itu_concrete`、`itu_marble`、`itu_metal`、`itu_wood`、`itu_wet_ground`
- 新增 camera sensor 和 constant emitter 定義
- 場景格式現在與 TestScene 完全一致

### scene_meta.json 內容

```json
{
  "scene_xml": "data\\datasets\\24.943834_121.369192\\scene.xml",
  "grid_res": 128,
  "area_m": 512.0,
  "pixel_size_m": 4.0,
  "frequency_hz": 3500000000.0,
  "tx_list": [
    { "role": "desired", "position_px": [45, 85], "ptx_dbm": 40.0, "height_m": 40.0 },
    { "role": "jammer", "position_px": [30, 110], "ptx_dbm": 40.0, "height_m": 40.0 }
  ],
  "rx_height": 1.5,
  "sionna": { "max_depth": 10, "samples_per_tx": 1000000 },
  "outputs": {
    "building_map": "building_height_128.npy",
    "dss": "sionna_dss.npy",
    "iss": "sionna_iss.npy",
    "tss": "sionna_tss.npy"
  }
}
```

---

## 逐檔改動詳述

### Commit 1：`09101ea` — TestScene Z-up to Y-up rotation fix

#### `backend/app/api/v1/router.py`
- 移除 `import random`
- 移除 `from app.domains.satellite.api.satellite_api import router as satellite_router`
- 移除衛星 CQRS 相關 import（`CQRSSatelliteService`, `SatellitePosition`, `GeoCoordinate`）
- 移除 Skyfield import（`load`, `wgs84`, `EarthSatellite`, `numpy`）
- 移除全域 TLE 載入邏輯（`SKYFIELD_LOADED`, `SATELLITE_COUNT`, Celestrak TLE 嘗試載入區塊 ~30 行）
- 移除 `satellite_router` 路由註冊
- 移除 `VisibleSatelliteInfo` Pydantic 模型
- 移除 `get_visible_satellites` 端點（~110 行）
- 移除所有 CQRS 衛星端點（~245 行）：`get_satellite_position_cqrs`、`get_batch_satellite_positions_cqrs`、`force_update_satellite_position_cqrs`、`calculate_satellite_trajectory_cqrs`、`find_visible_satellites_cqrs`、`get_cqrs_satellite_service_stats`

#### `backend/app/api/v1/interference/routes_sparse_scan.py`
- 文檔字串新增 `testscene: TestScene場景`
- Import 新增 TestScene 座標常數
- `frontend_coords_to_gps` 函數新增 `testscene` 分支

#### `backend/app/db/lifespan.py`
- 移除 `GroundStation`、`GroundStationCreate`、`synchronize_oneweb_tles` import
- 簡化 `create_db_and_tables()` 中的衛星相關註釋
- 移除 `seed_default_ground_station()` 函數（~30 行）
- 移除 lifespan 中的 `seed_default_ground_station(db_session)` 呼叫和 TLE 同步邏輯（~10 行）

#### `backend/app/domains/__init__.py`
- 更新模組文檔字串：移除 `satellite`、`integration` 的舊描述，新增 `device`、`coordinates`、`wireless`、`interference`、`drone_tracking`、`integration` 說明

#### `backend/app/domains/context_maps.py`
- `CONTEXT_MAP`：移除 `satellite` 項目；`simulation.depends_on` 移除 `satellite`；`coordinates.used_by` 移除 `satellite`
- `BOUNDED_CONTEXTS`：移除 `satellite` 的描述

#### `backend/app/domains/coordinates/services/coordinate_service.py`
- 新增 7 個 TestScene 座標轉換常數（基準 GPS：24.943834, 121.369192）

#### `backend/app/domains/drone_tracking/services/drone_tracking_service.py`
- `SCENE_CONFIG` 新增 `testscene` 配置（512m 區域，128 grid，4.0m/pixel）

#### `backend/app/domains/simulation/api/simulation_api.py`
- 所有端點場景參數描述追加 `testscene`

#### `backend/app/domains/simulation/services/sionna_service.py`
- 註解掉 NTPU XML 不相容阻擋
- `scene_mapping` 新增 `"testscene": "TestScene"`

#### `backend/requirements.txt`
- `skyfield` 的註釋更新

#### `frontend/src/components/scenes/MainScene.tsx`
- Import `getSceneCoordinateTransform`
- 取得 `sceneRotationX`、`sceneRotationY`
- 將 `<primitive>` 和 `deviceMeshes` 分別包裹在帶旋轉的 `<group>` 中
- 衛星 `<SatelliteManager>` 不受旋轉影響

#### `frontend/src/utils/sceneUtils.ts`
- `SCENE_MAPPING` 新增 `testscene: 'TestScene'`
- `SCENE_DISPLAY_NAMES` 新增 `testscene: '測試場景'`
- `SCENE_COORDINATE_TRANSFORMS` 所有場景新增 `rotationX: 0, rotationY: 0`
- TestScene 新增 `rotationX: -Math.PI / 2`
- `getSceneTextureName()` 新增 `TestScene` case
- `getSceneCoordinateTransform()` 回傳型別新增 `rotationX`, `rotationY`

#### `.env` / `.env.example`
- 移除 `SAT_TLE_LINE1`、`SAT_TLE_LINE2`

---

### Commit 2：`c518a35` — 調整場景光照參數以改善渲染效果

#### `frontend/src/components/scenes/MainScene.tsx`
- 新增 `isZUpScene` 布林變數偵測 Z-up 場景
- 紋理 anisotropy `16` → `2`
- 新增 `computeVertexNormals()` 自動計算缺失法線
- 新增 vertex colors 偵測與支援
- 修正材質替換 bug：`forEach` + 賦值 → `.map()` 回傳新陣列
- 地面偵測面積：Z-up 用 `x*y`、Y-up 用 `x*z`
- 地面材質：roughness `0.8` → `0.6`，移除 emissive/normalScale，新增 `DoubleSide`
- `useMemo` 依賴新增 `isZUpScene`

#### `frontend/src/components/scenes/StereogramView.tsx`
- `toneMappingExposure` `1.2` → `1.6`
- `ambientLight intensity` `0.2` → `0.5`
- Shadow map size `4096` → `1024`

/**
 * 場景相關的工具函數
 */

// 場景名稱映射：前端路由參數 -> 後端場景目錄名稱
export const SCENE_MAPPING = {
    nycu: 'NYCU',
    lotus: 'Lotus',
    ntpu: 'NTPU',
    nanliao: 'nnn',
    potou: 'potou',
    poto: 'poto',
    testscene: 'TestScene',
} as const

// 場景顯示名稱映射
export const SCENE_DISPLAY_NAMES = {
    nycu: '陽明交通大學',
    lotus: '荷花池',
    ntpu: '臺北大學',
    nanliao: '南寮漁港',
    potou: '破斗山',
    poto: '坡頭漁港',
    testscene: '測試場景',
} as const

// 場景座標轉換參數映射
// scale: 像素/公尺比例 (pixel/meter), 根據後端cell_size=4.0m/pixel，所以scale=1/4.0=0.25 pixel/meter
// rotationY: 場景模型繞 Y 軸旋轉角度 (弧度)，用於修正 GLB 匯出軸向差異
export const SCENE_COORDINATE_TRANSFORMS = {
    nycu: {
        offsetX: 865,
        offsetY: 640,
        scale: 0.25,  // 1 pixel = 4 meters, so scale = 1/4 = 0.25 pixel/meter
        rotationX: 0,
        rotationY: 0,
    },
    lotus: {
        offsetX: 1200,
        offsetY: 900,
        scale: 0.25,
        rotationX: 0,
        rotationY: 0,
    },
    ntpu: {
        offsetX: 900,
        offsetY: 620,
        scale: 0.25,
        rotationX: 0,
        rotationY: 0,
    },
    nanliao: {
        offsetX: 920,
        offsetY: 600,
        scale: 0.25,
        rotationX: 0,
        rotationY: 0,
    },
    potou: {
        offsetX: 900,
        offsetY: 600,
        scale: 0.25,
        rotationX: 0,
        rotationY: 0,
    },
    poto: {
        offsetX: 900,
        offsetY: 600,
        scale: 0.25,
        rotationX: 0,
        rotationY: 0,
    },
    testscene: { 
        offsetX: 64, 
        offsetY: 64, 
        scale: 0.25,
        rotationX: -Math.PI / 2,  // TestScene GLB 使用 Z-up，需繞 X 軸旋轉 -90° 修正為 three.js Y-up
        rotationY: 0,
    },
} as const

/**
 * 將前端路由參數轉換為後端場景名稱
 * @param sceneParam 前端路由參數 (如 'nycu', 'lotus')
 * @returns 後端場景目錄名稱 (如 'NYCU', 'Lotus')
 */
export function getBackendSceneName(sceneParam: string): string {
    const normalizedParam = sceneParam.toLowerCase()
    return SCENE_MAPPING[normalizedParam as keyof typeof SCENE_MAPPING] || SCENE_MAPPING.nycu
}

/**
 * 獲取場景的顯示名稱
 * @param sceneParam 前端路由參數 (如 'nycu', 'lotus')
 * @returns 場景顯示名稱 (如 '陽明交通大學', '荷花池')
 */
export function getSceneDisplayName(sceneParam: string): string {
    const normalizedParam = sceneParam.toLowerCase()
    return SCENE_DISPLAY_NAMES[normalizedParam as keyof typeof SCENE_DISPLAY_NAMES] || SCENE_DISPLAY_NAMES.nycu
}

/**
 * 獲取場景的紋理檔案名稱
 * @param sceneParam 前端路由參數
 * @returns 紋理檔案名稱
 */
export function getSceneTextureName(sceneParam: string): string {
    const backendName = getBackendSceneName(sceneParam)
    
    // 根據不同場景返回對應的紋理檔案名稱
    switch (backendName) {
        case 'NYCU':
            return 'EXPORT_GOOGLE_SAT_WM.png'
        case 'Lotus':
            return 'EXPORT_GOOGLE_SAT_WM.png'  // 假設 Lotus 也使用相同的紋理檔案
        case 'NTPU':
            return 'EXPORT_GOOGLE_SAT_WM.png'  // 臺北大學使用相同的紋理檔案
        case 'nnn':
            return 'EXPORT_GOOGLE_SAT_WM.002.png'  // nnn場景使用特定的紋理檔案
        case 'potou':
            return 'EXPORT_GOOGLE_SAT_WM.png'  // potou場景使用相同的紋理檔案
        case 'poto':
            return 'EXPORT_GOOGLE_SAT_WM.png'  // poto場景使用相同的紋理檔案
        case 'TestScene':
            return 'EXPORT_GOOGLE_SAT_WM.png'  // testscene場景使用相同的紋理檔案
        default:
            return 'EXPORT_GOOGLE_SAT_WM.png'
    }
}

/**
 * 獲取場景的座標轉換參數
 * @param sceneParam 前端路由參數
 * @returns 座標轉換參數
 */
export function getSceneCoordinateTransform(sceneParam: string): { offsetX: number; offsetY: number; scale: number; rotationX: number; rotationY: number } {
    const normalizedParam = sceneParam.toLowerCase()
    return SCENE_COORDINATE_TRANSFORMS[normalizedParam as keyof typeof SCENE_COORDINATE_TRANSFORMS] || SCENE_COORDINATE_TRANSFORMS.nycu
}

/**
 * 檢查場景參數是否有效
 * @param sceneParam 前端路由參數
 * @returns 是否為有效的場景參數
 */
export function isValidScene(sceneParam: string): boolean {
    const normalizedParam = sceneParam.toLowerCase()
    return normalizedParam in SCENE_MAPPING
} 
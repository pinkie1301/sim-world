import logging
import json
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm.attributes import flag_modified

from app.domains.drone_tracking.models.drone_tracking_model import (
    DronePosition,
    DroneTrackingMatrix,
    DroneTrackingSession,
    DroneTrackingExport,
    DroneTrackingStats,
)
from app.domains.drone_tracking.interfaces.drone_tracking_service_interface import (
    DroneTrackingServiceInterface,
)

logger = logging.getLogger(__name__)

# Scene configuration - based on frontend/src/utils/sceneUtils.ts
SCENE_CONFIG = {
    "nycu": {
        "bounds": {"min_x": -520, "max_x": 560, "min_y": -370, "max_y": 410},
        "resolution": 4.0,  # 4 meters per cell for manageable matrix size
        "matrix_size": 270,  # Approximately 1080/4 = 270 to cover full scene
        "offset_x": 865,
        "offset_y": 640,
        "scale": 1.0
    },
    "lotus": {
        "bounds": {"min_x": -64, "max_x": 64, "min_y": -64, "max_y": 64},
        "resolution": 1.0,
        "matrix_size": 128,
        "offset_x": 1200,
        "offset_y": 900,
        "scale": 1.0
    },
    "ntpu": {
        "bounds": {"min_x": -64, "max_x": 64, "min_y": -64, "max_y": 64},
        "resolution": 1.0,
        "matrix_size": 128,
        "offset_x": 900,
        "offset_y": 620,
        "scale": 1.0
    },
    "nanliao": {
        "bounds": {"min_x": -64, "max_x": 64, "min_y": -64, "max_y": 64},
        "resolution": 1.0,
        "matrix_size": 128,
        "offset_x": 920,
        "offset_y": 600,
        "scale": 1.0
    },
    "testscene": {
    "bounds": {"min_x": -256, "max_x": 256, "min_y": -256, "max_y": 256},  # area_m=512
    "resolution": 4.0,         # pixel_size_m from scene_meta.json
    "matrix_size": 128,        # grid_res from scene_meta.json
    "offset_x": 64,            # 需根據紋理圖的中心計算
    "offset_y": 64,
    "scale": 1.0
    }
}


class DroneTrackingService(DroneTrackingServiceInterface):
    """Service for tracking drone movements and generating coverage matrices."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def record_position(
        self,
        scene_name: str,
        scene_x: float,
        scene_y: float,
        scene_z: float
    ) -> bool:
        """Record a drone position in the tracking matrix."""
        try:
            logger.info(f"Recording position for {scene_name}: scene({scene_x:.2f}, {scene_y:.2f}, {scene_z:.2f})")
            
            # Get or create tracking session
            session = await self._get_or_create_session(scene_name)
            
            # Convert scene coordinates to matrix indices
            matrix_x, matrix_y = await self.convert_scene_to_matrix_coords(
                scene_name, scene_x, scene_y
            )
            
            logger.info(f"Coordinate conversion: scene({scene_x:.2f}, {scene_y:.2f}) -> matrix({matrix_x}, {matrix_y})")
            
            # Check if coordinates are within bounds
            config = SCENE_CONFIG.get(scene_name)
            if not config:
                logger.error(f"Unknown scene: {scene_name}")
                return False
            
            matrix_size = config["matrix_size"]
            if 0 <= matrix_x < matrix_size and 0 <= matrix_y < matrix_size:
                # Load current matrix
                matrix = session.get_matrix()
                
                # Mark position as visited and always increment position count
                matrix[matrix_y][matrix_x] = 1
                session.set_matrix(matrix)
                # Force ORM to detect JSON field changes
                flag_modified(session, "matrix_data")
                session.position_count += 1
                session.updated_at = datetime.utcnow()
                
                # Explicitly add session to ensure ORM detects changes
                self.db_session.add(session)
                await self.db_session.commit()
                
                logger.info(
                    f"Recorded position for {scene_name}: "
                    f"scene({scene_x:.2f}, {scene_y:.2f}) -> "
                    f"matrix({matrix_x}, {matrix_y}), "
                    f"total_positions: {session.position_count}"
                )
                
                return True
            else:
                logger.warning(
                    f"Position out of bounds for {scene_name}: "
                    f"matrix({matrix_x}, {matrix_y}) not in [0, {matrix_size})"
                )
                return False
                
        except Exception as e:
            logger.error(f"Error recording position: {e}")
            await self.db_session.rollback()
            return False
    
    async def get_tracking_matrix(
        self,
        scene_name: str
    ) -> Optional[DroneTrackingMatrix]:
        """Get the current tracking matrix for a scene."""
        try:
            session = await self._get_session(scene_name)
            if not session:
                return None
            
            return DroneTrackingMatrix(
                scene_name=session.scene_name,
                matrix_size=session.matrix_size,
                resolution=session.resolution,
                matrix=session.get_matrix(),
                bounds=session.get_bounds(),
                created_at=session.created_at,
                updated_at=session.updated_at
            )
            
        except Exception as e:
            logger.error(f"Error getting tracking matrix: {e}")
            return None
    
    async def clear_tracking_matrix(
        self,
        scene_name: str
    ) -> bool:
        """Clear the tracking matrix for a scene."""
        try:
            # Delete existing session
            stmt = delete(DroneTrackingSession).where(
                DroneTrackingSession.scene_name == scene_name
            )
            await self.db_session.execute(stmt)
            await self.db_session.commit()
            
            logger.info(f"Cleared tracking matrix for {scene_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing tracking matrix: {e}")
            await self.db_session.rollback()
            return False
    
    async def export_tracking_data(
        self,
        scene_name: str,
        export_format: str = "json"
    ) -> Optional[DroneTrackingExport]:
        """Export tracking data in specified format."""
        try:
            session = await self._get_session(scene_name)
            if not session:
                return None
            
            return DroneTrackingExport(
                scene_name=session.scene_name,
                matrix_size=session.matrix_size,
                resolution=session.resolution,
                matrix=session.get_matrix(),
                bounds=session.get_bounds(),
                position_count=session.position_count,
                export_timestamp=datetime.utcnow(),
                export_format=export_format
            )
            
        except Exception as e:
            logger.error(f"Error exporting tracking data: {e}")
            return None
    
    async def get_tracking_stats(
        self,
        scene_name: str
    ) -> Optional[DroneTrackingStats]:
        """Get tracking statistics for a scene."""
        try:
            session = await self._get_session(scene_name)
            if not session:
                return None
            
            matrix = session.get_matrix()
            bounds = session.get_bounds()
            
            # Calculate statistics
            visited_cells = sum(sum(row) for row in matrix)
            total_cells = session.matrix_size * session.matrix_size
            coverage_percentage = (visited_cells / total_cells) * 100 if total_cells > 0 else 0
            
            # Calculate approximate path length (simplified)
            path_length = visited_cells * session.resolution
            
            # Calculate session duration
            session_duration = (session.updated_at - session.created_at).total_seconds()
            
            return DroneTrackingStats(
                scene_name=session.scene_name,
                total_positions=session.position_count,
                visited_cells=visited_cells,
                coverage_percentage=coverage_percentage,
                path_length=path_length,
                session_duration=session_duration,
                bounds=bounds
            )
            
        except Exception as e:
            logger.error(f"Error getting tracking stats: {e}")
            return None
    
    async def convert_scene_to_matrix_coords(
        self,
        scene_name: str,
        scene_x: float,
        scene_y: float
    ) -> Tuple[int, int]:
        """Convert scene coordinates to matrix indices."""
        config = SCENE_CONFIG.get(scene_name)
        if not config:
            raise ValueError(f"Unknown scene: {scene_name}")
        
        bounds = config["bounds"]
        matrix_size = config["matrix_size"]
        resolution = config["resolution"]
        
        # Convert scene coordinates to matrix indices
        # Scene coordinates are centered at (0,0), matrix indices start at (0,0)
        matrix_x = int((scene_x - bounds["min_x"]) / resolution)
        matrix_y = int((scene_y - bounds["min_y"]) / resolution)
        
        # Clamp to valid range
        matrix_x = max(0, min(matrix_size - 1, matrix_x))
        matrix_y = max(0, min(matrix_size - 1, matrix_y))
        
        return matrix_x, matrix_y
    
    async def convert_matrix_to_scene_coords(
        self,
        scene_name: str,
        matrix_x: int,
        matrix_y: int
    ) -> Tuple[float, float]:
        """Convert matrix indices to scene coordinates."""
        config = SCENE_CONFIG.get(scene_name)
        if not config:
            raise ValueError(f"Unknown scene: {scene_name}")
        
        bounds = config["bounds"]
        resolution = config["resolution"]
        
        # Convert matrix indices to scene coordinates
        scene_x = bounds["min_x"] + (matrix_x * resolution)
        scene_y = bounds["min_y"] + (matrix_y * resolution)
        
        return scene_x, scene_y
    
    async def _get_session(self, scene_name: str) -> Optional[DroneTrackingSession]:
        """Get existing tracking session for a scene."""
        stmt = select(DroneTrackingSession).where(
            DroneTrackingSession.scene_name == scene_name
        )
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _get_or_create_session(self, scene_name: str) -> DroneTrackingSession:
        """Get or create tracking session for a scene."""
        session = await self._get_session(scene_name)
        
        if session is None:
            # Create new session
            config = SCENE_CONFIG.get(scene_name)
            if not config:
                raise ValueError(f"Unknown scene: {scene_name}")
            
            matrix_size = config["matrix_size"]
            resolution = config["resolution"]
            bounds = config["bounds"]
            
            # Initialize empty matrix
            matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
            
            session = DroneTrackingSession(
                scene_name=scene_name,
                matrix_size=matrix_size,
                resolution=resolution,
                matrix_data=json.dumps(matrix),
                bounds_data=json.dumps(bounds),
                position_count=0
            )
            
            self.db_session.add(session)
            await self.db_session.commit()
            
            logger.info(f"Created new tracking session for {scene_name}")
        
        return session

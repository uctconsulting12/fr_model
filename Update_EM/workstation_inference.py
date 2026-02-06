"""
Workstation Occupancy Inference System

Real-time person detection and workstation monitoring using YOLOv8
"""

import cv2
import time
import numpy as np
import psycopg2
import threading
import copy
from ultralytics import YOLO
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class WorkstationState:
    """Represents the current state of a workstation"""
    name: str
    roi: Tuple[int, int, int, int]  # x1, y1, x2, y2
    
    # Detection state
    occupied: bool = False
    status: str = "VACANT"
    confidence: float = 0.0
    
    # Timing
    last_seen_time: Optional[float] = None
    first_seen_time: Optional[float] = None
    last_present_time: Optional[float] = None
    last_status_change: Optional[float] = None
    
    # Metrics
    time_active: float = 0.0
    time_vacant: float = 0.0
    missing_count: int = 0
    missing_duration: float = 0.0
    
    # Internal tracking
    _current_vacancy_start: Optional[float] = None
    _last_status: str = "VACANT"
    _last_update: Optional[float] = None


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str = "localhost"
    port: int = 5432
    dbname: str = "test"
    user: str = "postgres"
    password: str = "admin123"


class WorkstationInference:
    """
    Real-time workstation occupancy detection and tracking
    """
    
    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        confidence_threshold: float = 0.4,
        missing_threshold: float = 3.0,
        visual_countdown: float = 3.0,
        db_config: Optional[DatabaseConfig] = None,
        org_id: Optional[int] = None,
        cam_id: Optional[int] = None,
        db_update_interval: float = 10.0,
        auto_update_db: bool = True
    ):
        """
        Initialize the inference system
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for person detection
            missing_threshold: Seconds before marking as vacant
            visual_countdown: Seconds to show countdown in UI
            db_config: Database configuration (optional)
            org_id: Organization ID for loading workstations from DB
            cam_id: Camera ID for loading workstations from DB
            db_update_interval: Seconds between database updates (default: 10.0)
            auto_update_db: Automatically update database (default: True)
        """
        print("⏳ Loading YOLO model...")
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.missing_threshold = missing_threshold
        self.visual_countdown = visual_countdown
        
        self.workstations: Dict[str, WorkstationState] = {}
        self.start_time = time.time()
        
        # Database configuration
        self.db_config = db_config or DatabaseConfig()
        self.org_id = org_id
        self.cam_id = cam_id
        self.db_update_interval = db_update_interval
        self.auto_update_db = auto_update_db
        self.last_db_update = 0
        self.current_date = datetime.now().date()
        
        print("✅ Model loaded successfully")
        
        # Auto-load workstations if org_id and cam_id provided
        if org_id is not None and cam_id is not None:
            self.load_workstations_from_db(org_id, cam_id)
            
            if self.auto_update_db and self.workstations:
                # Ensure daily rows exist
                self._create_daily_rows(self.current_date)
                # Hydrate state from existing data
                self._hydrate_state_from_db(self.current_date)
                print(f"✅ Auto-update enabled (every {db_update_interval}s)")
    
    def _get_db_connection(self):
        """Create and return a database connection"""
        return psycopg2.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            dbname=self.db_config.dbname,
            user=self.db_config.user,
            password=self.db_config.password
        )
    
    def load_workstations_from_db(
        self, 
        org_id: int, 
        cam_id: int
    ) -> None:
        """
        Load workstations from database for specified organization and camera
        
        Args:
            org_id: Organization ID
            cam_id: Camera ID
        """
        try:
            print(f"⏳ Loading workstations from database (org_id={org_id}, cam_id={cam_id})...")
            
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT name, x1, y1, x2, y2
                FROM workstations
                WHERE org_id=%s AND cam_id=%s
                ORDER BY name
            """, (org_id, cam_id))
            
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            if not rows:
                print(f"⚠️  No workstations found for org_id={org_id}, cam_id={cam_id}")
                return
            
            # Clear existing workstations
            self.workstations.clear()
            
            # Load from database
            for name, x1, y1, x2, y2 in rows:
                self.workstations[name] = WorkstationState(
                    name=name, 
                    roi=(x1, y1, x2, y2)
                )
            
            print(f"✅ Loaded {len(rows)} workstations from database")
            for name in self.workstations.keys():
                print(f"   📍 {name}")
                
        except Exception as e:
            print(f"❌ Database Error: {e}")
            raise
    
    def _create_daily_rows(self, analytics_date):
        """
        Create daily analytics rows for all workstations (if not exist)
        
        Args:
            analytics_date: Date object for analytics
        """
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            for ws_name in self.workstations.keys():
                cur.execute("""
                    INSERT INTO workstation_daily_analytics (
                        org_id, cam_id, workstation_name,
                        analytics_date,
                        active_seconds, vacant_seconds,
                        utilization_percent,
                        missing_count, missing_duration
                    )
                    VALUES (%s, %s, %s, %s, 0, 0, 0, 0, 0)
                    ON CONFLICT (org_id, cam_id, workstation_name, analytics_date) 
                    DO NOTHING
                """, (self.org_id, self.cam_id, ws_name, analytics_date))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Error creating daily rows: {e}")
    
    def _hydrate_state_from_db(self, analytics_date):
        """
        Load existing metrics from database to continue tracking
        
        Args:
            analytics_date: Date object for analytics
        """
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT workstation_name, active_seconds, vacant_seconds, 
                       first_seen_time, missing_count, missing_duration
                FROM workstation_daily_analytics
                WHERE org_id=%s AND cam_id=%s AND analytics_date=%s
            """, (self.org_id, self.cam_id, analytics_date))
            
            rows = cur.fetchall()
            loaded_count = 0
            
            for row in rows:
                name, active, vacant, first_seen_db, m_count, m_dur = row
                
                if name in self.workstations:
                    ws = self.workstations[name]
                    ws.time_active = float(active) if active else 0.0
                    ws.time_vacant = float(vacant) if vacant else 0.0
                    ws.missing_count = m_count if m_count else 0
                    ws.missing_duration = float(m_dur) if m_dur else 0.0
                    
                    if first_seen_db is not None:
                        full_dt = datetime.combine(analytics_date, first_seen_db)
                        ws.first_seen_time = full_dt.timestamp()
                    
                    loaded_count += 1
            
            if loaded_count > 0:
                print(f"✅ Hydrated state for {loaded_count} workstations from database")
            
            cur.close()
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Hydration error: {e}")
    
    def _update_database(self, analytics_date):
        """
        Update database with current metrics (runs in background thread)
        
        Args:
            analytics_date: Date object for analytics
        """
        try:
            # Create deep copy of state for thread safety
            ws_state_copy = {}
            for name, ws in self.workstations.items():
                ws_state_copy[name] = {
                    'time_active': ws.time_active,
                    'time_vacant': ws.time_vacant,
                    'missing_count': ws.missing_count,
                    'missing_duration': ws.missing_duration,
                    'first_seen_time': ws.first_seen_time,
                    'last_present_time': ws.last_present_time
                }
            
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            for ws_name, data in ws_state_copy.items():
                total = data['time_active'] + data['time_vacant']
                util = (data['time_active'] / total * 100) if total > 0 else 0
                
                # Build dynamic SQL
                sql = """
                    UPDATE workstation_daily_analytics
                    SET
                        active_seconds=%s,
                        vacant_seconds=%s,
                        utilization_percent=%s,
                        missing_count=%s,
                        missing_duration=%s,
                        updated_at=NOW()
                """
                params = [
                    round(data['time_active'], 2),
                    round(data['time_vacant'], 2),
                    round(util, 2),
                    data['missing_count'],
                    round(data['missing_duration'], 2)
                ]
                
                # Add first_seen_time if available
                if data['first_seen_time'] is not None:
                    sql += ", first_seen_time=%s"
                    params.append(datetime.fromtimestamp(data['first_seen_time']).time())
                
                # Add last_present_time if available
                if data['last_present_time'] is not None:
                    sql += ", last_present_time=%s"
                    params.append(datetime.fromtimestamp(data['last_present_time']).time())
                
                sql += """
                    WHERE org_id=%s AND cam_id=%s 
                    AND workstation_name=%s 
                    AND analytics_date=%s
                """
                params.extend([self.org_id, self.cam_id, ws_name, analytics_date])
                
                cur.execute(sql, tuple(params))
            
            conn.commit()
            cur.close()
            conn.close()
            
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] 💾 Database updated for {len(ws_state_copy)} workstations")
            
        except Exception as e:
            print(f"⚠️  Database update failed: {e}")
    
    def add_workstation(
        self, 
        name: str, 
        roi: Tuple[int, int, int, int]
    ) -> None:
        """
        Add a workstation region of interest
        
        Args:
            name: Workstation identifier
            roi: Bounding box (x1, y1, x2, y2)
        """
        self.workstations[name] = WorkstationState(name=name, roi=roi)
        print(f"📍 Added workstation: {name}")
    
    def load_workstations_from_config(
        self, 
        config: Dict[str, Tuple[int, int, int, int]]
    ) -> None:
        """
        Load multiple workstations from configuration dictionary
        
        Args:
            config: Dict mapping workstation names to ROI coordinates
        """
        for name, roi in config.items():
            self.add_workstation(name, roi)
    
    def _detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect person centroids in frame
        
        Args:
            frame: Input video frame
            
        Returns:
            List of (cx, cy) centroid coordinates
        """
        results = self.model(
            frame, 
            conf=self.conf_threshold, 
            classes=[0],  # Person class
            verbose=False
        )
        
        centroids = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                conf = float(box.conf[0])
                centroids.append((cx, cy, conf))
        
        return centroids
    
    def _is_point_in_roi(
        self, 
        point: Tuple[int, int], 
        roi: Tuple[int, int, int, int]
    ) -> bool:
        """Check if point is inside ROI"""
        cx, cy = point
        x1, y1, x2, y2 = roi
        return x1 <= cx <= x2 and y1 <= cy <= y2
    
    def _update_workstation_state(
        self, 
        ws: WorkstationState, 
        now: float
    ) -> None:
        """
        Update workstation state based on detection and timing
        
        Args:
            ws: Workstation state object
            now: Current timestamp
        """
        # Initialize timing
        if ws._last_update is None:
            ws._last_update = now
            ws.last_status_change = now
            return
        
        elapsed = now - ws._last_update
        
        # Determine current status with hysteresis
        if ws.occupied:
            current_status = "ACTIVE"
        elif ws.last_seen_time and (now - ws.last_seen_time) < self.missing_threshold:
            current_status = "ACTIVE"  # Grace period
        else:
            current_status = "VACANT"
        
        # Accumulate time
        if ws._last_status == "ACTIVE":
            ws.time_active += elapsed
        else:
            ws.time_vacant += elapsed
            if ws.first_seen_time is not None:
                ws.missing_duration += elapsed
        
        # Handle state transitions
        if current_status == "VACANT" and ws._last_status == "ACTIVE":
            # Just left
            ws.last_present_time = ws.last_seen_time
            if ws.first_seen_time is not None:
                ws.missing_count += 1
                ws._current_vacancy_start = now
                
        elif current_status == "ACTIVE" and ws._last_status == "VACANT":
            # Just returned
            if ws._current_vacancy_start is not None:
                vacancy_duration = now - ws._current_vacancy_start
                ws._current_vacancy_start = None
        
        # Update status
        if current_status != ws._last_status:
            ws.last_status_change = now
        
        ws.status = current_status
        ws._last_status = current_status
        ws._last_update = now
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        timestamp: Optional[float] = None
    ) -> np.ndarray:
        """
        Process a single frame and update all workstation states
        Automatically updates database if enabled
        
        Args:
            frame: Input video frame
            timestamp: Optional explicit timestamp (default: current time)
            
        Returns:
            Annotated frame with detection visualizations
        """
        now = timestamp if timestamp is not None else time.time()
        today = datetime.fromtimestamp(now).date()
        
        # Handle date rollover
        if today != self.current_date:
            print(f"📅 Date rollover detected: {self.current_date} -> {today}")
            
            # Final update for previous day
            if self.auto_update_db:
                self._update_database(self.current_date)
            
            # Reset for new day
            self.current_date = today
            for ws in self.workstations.values():
                ws.time_active = 0.0
                ws.time_vacant = 0.0
                ws.missing_count = 0
                ws.missing_duration = 0.0
                ws.first_seen_time = None
                ws._current_vacancy_start = None
                if ws.occupied:
                    ws.first_seen_time = now
            
            if self.auto_update_db:
                self._create_daily_rows(self.current_date)
        
        # Reset occupancy flags
        for ws in self.workstations.values():
            ws.occupied = False
            ws.confidence = 0.0
        
        # Detect persons
        centroids = self._detect_persons(frame)
        
        # Check occupancy for each workstation
        for cx, cy, conf in centroids:
            for ws in self.workstations.values():
                if self._is_point_in_roi((cx, cy), ws.roi):
                    ws.occupied = True
                    ws.confidence = max(ws.confidence, conf)
                    ws.last_seen_time = now
                    if ws.first_seen_time is None:
                        ws.first_seen_time = now
        
        # Update all workstation states
        for ws in self.workstations.values():
            self._update_workstation_state(ws, now)
        
        # Automatic database update
        if self.auto_update_db and self.org_id is not None and self.cam_id is not None:
            if now - self.last_db_update >= self.db_update_interval:
                # Run in background thread
                threading.Thread(
                    target=self._update_database,
                    args=(self.current_date,),
                    daemon=True
                ).start()
                self.last_db_update = now
        
        # Draw annotations
        annotated_frame = self._draw_annotations(frame, now)
        
        return annotated_frame
    
    def _draw_annotations(
        self, 
        frame: np.ndarray, 
        now: float
    ) -> np.ndarray:
        """
        Draw workstation ROIs and status information
        
        Args:
            frame: Input frame
            now: Current timestamp
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        for ws in self.workstations.values():
            x1, y1, x2, y2 = ws.roi
            is_active = (ws.status == "ACTIVE")
            
            # Color coding
            color = (0, 255, 0) if is_active else (0, 0, 255)
            
            # Draw ROI rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Workstation name
            cv2.putText(
                output, ws.name, 
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            
            # Status text
            status_text = f"{ws.status}"
            if ws.confidence > 0:
                status_text += f" ({ws.confidence:.2f})"
            cv2.putText(
                output, status_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Countdown timer (grace period)
            if is_active and not ws.occupied and ws.last_seen_time:
                elapsed_since_seen = now - ws.last_seen_time
                remaining = self.missing_threshold - elapsed_since_seen
                if remaining > 0:
                    cv2.putText(
                        output, f"Missing: {remaining:.1f}s",
                        (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2
                    )
            
            # Utilization stats
            total_time = ws.time_active + ws.time_vacant
            if total_time > 0:
                utilization = (ws.time_active / total_time) * 100
                cv2.putText(
                    output, f"Util: {utilization:.1f}%",
                    (x1, y2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
                )
        
        # Global timestamp
        timestamp_text = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        cv2.putText(
            output, timestamp_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        
        return output
    
    def get_metrics(self) -> Dict[str, Dict]:
        """
        Get current metrics for all workstations
        
        Returns:
            Dictionary mapping workstation names to their metrics
        """
        metrics = {}
        
        for name, ws in self.workstations.items():
            total_time = ws.time_active + ws.time_vacant
            utilization = (ws.time_active / total_time * 100) if total_time > 0 else 0
            
            metrics[name] = {
                "status": ws.status,
                "occupied": ws.occupied,
                "confidence": ws.confidence,
                "time_active": round(ws.time_active, 2),
                "time_vacant": round(ws.time_vacant, 2),
                "utilization_percent": round(utilization, 2),
                "missing_count": ws.missing_count,
                "missing_duration": round(ws.missing_duration, 2),
                "first_seen": datetime.fromtimestamp(ws.first_seen_time).strftime("%H:%M:%S") 
                             if ws.first_seen_time else None,
                "last_seen": datetime.fromtimestamp(ws.last_seen_time).strftime("%H:%M:%S")
                            if ws.last_seen_time else None,
            }
        
        return metrics
    
    def print_metrics(self) -> None:
        """Print current metrics to console"""
        metrics = self.get_metrics()
        
        print("\n" + "="*80)
        print("WORKSTATION METRICS")
        print("="*80)
        
        for name, data in metrics.items():
            print(f"\n📍 {name}:")
            print(f"   Status: {data['status']} {'🟢' if data['occupied'] else '🔴'}")
            print(f"   Active: {data['time_active']:.1f}s | Vacant: {data['time_vacant']:.1f}s")
            print(f"   Utilization: {data['utilization_percent']:.1f}%")
            print(f"   Missing: {data['missing_count']}x ({data['missing_duration']:.1f}s)")
            if data['first_seen']:
                print(f"   First Seen: {data['first_seen']} | Last Seen: {data['last_seen']}")
        
        print("="*80 + "\n")
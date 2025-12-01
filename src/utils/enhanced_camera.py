"""
Enhanced Camera Manager for LESSA with quality assessment and multi-camera support.
Provides automatic camera detection, quality analysis, and optimization for sign language recognition.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from .config import config

@dataclass
class CameraInfo:
    """Information about a camera device."""
    device_id: int
    name: str
    max_width: int
    max_height: int
    max_fps: float
    quality_score: float
    is_available: bool
    supports_autofocus: bool
    latency_ms: float

@dataclass
class CameraQuality:
    """Camera quality assessment results."""
    resolution_score: float  # 0-100
    frame_rate_score: float  # 0-100
    latency_score: float     # 0-100
    stability_score: float   # 0-100
    brightness_score: float  # 0-100
    sharpness_score: float   # 0-100
    overall_score: float     # 0-100
    recommendations: List[str]

class EnhancedCamera:
    """Enhanced camera with quality assessment and optimization for LESSA."""
    
    def __init__(self, device_id: Optional[int] = None):
        """Initialize enhanced camera."""
        camera_config = config.get_camera_config()
        
        self.device_id = device_id if device_id is not None else camera_config.get('device_id', 0)
        self.target_width = camera_config.get('width', 1280)  # Higher default for better quality
        self.target_height = camera_config.get('height', 720)
        self.target_fps = camera_config.get('fps', 30)
        
        self.camera = None
        self.is_opened = False
        self.current_quality = None
        self.frame_buffer = []
        self.performance_metrics = {
            'avg_process_time': 0.0,
            'frame_drops': 0,
            'total_frames': 0
        }
    
    def start(self, optimize_quality: bool = True) -> bool:
        """Start camera with optional quality optimization."""
        try:
            self.camera = cv2.VideoCapture(self.device_id)
            
            if not self.camera.isOpened():
                print(f"❌ Could not open camera {self.device_id}")
                return False
            
            # Try to set optimal properties
            self._set_optimal_properties()
            
            # Verify actual settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.is_opened = True
            
            print(f"📹 Camera {self.device_id} started: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            
            # Assess quality if requested
            if optimize_quality:
                print("🔍 Assessing camera quality...")
                self.current_quality = self.assess_quality()
                self._apply_quality_optimizations()
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def _set_optimal_properties(self):
        """Set optimal camera properties for LESSA."""
        # Resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        
        # Frame rate
        self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Quality settings
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Partial auto exposure
        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        self.camera.set(cv2.CAP_PROP_CONTRAST, 0.5)
        self.camera.set(cv2.CAP_PROP_SATURATION, 0.5)
        
        # Buffer size for low latency
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame with performance tracking."""
        if not self.is_opened or self.camera is None:
            return False, None
        
        start_time = time.time()
        
        ret, frame = self.camera.read()
        if ret:
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Track performance
            process_time = (time.time() - start_time) * 1000  # ms
            self.performance_metrics['total_frames'] += 1
            self.performance_metrics['avg_process_time'] = (
                (self.performance_metrics['avg_process_time'] * (self.performance_metrics['total_frames'] - 1) + process_time) /
                self.performance_metrics['total_frames']
            )
            
            # Add to buffer for quality assessment
            if len(self.frame_buffer) < 10:
                self.frame_buffer.append(frame.copy())
            else:
                self.frame_buffer.pop(0)
                self.frame_buffer.append(frame.copy())
        else:
            self.performance_metrics['frame_drops'] += 1
        
        return ret, frame
    
    def assess_quality(self) -> CameraQuality:
        """Assess camera quality for sign language recognition."""
        if not self.is_opened:
            return None
        
        print("📊 Running quality assessment...")
        
        # Collect test frames
        test_frames = []
        latencies = []
        
        for i in range(30):  # Test with 30 frames
            start_time = time.time()
            ret, frame = self.camera.read()
            if ret:
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                test_frames.append(frame)
            time.sleep(0.033)  # ~30fps
        
        if not test_frames:
            return None
        
        # Calculate quality metrics
        resolution_score = self._assess_resolution(test_frames[0])
        frame_rate_score = self._assess_frame_rate()
        latency_score = self._assess_latency(latencies)
        stability_score = self._assess_stability(test_frames)
        brightness_score = self._assess_brightness(test_frames)
        sharpness_score = self._assess_sharpness(test_frames)
        
        # Calculate overall score
        weights = {
            'resolution': 0.2,
            'frame_rate': 0.15,
            'latency': 0.15,
            'stability': 0.2,
            'brightness': 0.15,
            'sharpness': 0.15
        }
        
        overall_score = (
            resolution_score * weights['resolution'] +
            frame_rate_score * weights['frame_rate'] +
            latency_score * weights['latency'] +
            stability_score * weights['stability'] +
            brightness_score * weights['brightness'] +
            sharpness_score * weights['sharpness']
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations({
            'resolution': resolution_score,
            'frame_rate': frame_rate_score,
            'latency': latency_score,
            'stability': stability_score,
            'brightness': brightness_score,
            'sharpness': sharpness_score
        })
        
        quality = CameraQuality(
            resolution_score=resolution_score,
            frame_rate_score=frame_rate_score,
            latency_score=latency_score,
            stability_score=stability_score,
            brightness_score=brightness_score,
            sharpness_score=sharpness_score,
            overall_score=overall_score,
            recommendations=recommendations
        )
        
        print(f"📊 Quality Assessment Complete: {overall_score:.1f}/100")
        return quality
    
    def _assess_resolution(self, frame: np.ndarray) -> float:
        """Assess resolution quality."""
        height, width = frame.shape[:2]
        total_pixels = height * width
        
        # Score based on total pixels
        if total_pixels >= 1920 * 1080:  # 1080p+
            return 100.0
        elif total_pixels >= 1280 * 720:  # 720p
            return 80.0
        elif total_pixels >= 640 * 480:   # 480p
            return 60.0
        else:
            return 40.0
    
    def _assess_frame_rate(self) -> float:
        """Assess frame rate quality."""
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        if actual_fps >= 60:
            return 100.0
        elif actual_fps >= 30:
            return 80.0
        elif actual_fps >= 24:
            return 60.0
        else:
            return 40.0
    
    def _assess_latency(self, latencies: List[float]) -> float:
        """Assess latency quality."""
        if not latencies:
            return 0.0
        
        avg_latency = np.mean(latencies)
        
        if avg_latency <= 20:      # < 20ms excellent
            return 100.0
        elif avg_latency <= 50:    # < 50ms good
            return 80.0
        elif avg_latency <= 100:   # < 100ms acceptable
            return 60.0
        else:                      # > 100ms poor
            return 30.0
    
    def _assess_stability(self, frames: List[np.ndarray]) -> float:
        """Assess frame stability (consistency)."""
        if len(frames) < 2:
            return 0.0
        
        # Calculate frame-to-frame differences
        differences = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i-1], frames[i])
            mean_diff = np.mean(diff)
            differences.append(mean_diff)
        
        # Lower variance = more stable
        variance = np.var(differences)
        stability = max(0, 100 - variance * 2)  # Scale appropriately
        
        return min(100.0, stability)
    
    def _assess_brightness(self, frames: List[np.ndarray]) -> float:
        """Assess brightness quality."""
        if not frames:
            return 0.0
        
        # Calculate average brightness
        brightnesses = [np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) for frame in frames]
        avg_brightness = np.mean(brightnesses)
        
        # Optimal brightness is around 120-140 (out of 255)
        optimal_range = (100, 150)
        
        if optimal_range[0] <= avg_brightness <= optimal_range[1]:
            return 100.0
        elif 80 <= avg_brightness <= 180:
            return 70.0
        elif 50 <= avg_brightness <= 200:
            return 50.0
        else:
            return 30.0
    
    def _assess_sharpness(self, frames: List[np.ndarray]) -> float:
        """Assess sharpness/focus quality."""
        if not frames:
            return 0.0
        
        # Use Laplacian variance to measure sharpness
        sharpness_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_scores.append(sharpness)
        
        avg_sharpness = np.mean(sharpness_scores)
        
        # Scale sharpness score
        if avg_sharpness >= 1000:
            return 100.0
        elif avg_sharpness >= 500:
            return 80.0
        elif avg_sharpness >= 200:
            return 60.0
        else:
            return 40.0
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if scores['resolution'] < 70:
            recommendations.append("📐 Consider using a higher resolution camera for better landmark detection")
        
        if scores['frame_rate'] < 70:
            recommendations.append("⚡ Increase frame rate or use a camera with higher FPS capability")
        
        if scores['latency'] < 70:
            recommendations.append("🚀 Reduce latency by closing other applications or using USB 3.0")
        
        if scores['brightness'] < 70:
            recommendations.append("💡 Improve lighting conditions or adjust camera brightness")
        
        if scores['sharpness'] < 70:
            recommendations.append("🔍 Clean camera lens or enable autofocus for sharper images")
        
        if scores['stability'] < 70:
            recommendations.append("📹 Ensure stable camera mounting and consistent lighting")
        
        if not recommendations:
            recommendations.append("✅ Camera quality is excellent for LESSA!")
        
        return recommendations
    
    def _apply_quality_optimizations(self):
        """Apply optimizations based on quality assessment."""
        if not self.current_quality:
            return
        
        print("⚙️  Applying quality optimizations...")
        
        # Adjust based on quality scores
        if self.current_quality.brightness_score < 70:
            # Try to improve brightness
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
        
        if self.current_quality.sharpness_score < 70:
            # Ensure autofocus is enabled
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        if self.current_quality.latency_score < 70:
            # Reduce buffer size for lower latency
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def get_quality_report(self) -> str:
        """Get formatted quality report."""
        if not self.current_quality:
            return "Quality assessment not available"
        
        report = f"""
🎯 LESSA Camera Quality Report:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 Resolution:    {self.current_quality.resolution_score:.1f}/100
⚡ Frame Rate:    {self.current_quality.frame_rate_score:.1f}/100
🚀 Latency:       {self.current_quality.latency_score:.1f}/100
📹 Stability:     {self.current_quality.stability_score:.1f}/100
💡 Brightness:    {self.current_quality.brightness_score:.1f}/100
🔍 Sharpness:     {self.current_quality.sharpness_score:.1f}/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎖️  Overall:      {self.current_quality.overall_score:.1f}/100

📋 Recommendations:
"""
        
        for i, rec in enumerate(self.current_quality.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def stop(self):
        """Stop camera and release resources."""
        if self.camera is not None:
            self.camera.release()
            self.is_opened = False
            print("📹 Camera stopped and resources released")

class CameraManager:
    """Manager for detecting and selecting cameras."""
    
    @staticmethod
    def detect_cameras() -> List[CameraInfo]:
        """Detect all available cameras and assess their capabilities."""
        print("🔍 Detecting available cameras...")
        
        cameras = []
        
        # Check first 10 possible device IDs
        for device_id in range(10):
            try:
                test_camera = cv2.VideoCapture(device_id)
                
                if test_camera.isOpened():
                    # Test if we can actually read frames
                    ret, frame = test_camera.read()
                    
                    if ret and frame is not None:
                        # Get camera properties
                        width = int(test_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(test_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = test_camera.get(cv2.CAP_PROP_FPS)
                        
                        # Quick quality assessment
                        start_time = time.time()
                        test_camera.read()
                        latency = (time.time() - start_time) * 1000
                        
                        # Calculate basic quality score
                        quality_score = CameraManager._calculate_basic_quality(width, height, fps, latency)
                        
                        camera_info = CameraInfo(
                            device_id=device_id,
                            name=f"Camera {device_id}",
                            max_width=width,
                            max_height=height,
                            max_fps=fps,
                            quality_score=quality_score,
                            is_available=True,
                            supports_autofocus=True,  # Assume yes, will be tested later
                            latency_ms=latency
                        )
                        
                        cameras.append(camera_info)
                        print(f"✅ Found Camera {device_id}: {width}x{height} @ {fps:.1f}fps (Quality: {quality_score:.1f})")
                
                test_camera.release()
                
            except Exception as e:
                continue  # Skip failed cameras
        
        if not cameras:
            print("❌ No cameras detected")
        else:
            print(f"📹 Detected {len(cameras)} camera(s)")
        
        return sorted(cameras, key=lambda x: x.quality_score, reverse=True)
    
    @staticmethod
    def _calculate_basic_quality(width: int, height: int, fps: float, latency: float) -> float:
        """Calculate basic quality score for camera ranking."""
        # Resolution score
        pixels = width * height
        if pixels >= 1920 * 1080:
            res_score = 100
        elif pixels >= 1280 * 720:
            res_score = 80
        else:
            res_score = 60
        
        # FPS score
        if fps >= 60:
            fps_score = 100
        elif fps >= 30:
            fps_score = 80
        else:
            fps_score = 60
        
        # Latency score
        if latency <= 20:
            lat_score = 100
        elif latency <= 50:
            lat_score = 80
        else:
            lat_score = 60
        
        return (res_score + fps_score + lat_score) / 3
    
    @staticmethod
    def get_best_camera() -> Optional[CameraInfo]:
        """Get the best available camera for LESSA."""
        cameras = CameraManager.detect_cameras()
        return cameras[0] if cameras else None
    
    @staticmethod
    def format_camera_list(cameras: List[CameraInfo]) -> str:
        """Format camera list for display."""
        if not cameras:
            return "No cameras detected"
        
        output = "📹 Available Cameras:\n"
        output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        for i, cam in enumerate(cameras):
            status = "🟢" if cam.quality_score >= 80 else "🟡" if cam.quality_score >= 60 else "🔴"
            output += f"{status} Camera {cam.device_id}: {cam.max_width}x{cam.max_height} @ {cam.max_fps:.1f}fps\n"
            output += f"   Quality: {cam.quality_score:.1f}/100, Latency: {cam.latency_ms:.1f}ms\n\n"
        
        return output
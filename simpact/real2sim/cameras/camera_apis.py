from .sensor import Sensor
from klampt.math import se3,so3,vectorops
from dataclasses import dataclass
import dacite
from typing import Optional,List
from klampt.model.typing import RigidTransform

@dataclass
class CameraIntrinsics:
    """Intrinsics for a camera sensor. 
    """
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: Optional[str] = None
    distortion: Optional[List[float]] = None

@dataclass
class CameraExtrinics:
    """Extrinsics for a camera sensor.  If frame = None
    then it is assumed to be in the world frame.  Otherwise, this
    should name a robot link or object frame.
    """
    pose: RigidTransform
    frame : Optional[str] = None

@dataclass
class CameraSettings:
    """Settings for a generic mono or RGB camera sensor. 
    
    Color format can be 'rgb8', 'rgba8', 'bgr8', 'bgra8', 'u8',
    'u16', 'f32', or 'f64'.

    Rate is given in Hz.  If None, the rate is unknown.
    
    The intrinsics, if present, must contain 'fx', 'fy', 'cx', and 'cy',
    and optional 'distortion' parameters.  The
    extrinsics are assumed to be a Klampt se3 element.
    """
    width: int
    height: int
    color: bool
    color_format: str = 'rgb8'
    rate : Optional[float] = None
    intrinsics: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinics] = None


class Camera(Sensor):
    """Represents a mono or RGB camera sensor. 
    
    Settings can be passed to the constructor as a CameraSettings
    object or keyword arguments.
    """
    def settings_object(self) -> CameraSettings:
        return dacite.from_dict(CameraSettings, self.settings())

    def rate(self) -> Optional[float]:
        return self.settings_object().rate
    
    def channels(self) -> int:
        return 0
    

@dataclass
class StereoCameraSettings(CameraSettings):
    """Settings for a generic stereo pair. 
    
    The intrinsics are assumed to be for the left camera, and
    are assumed to be the same for both cameras unless
    right_intrinsics is defined. 
    
    The extrinsics are defined relative to the left camera.  If
    right_extrinsics is not defined, then the right camera
    is assumed to be offset by the baseline in the x direction. 
    """
    baseline: float = 0.0
    disparity_max: Optional[float] = None
    right_intrinsics: Optional[CameraIntrinsics] = None
    right_extrinsics: Optional[CameraExtrinics] = None


class StereoCamera(Sensor):
    """Represents a stereo camera sensor. 
    
    Settings can be passed to the constructor as a StereoCameraSettings
    object or keyword arguments.

    The value() method should return a dictionary with keys 'left' and
    'right'.  If the camera is not rectified, then value() can also
    provide the 'left_rectified' and 'right_rectified' channels. 
    """
    def settings_object(self) -> StereoCameraSettings:
        return dacite.from_dict(StereoCameraSettings, self.settings())
    
    def channels(self):
        return ['left', 'right']
    
    def rate(self) -> Optional[float]:
        return self.settings_object().rate
    
    def left_intrinsics(self) -> Optional[CameraIntrinsics]:
        return self.settings_object().intrinsics

    def left_extrinsics(self) -> Optional[CameraExtrinics]:
        return self.settings_object().extrinsics
    
    def right_intrinsics(self) -> Optional[CameraIntrinsics]:
        settings = self.settings_object()
        if settings.right_intrinsics is not None:
            return settings.right_intrinsics
        return settings.intrinsics
    
    def right_extrinsics(self) -> Optional[CameraExtrinics]:
        settings = self.settings_object()
        if settings.right_extrinsics is not None:
            return settings.right_extrinsics
        extrinsics = settings.extrinsics
        if extrinsics is None:
            return None
        R,t = extrinsics
        return (R, vectorops.add(t, so3.apply(R, [settings.baseline,0,0])))


class StereoCameraWithDepth(StereoCamera):
    """Represents a stereo camera sensor with a depth channel. 
    
    Settings can be passed to the constructor as a StereoCameraSettings
    object or keyword arguments.

    The value() method should return a dictionary with keys 'left',
    'right', and 'depth'.

    If the camera is not rectified, then value() can also
    provide the 'left_rectified' and 'right_rectified' channels.
    """
    def channels(self):
        return ['left', 'right', 'depth']


@dataclass
class RGBDCameraSettings:
    """Settings for a generic RGBD camera sensor. 
    
    The depth image is given by bits and should be scaled by the given.
    depth_scale to retrieve the depth in meters. The (scaled) depth is
    assumed to be in meters.

    If depth_aligned is True, the raw depth is aligned with the RGB image.
    Otherwise, the raw depth is not aligned and the sensor may or may not
    provide the depth_aligned image. 
    """
    rgb : CameraSettings
    depth: CameraSettings
    depth_scale: float = 1.0
    depth_min: float = 0.0
    depth_max: float = float('inf')
    depth_aligned: bool = False


class RGBDCamera(Sensor):
    """Represents a generic RGBD camera sensor. 
    
    Settings can be passed to the constructor as a CameraSettings
    object or keyword arguments.

    The value() method should return a dictionary with keys 'rgb' and
    'depth'.  If the depth is not aligned to the RGB image, then
    value() can also provide the 'depth_aligned' channel and/or the
    'rgb_aligned' channel (RGB aligned to depth).
    """
    def settings_object(self) -> RGBDCameraSettings:
        return dacite.from_dict(RGBDCameraSettings, self.settings())
    
    def channels(self):
        return ['rgb', 'depth']

    def rate(self) -> Optional[float]:
        return self.settings_object().rgb.rate

    def depth_intrinsics(self) -> Optional[CameraIntrinsics]:
        settings = self.settings_object()
        if settings.depth.intrinsics is None:
            if settings.depth_aligned:
                return settings.rgb.intrinsics
        return settings.depth.intrinsics

    def depth_extrinsics(self) -> Optional[CameraExtrinics]:
        settings = self.settings_object()
        if settings.depth.extrinsics is None:
            if settings.depth_aligned:
                return settings.rgb.extrinsics
        return settings.depth.extrinsics
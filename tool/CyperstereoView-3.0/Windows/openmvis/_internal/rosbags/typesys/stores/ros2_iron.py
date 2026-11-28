# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
#
# THIS FILE IS GENERATED, DO NOT EDIT
"""Message type definitions."""

# ruff: noqa: N801,N816,RUF100,TC004

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rosbags.interfaces import Nodetype as T

from . import ros2_humble as base

if TYPE_CHECKING:
    from typing import ClassVar

    import numpy as np

    from rosbags.interfaces.typing import Typesdict


action_msgs__msg__GoalInfo = base.action_msgs__msg__GoalInfo
action_msgs__msg__GoalStatus = base.action_msgs__msg__GoalStatus
action_msgs__msg__GoalStatusArray = base.action_msgs__msg__GoalStatusArray
actionlib_msgs__msg__GoalID = base.actionlib_msgs__msg__GoalID
actionlib_msgs__msg__GoalStatus = base.actionlib_msgs__msg__GoalStatus
actionlib_msgs__msg__GoalStatusArray = base.actionlib_msgs__msg__GoalStatusArray
builtin_interfaces__msg__Duration = base.builtin_interfaces__msg__Duration
builtin_interfaces__msg__Time = base.builtin_interfaces__msg__Time
diagnostic_msgs__msg__DiagnosticArray = base.diagnostic_msgs__msg__DiagnosticArray
diagnostic_msgs__msg__DiagnosticStatus = base.diagnostic_msgs__msg__DiagnosticStatus
diagnostic_msgs__msg__KeyValue = base.diagnostic_msgs__msg__KeyValue
geometry_msgs__msg__Accel = base.geometry_msgs__msg__Accel
geometry_msgs__msg__AccelStamped = base.geometry_msgs__msg__AccelStamped
geometry_msgs__msg__AccelWithCovariance = base.geometry_msgs__msg__AccelWithCovariance
geometry_msgs__msg__AccelWithCovarianceStamped = base.geometry_msgs__msg__AccelWithCovarianceStamped
geometry_msgs__msg__Inertia = base.geometry_msgs__msg__Inertia
geometry_msgs__msg__InertiaStamped = base.geometry_msgs__msg__InertiaStamped
geometry_msgs__msg__Point = base.geometry_msgs__msg__Point
geometry_msgs__msg__Point32 = base.geometry_msgs__msg__Point32
geometry_msgs__msg__PointStamped = base.geometry_msgs__msg__PointStamped
geometry_msgs__msg__Polygon = base.geometry_msgs__msg__Polygon
geometry_msgs__msg__PolygonStamped = base.geometry_msgs__msg__PolygonStamped
geometry_msgs__msg__Pose = base.geometry_msgs__msg__Pose
geometry_msgs__msg__Pose2D = base.geometry_msgs__msg__Pose2D
geometry_msgs__msg__PoseArray = base.geometry_msgs__msg__PoseArray
geometry_msgs__msg__PoseStamped = base.geometry_msgs__msg__PoseStamped
geometry_msgs__msg__PoseWithCovariance = base.geometry_msgs__msg__PoseWithCovariance
geometry_msgs__msg__PoseWithCovarianceStamped = base.geometry_msgs__msg__PoseWithCovarianceStamped
geometry_msgs__msg__Quaternion = base.geometry_msgs__msg__Quaternion
geometry_msgs__msg__QuaternionStamped = base.geometry_msgs__msg__QuaternionStamped
geometry_msgs__msg__Transform = base.geometry_msgs__msg__Transform
geometry_msgs__msg__TransformStamped = base.geometry_msgs__msg__TransformStamped
geometry_msgs__msg__Twist = base.geometry_msgs__msg__Twist
geometry_msgs__msg__TwistStamped = base.geometry_msgs__msg__TwistStamped
geometry_msgs__msg__TwistWithCovariance = base.geometry_msgs__msg__TwistWithCovariance
geometry_msgs__msg__TwistWithCovarianceStamped = base.geometry_msgs__msg__TwistWithCovarianceStamped
geometry_msgs__msg__Vector3 = base.geometry_msgs__msg__Vector3
geometry_msgs__msg__Vector3Stamped = base.geometry_msgs__msg__Vector3Stamped
geometry_msgs__msg__Wrench = base.geometry_msgs__msg__Wrench
geometry_msgs__msg__WrenchStamped = base.geometry_msgs__msg__WrenchStamped
lifecycle_msgs__msg__State = base.lifecycle_msgs__msg__State
lifecycle_msgs__msg__Transition = base.lifecycle_msgs__msg__Transition
lifecycle_msgs__msg__TransitionDescription = base.lifecycle_msgs__msg__TransitionDescription
lifecycle_msgs__msg__TransitionEvent = base.lifecycle_msgs__msg__TransitionEvent
nav_msgs__msg__GridCells = base.nav_msgs__msg__GridCells
nav_msgs__msg__MapMetaData = base.nav_msgs__msg__MapMetaData
nav_msgs__msg__OccupancyGrid = base.nav_msgs__msg__OccupancyGrid
nav_msgs__msg__Odometry = base.nav_msgs__msg__Odometry
nav_msgs__msg__Path = base.nav_msgs__msg__Path
rcl_interfaces__msg__FloatingPointRange = base.rcl_interfaces__msg__FloatingPointRange
rcl_interfaces__msg__IntegerRange = base.rcl_interfaces__msg__IntegerRange
rcl_interfaces__msg__ListParametersResult = base.rcl_interfaces__msg__ListParametersResult
rcl_interfaces__msg__Log = base.rcl_interfaces__msg__Log
rcl_interfaces__msg__Parameter = base.rcl_interfaces__msg__Parameter
rcl_interfaces__msg__ParameterDescriptor = base.rcl_interfaces__msg__ParameterDescriptor
rcl_interfaces__msg__ParameterEvent = base.rcl_interfaces__msg__ParameterEvent
rcl_interfaces__msg__ParameterEventDescriptors = base.rcl_interfaces__msg__ParameterEventDescriptors
rcl_interfaces__msg__ParameterType = base.rcl_interfaces__msg__ParameterType
rcl_interfaces__msg__ParameterValue = base.rcl_interfaces__msg__ParameterValue
rcl_interfaces__msg__SetParametersResult = base.rcl_interfaces__msg__SetParametersResult
rmw_dds_common__msg__NodeEntitiesInfo = base.rmw_dds_common__msg__NodeEntitiesInfo
rmw_dds_common__msg__ParticipantEntitiesInfo = base.rmw_dds_common__msg__ParticipantEntitiesInfo
rosbag2_interfaces__msg__ReadSplitEvent = base.rosbag2_interfaces__msg__ReadSplitEvent
rosbag2_interfaces__msg__WriteSplitEvent = base.rosbag2_interfaces__msg__WriteSplitEvent
rosgraph_msgs__msg__Clock = base.rosgraph_msgs__msg__Clock
sensor_msgs__msg__CameraInfo = base.sensor_msgs__msg__CameraInfo
sensor_msgs__msg__ChannelFloat32 = base.sensor_msgs__msg__ChannelFloat32
sensor_msgs__msg__CompressedImage = base.sensor_msgs__msg__CompressedImage
sensor_msgs__msg__FluidPressure = base.sensor_msgs__msg__FluidPressure
sensor_msgs__msg__Illuminance = base.sensor_msgs__msg__Illuminance
sensor_msgs__msg__Image = base.sensor_msgs__msg__Image
sensor_msgs__msg__Imu = base.sensor_msgs__msg__Imu
sensor_msgs__msg__JointState = base.sensor_msgs__msg__JointState
sensor_msgs__msg__Joy = base.sensor_msgs__msg__Joy
sensor_msgs__msg__JoyFeedback = base.sensor_msgs__msg__JoyFeedback
sensor_msgs__msg__JoyFeedbackArray = base.sensor_msgs__msg__JoyFeedbackArray
sensor_msgs__msg__LaserEcho = base.sensor_msgs__msg__LaserEcho
sensor_msgs__msg__LaserScan = base.sensor_msgs__msg__LaserScan
sensor_msgs__msg__MagneticField = base.sensor_msgs__msg__MagneticField
sensor_msgs__msg__MultiDOFJointState = base.sensor_msgs__msg__MultiDOFJointState
sensor_msgs__msg__MultiEchoLaserScan = base.sensor_msgs__msg__MultiEchoLaserScan
sensor_msgs__msg__NavSatFix = base.sensor_msgs__msg__NavSatFix
sensor_msgs__msg__NavSatStatus = base.sensor_msgs__msg__NavSatStatus
sensor_msgs__msg__PointCloud = base.sensor_msgs__msg__PointCloud
sensor_msgs__msg__PointCloud2 = base.sensor_msgs__msg__PointCloud2
sensor_msgs__msg__PointField = base.sensor_msgs__msg__PointField
sensor_msgs__msg__RegionOfInterest = base.sensor_msgs__msg__RegionOfInterest
sensor_msgs__msg__RelativeHumidity = base.sensor_msgs__msg__RelativeHumidity
sensor_msgs__msg__Temperature = base.sensor_msgs__msg__Temperature
sensor_msgs__msg__TimeReference = base.sensor_msgs__msg__TimeReference
shape_msgs__msg__Mesh = base.shape_msgs__msg__Mesh
shape_msgs__msg__MeshTriangle = base.shape_msgs__msg__MeshTriangle
shape_msgs__msg__Plane = base.shape_msgs__msg__Plane
shape_msgs__msg__SolidPrimitive = base.shape_msgs__msg__SolidPrimitive
statistics_msgs__msg__MetricsMessage = base.statistics_msgs__msg__MetricsMessage
statistics_msgs__msg__StatisticDataPoint = base.statistics_msgs__msg__StatisticDataPoint
statistics_msgs__msg__StatisticDataType = base.statistics_msgs__msg__StatisticDataType
std_msgs__msg__Bool = base.std_msgs__msg__Bool
std_msgs__msg__Byte = base.std_msgs__msg__Byte
std_msgs__msg__ByteMultiArray = base.std_msgs__msg__ByteMultiArray
std_msgs__msg__Char = base.std_msgs__msg__Char
std_msgs__msg__ColorRGBA = base.std_msgs__msg__ColorRGBA
std_msgs__msg__Empty = base.std_msgs__msg__Empty
std_msgs__msg__Float32 = base.std_msgs__msg__Float32
std_msgs__msg__Float32MultiArray = base.std_msgs__msg__Float32MultiArray
std_msgs__msg__Float64 = base.std_msgs__msg__Float64
std_msgs__msg__Float64MultiArray = base.std_msgs__msg__Float64MultiArray
std_msgs__msg__Header = base.std_msgs__msg__Header
std_msgs__msg__Int16 = base.std_msgs__msg__Int16
std_msgs__msg__Int16MultiArray = base.std_msgs__msg__Int16MultiArray
std_msgs__msg__Int32 = base.std_msgs__msg__Int32
std_msgs__msg__Int32MultiArray = base.std_msgs__msg__Int32MultiArray
std_msgs__msg__Int64 = base.std_msgs__msg__Int64
std_msgs__msg__Int64MultiArray = base.std_msgs__msg__Int64MultiArray
std_msgs__msg__Int8 = base.std_msgs__msg__Int8
std_msgs__msg__Int8MultiArray = base.std_msgs__msg__Int8MultiArray
std_msgs__msg__MultiArrayDimension = base.std_msgs__msg__MultiArrayDimension
std_msgs__msg__MultiArrayLayout = base.std_msgs__msg__MultiArrayLayout
std_msgs__msg__String = base.std_msgs__msg__String
std_msgs__msg__UInt16 = base.std_msgs__msg__UInt16
std_msgs__msg__UInt16MultiArray = base.std_msgs__msg__UInt16MultiArray
std_msgs__msg__UInt32 = base.std_msgs__msg__UInt32
std_msgs__msg__UInt32MultiArray = base.std_msgs__msg__UInt32MultiArray
std_msgs__msg__UInt64 = base.std_msgs__msg__UInt64
std_msgs__msg__UInt64MultiArray = base.std_msgs__msg__UInt64MultiArray
std_msgs__msg__UInt8 = base.std_msgs__msg__UInt8
std_msgs__msg__UInt8MultiArray = base.std_msgs__msg__UInt8MultiArray
stereo_msgs__msg__DisparityImage = base.stereo_msgs__msg__DisparityImage
tf2_msgs__msg__TF2Error = base.tf2_msgs__msg__TF2Error
tf2_msgs__msg__TFMessage = base.tf2_msgs__msg__TFMessage
trajectory_msgs__msg__JointTrajectory = base.trajectory_msgs__msg__JointTrajectory
trajectory_msgs__msg__JointTrajectoryPoint = base.trajectory_msgs__msg__JointTrajectoryPoint
trajectory_msgs__msg__MultiDOFJointTrajectory = base.trajectory_msgs__msg__MultiDOFJointTrajectory
trajectory_msgs__msg__MultiDOFJointTrajectoryPoint = (
    base.trajectory_msgs__msg__MultiDOFJointTrajectoryPoint
)
unique_identifier_msgs__msg__UUID = base.unique_identifier_msgs__msg__UUID
visualization_msgs__msg__ImageMarker = base.visualization_msgs__msg__ImageMarker
visualization_msgs__msg__InteractiveMarker = base.visualization_msgs__msg__InteractiveMarker
visualization_msgs__msg__InteractiveMarkerControl = (
    base.visualization_msgs__msg__InteractiveMarkerControl
)
visualization_msgs__msg__InteractiveMarkerFeedback = (
    base.visualization_msgs__msg__InteractiveMarkerFeedback
)
visualization_msgs__msg__InteractiveMarkerInit = base.visualization_msgs__msg__InteractiveMarkerInit
visualization_msgs__msg__InteractiveMarkerPose = base.visualization_msgs__msg__InteractiveMarkerPose
visualization_msgs__msg__InteractiveMarkerUpdate = (
    base.visualization_msgs__msg__InteractiveMarkerUpdate
)
visualization_msgs__msg__Marker = base.visualization_msgs__msg__Marker
visualization_msgs__msg__MarkerArray = base.visualization_msgs__msg__MarkerArray
visualization_msgs__msg__MenuEntry = base.visualization_msgs__msg__MenuEntry
visualization_msgs__msg__MeshFile = base.visualization_msgs__msg__MeshFile
visualization_msgs__msg__UVCoordinate = base.visualization_msgs__msg__UVCoordinate


@dataclass
class rcl_interfaces__msg__LoggerLevel:
    """Class for rcl_interfaces/msg/LoggerLevel."""

    name: str
    level: int
    LOG_LEVEL_UNKNOWN: ClassVar[int] = 0
    LOG_LEVEL_DEBUG: ClassVar[int] = 10
    LOG_LEVEL_INFO: ClassVar[int] = 20
    LOG_LEVEL_WARN: ClassVar[int] = 30
    LOG_LEVEL_ERROR: ClassVar[int] = 40
    LOG_LEVEL_FATAL: ClassVar[int] = 50
    __msgtype__: ClassVar[str] = 'rcl_interfaces/msg/LoggerLevel'


@dataclass
class rcl_interfaces__msg__SetLoggerLevelsResult:
    """Class for rcl_interfaces/msg/SetLoggerLevelsResult."""

    successful: bool
    reason: str
    __msgtype__: ClassVar[str] = 'rcl_interfaces/msg/SetLoggerLevelsResult'


@dataclass
class rmw_dds_common__msg__Gid:
    """Class for rmw_dds_common/msg/Gid."""

    data: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
    __msgtype__: ClassVar[str] = 'rmw_dds_common/msg/Gid'


@dataclass
class sensor_msgs__msg__BatteryState:
    """Class for sensor_msgs/msg/BatteryState."""

    header: std_msgs__msg__Header
    voltage: float
    temperature: float
    current: float
    charge: float
    capacity: float
    design_capacity: float
    percentage: float
    power_supply_status: int
    power_supply_health: int
    power_supply_technology: int
    present: bool
    cell_voltage: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
    cell_temperature: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
    location: str
    serial_number: str
    POWER_SUPPLY_STATUS_UNKNOWN: ClassVar[int] = 0
    POWER_SUPPLY_STATUS_CHARGING: ClassVar[int] = 1
    POWER_SUPPLY_STATUS_DISCHARGING: ClassVar[int] = 2
    POWER_SUPPLY_STATUS_NOT_CHARGING: ClassVar[int] = 3
    POWER_SUPPLY_STATUS_FULL: ClassVar[int] = 4
    POWER_SUPPLY_HEALTH_UNKNOWN: ClassVar[int] = 0
    POWER_SUPPLY_HEALTH_GOOD: ClassVar[int] = 1
    POWER_SUPPLY_HEALTH_OVERHEAT: ClassVar[int] = 2
    POWER_SUPPLY_HEALTH_DEAD: ClassVar[int] = 3
    POWER_SUPPLY_HEALTH_OVERVOLTAGE: ClassVar[int] = 4
    POWER_SUPPLY_HEALTH_UNSPEC_FAILURE: ClassVar[int] = 5
    POWER_SUPPLY_HEALTH_COLD: ClassVar[int] = 6
    POWER_SUPPLY_HEALTH_WATCHDOG_TIMER_EXPIRE: ClassVar[int] = 7
    POWER_SUPPLY_HEALTH_SAFETY_TIMER_EXPIRE: ClassVar[int] = 8
    POWER_SUPPLY_TECHNOLOGY_UNKNOWN: ClassVar[int] = 0
    POWER_SUPPLY_TECHNOLOGY_NIMH: ClassVar[int] = 1
    POWER_SUPPLY_TECHNOLOGY_LION: ClassVar[int] = 2
    POWER_SUPPLY_TECHNOLOGY_LIPO: ClassVar[int] = 3
    POWER_SUPPLY_TECHNOLOGY_LIFE: ClassVar[int] = 4
    POWER_SUPPLY_TECHNOLOGY_NICD: ClassVar[int] = 5
    POWER_SUPPLY_TECHNOLOGY_LIMN: ClassVar[int] = 6
    POWER_SUPPLY_TECHNOLOGY_TERNARY: ClassVar[int] = 7
    POWER_SUPPLY_TECHNOLOGY_VRLA: ClassVar[int] = 8
    __msgtype__: ClassVar[str] = 'sensor_msgs/msg/BatteryState'


@dataclass
class sensor_msgs__msg__Range:
    """Class for sensor_msgs/msg/Range."""

    header: std_msgs__msg__Header
    radiation_type: int
    field_of_view: float
    min_range: float
    max_range: float
    range: float
    variance: float
    ULTRASOUND: ClassVar[int] = 0
    INFRARED: ClassVar[int] = 1
    __msgtype__: ClassVar[str] = 'sensor_msgs/msg/Range'


@dataclass
class service_msgs__msg__ServiceEventInfo:
    """Class for service_msgs/msg/ServiceEventInfo."""

    event_type: int
    stamp: builtin_interfaces__msg__Time
    client_gid: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
    sequence_number: int
    REQUEST_SENT: ClassVar[int] = 0
    REQUEST_RECEIVED: ClassVar[int] = 1
    RESPONSE_SENT: ClassVar[int] = 2
    RESPONSE_RECEIVED: ClassVar[int] = 3
    __msgtype__: ClassVar[str] = 'service_msgs/msg/ServiceEventInfo'


@dataclass
class type_description_interfaces__msg__Field:
    """Class for type_description_interfaces/msg/Field."""

    name: str
    type: type_description_interfaces__msg__FieldType
    default_value: str
    __msgtype__: ClassVar[str] = 'type_description_interfaces/msg/Field'


@dataclass
class type_description_interfaces__msg__FieldType:
    """Class for type_description_interfaces/msg/FieldType."""

    type_id: int
    capacity: int
    string_capacity: int
    nested_type_name: str
    FIELD_TYPE_NOT_SET: ClassVar[int] = 0
    FIELD_TYPE_NESTED_TYPE: ClassVar[int] = 1
    FIELD_TYPE_INT8: ClassVar[int] = 2
    FIELD_TYPE_UINT8: ClassVar[int] = 3
    FIELD_TYPE_INT16: ClassVar[int] = 4
    FIELD_TYPE_UINT16: ClassVar[int] = 5
    FIELD_TYPE_INT32: ClassVar[int] = 6
    FIELD_TYPE_UINT32: ClassVar[int] = 7
    FIELD_TYPE_INT64: ClassVar[int] = 8
    FIELD_TYPE_UINT64: ClassVar[int] = 9
    FIELD_TYPE_FLOAT: ClassVar[int] = 10
    FIELD_TYPE_DOUBLE: ClassVar[int] = 11
    FIELD_TYPE_LONG_DOUBLE: ClassVar[int] = 12
    FIELD_TYPE_CHAR: ClassVar[int] = 13
    FIELD_TYPE_WCHAR: ClassVar[int] = 14
    FIELD_TYPE_BOOLEAN: ClassVar[int] = 15
    FIELD_TYPE_BYTE: ClassVar[int] = 16
    FIELD_TYPE_STRING: ClassVar[int] = 17
    FIELD_TYPE_WSTRING: ClassVar[int] = 18
    FIELD_TYPE_FIXED_STRING: ClassVar[int] = 19
    FIELD_TYPE_FIXED_WSTRING: ClassVar[int] = 20
    FIELD_TYPE_BOUNDED_STRING: ClassVar[int] = 21
    FIELD_TYPE_BOUNDED_WSTRING: ClassVar[int] = 22
    FIELD_TYPE_NESTED_TYPE_ARRAY: ClassVar[int] = 49
    FIELD_TYPE_INT8_ARRAY: ClassVar[int] = 50
    FIELD_TYPE_UINT8_ARRAY: ClassVar[int] = 51
    FIELD_TYPE_INT16_ARRAY: ClassVar[int] = 52
    FIELD_TYPE_UINT16_ARRAY: ClassVar[int] = 53
    FIELD_TYPE_INT32_ARRAY: ClassVar[int] = 54
    FIELD_TYPE_UINT32_ARRAY: ClassVar[int] = 55
    FIELD_TYPE_INT64_ARRAY: ClassVar[int] = 56
    FIELD_TYPE_UINT64_ARRAY: ClassVar[int] = 57
    FIELD_TYPE_FLOAT_ARRAY: ClassVar[int] = 58
    FIELD_TYPE_DOUBLE_ARRAY: ClassVar[int] = 59
    FIELD_TYPE_LONG_DOUBLE_ARRAY: ClassVar[int] = 60
    FIELD_TYPE_CHAR_ARRAY: ClassVar[int] = 61
    FIELD_TYPE_WCHAR_ARRAY: ClassVar[int] = 62
    FIELD_TYPE_BOOLEAN_ARRAY: ClassVar[int] = 63
    FIELD_TYPE_BYTE_ARRAY: ClassVar[int] = 64
    FIELD_TYPE_STRING_ARRAY: ClassVar[int] = 65
    FIELD_TYPE_WSTRING_ARRAY: ClassVar[int] = 66
    FIELD_TYPE_FIXED_STRING_ARRAY: ClassVar[int] = 67
    FIELD_TYPE_FIXED_WSTRING_ARRAY: ClassVar[int] = 68
    FIELD_TYPE_BOUNDED_STRING_ARRAY: ClassVar[int] = 69
    FIELD_TYPE_BOUNDED_WSTRING_ARRAY: ClassVar[int] = 70
    FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE: ClassVar[int] = 97
    FIELD_TYPE_INT8_BOUNDED_SEQUENCE: ClassVar[int] = 98
    FIELD_TYPE_UINT8_BOUNDED_SEQUENCE: ClassVar[int] = 99
    FIELD_TYPE_INT16_BOUNDED_SEQUENCE: ClassVar[int] = 100
    FIELD_TYPE_UINT16_BOUNDED_SEQUENCE: ClassVar[int] = 101
    FIELD_TYPE_INT32_BOUNDED_SEQUENCE: ClassVar[int] = 102
    FIELD_TYPE_UINT32_BOUNDED_SEQUENCE: ClassVar[int] = 103
    FIELD_TYPE_INT64_BOUNDED_SEQUENCE: ClassVar[int] = 104
    FIELD_TYPE_UINT64_BOUNDED_SEQUENCE: ClassVar[int] = 105
    FIELD_TYPE_FLOAT_BOUNDED_SEQUENCE: ClassVar[int] = 106
    FIELD_TYPE_DOUBLE_BOUNDED_SEQUENCE: ClassVar[int] = 107
    FIELD_TYPE_LONG_DOUBLE_BOUNDED_SEQUENCE: ClassVar[int] = 108
    FIELD_TYPE_CHAR_BOUNDED_SEQUENCE: ClassVar[int] = 109
    FIELD_TYPE_WCHAR_BOUNDED_SEQUENCE: ClassVar[int] = 110
    FIELD_TYPE_BOOLEAN_BOUNDED_SEQUENCE: ClassVar[int] = 111
    FIELD_TYPE_BYTE_BOUNDED_SEQUENCE: ClassVar[int] = 112
    FIELD_TYPE_STRING_BOUNDED_SEQUENCE: ClassVar[int] = 113
    FIELD_TYPE_WSTRING_BOUNDED_SEQUENCE: ClassVar[int] = 114
    FIELD_TYPE_FIXED_STRING_BOUNDED_SEQUENCE: ClassVar[int] = 115
    FIELD_TYPE_FIXED_WSTRING_BOUNDED_SEQUENCE: ClassVar[int] = 116
    FIELD_TYPE_BOUNDED_STRING_BOUNDED_SEQUENCE: ClassVar[int] = 117
    FIELD_TYPE_BOUNDED_WSTRING_BOUNDED_SEQUENCE: ClassVar[int] = 118
    FIELD_TYPE_NESTED_TYPE_UNBOUNDED_SEQUENCE: ClassVar[int] = 145
    FIELD_TYPE_INT8_UNBOUNDED_SEQUENCE: ClassVar[int] = 146
    FIELD_TYPE_UINT8_UNBOUNDED_SEQUENCE: ClassVar[int] = 147
    FIELD_TYPE_INT16_UNBOUNDED_SEQUENCE: ClassVar[int] = 148
    FIELD_TYPE_UINT16_UNBOUNDED_SEQUENCE: ClassVar[int] = 149
    FIELD_TYPE_INT32_UNBOUNDED_SEQUENCE: ClassVar[int] = 150
    FIELD_TYPE_UINT32_UNBOUNDED_SEQUENCE: ClassVar[int] = 151
    FIELD_TYPE_INT64_UNBOUNDED_SEQUENCE: ClassVar[int] = 152
    FIELD_TYPE_UINT64_UNBOUNDED_SEQUENCE: ClassVar[int] = 153
    FIELD_TYPE_FLOAT_UNBOUNDED_SEQUENCE: ClassVar[int] = 154
    FIELD_TYPE_DOUBLE_UNBOUNDED_SEQUENCE: ClassVar[int] = 155
    FIELD_TYPE_LONG_DOUBLE_UNBOUNDED_SEQUENCE: ClassVar[int] = 156
    FIELD_TYPE_CHAR_UNBOUNDED_SEQUENCE: ClassVar[int] = 157
    FIELD_TYPE_WCHAR_UNBOUNDED_SEQUENCE: ClassVar[int] = 158
    FIELD_TYPE_BOOLEAN_UNBOUNDED_SEQUENCE: ClassVar[int] = 159
    FIELD_TYPE_BYTE_UNBOUNDED_SEQUENCE: ClassVar[int] = 160
    FIELD_TYPE_STRING_UNBOUNDED_SEQUENCE: ClassVar[int] = 161
    FIELD_TYPE_WSTRING_UNBOUNDED_SEQUENCE: ClassVar[int] = 162
    FIELD_TYPE_FIXED_STRING_UNBOUNDED_SEQUENCE: ClassVar[int] = 163
    FIELD_TYPE_FIXED_WSTRING_UNBOUNDED_SEQUENCE: ClassVar[int] = 164
    FIELD_TYPE_BOUNDED_STRING_UNBOUNDED_SEQUENCE: ClassVar[int] = 165
    FIELD_TYPE_BOUNDED_WSTRING_UNBOUNDED_SEQUENCE: ClassVar[int] = 166
    __msgtype__: ClassVar[str] = 'type_description_interfaces/msg/FieldType'


@dataclass
class type_description_interfaces__msg__IndividualTypeDescription:
    """Class for type_description_interfaces/msg/IndividualTypeDescription."""

    type_name: str
    fields: list[type_description_interfaces__msg__Field]
    __msgtype__: ClassVar[str] = 'type_description_interfaces/msg/IndividualTypeDescription'


@dataclass
class type_description_interfaces__msg__KeyValue:
    """Class for type_description_interfaces/msg/KeyValue."""

    key: str
    value: str
    __msgtype__: ClassVar[str] = 'type_description_interfaces/msg/KeyValue'


@dataclass
class type_description_interfaces__msg__TypeDescription:
    """Class for type_description_interfaces/msg/TypeDescription."""

    type_description: type_description_interfaces__msg__IndividualTypeDescription
    referenced_type_descriptions: list[type_description_interfaces__msg__IndividualTypeDescription]
    __msgtype__: ClassVar[str] = 'type_description_interfaces/msg/TypeDescription'


@dataclass
class type_description_interfaces__msg__TypeSource:
    """Class for type_description_interfaces/msg/TypeSource."""

    type_name: str
    encoding: str
    raw_file_contents: str
    __msgtype__: ClassVar[str] = 'type_description_interfaces/msg/TypeSource'


FIELDDEFS: Typesdict = {
    'action_msgs/msg/GoalInfo': base.FIELDDEFS['action_msgs/msg/GoalInfo'],
    'action_msgs/msg/GoalStatus': base.FIELDDEFS['action_msgs/msg/GoalStatus'],
    'action_msgs/msg/GoalStatusArray': base.FIELDDEFS['action_msgs/msg/GoalStatusArray'],
    'actionlib_msgs/msg/GoalID': base.FIELDDEFS['actionlib_msgs/msg/GoalID'],
    'actionlib_msgs/msg/GoalStatus': base.FIELDDEFS['actionlib_msgs/msg/GoalStatus'],
    'actionlib_msgs/msg/GoalStatusArray': base.FIELDDEFS['actionlib_msgs/msg/GoalStatusArray'],
    'builtin_interfaces/msg/Duration': base.FIELDDEFS['builtin_interfaces/msg/Duration'],
    'builtin_interfaces/msg/Time': base.FIELDDEFS['builtin_interfaces/msg/Time'],
    'diagnostic_msgs/msg/DiagnosticArray': base.FIELDDEFS['diagnostic_msgs/msg/DiagnosticArray'],
    'diagnostic_msgs/msg/DiagnosticStatus': base.FIELDDEFS['diagnostic_msgs/msg/DiagnosticStatus'],
    'diagnostic_msgs/msg/KeyValue': base.FIELDDEFS['diagnostic_msgs/msg/KeyValue'],
    'geometry_msgs/msg/Accel': base.FIELDDEFS['geometry_msgs/msg/Accel'],
    'geometry_msgs/msg/AccelStamped': base.FIELDDEFS['geometry_msgs/msg/AccelStamped'],
    'geometry_msgs/msg/AccelWithCovariance': base.FIELDDEFS[
        'geometry_msgs/msg/AccelWithCovariance'
    ],
    'geometry_msgs/msg/AccelWithCovarianceStamped': base.FIELDDEFS[
        'geometry_msgs/msg/AccelWithCovarianceStamped'
    ],
    'geometry_msgs/msg/Inertia': base.FIELDDEFS['geometry_msgs/msg/Inertia'],
    'geometry_msgs/msg/InertiaStamped': base.FIELDDEFS['geometry_msgs/msg/InertiaStamped'],
    'geometry_msgs/msg/Point': base.FIELDDEFS['geometry_msgs/msg/Point'],
    'geometry_msgs/msg/Point32': base.FIELDDEFS['geometry_msgs/msg/Point32'],
    'geometry_msgs/msg/PointStamped': base.FIELDDEFS['geometry_msgs/msg/PointStamped'],
    'geometry_msgs/msg/Polygon': base.FIELDDEFS['geometry_msgs/msg/Polygon'],
    'geometry_msgs/msg/PolygonStamped': base.FIELDDEFS['geometry_msgs/msg/PolygonStamped'],
    'geometry_msgs/msg/Pose': base.FIELDDEFS['geometry_msgs/msg/Pose'],
    'geometry_msgs/msg/Pose2D': base.FIELDDEFS['geometry_msgs/msg/Pose2D'],
    'geometry_msgs/msg/PoseArray': base.FIELDDEFS['geometry_msgs/msg/PoseArray'],
    'geometry_msgs/msg/PoseStamped': base.FIELDDEFS['geometry_msgs/msg/PoseStamped'],
    'geometry_msgs/msg/PoseWithCovariance': base.FIELDDEFS['geometry_msgs/msg/PoseWithCovariance'],
    'geometry_msgs/msg/PoseWithCovarianceStamped': base.FIELDDEFS[
        'geometry_msgs/msg/PoseWithCovarianceStamped'
    ],
    'geometry_msgs/msg/Quaternion': base.FIELDDEFS['geometry_msgs/msg/Quaternion'],
    'geometry_msgs/msg/QuaternionStamped': base.FIELDDEFS['geometry_msgs/msg/QuaternionStamped'],
    'geometry_msgs/msg/Transform': base.FIELDDEFS['geometry_msgs/msg/Transform'],
    'geometry_msgs/msg/TransformStamped': base.FIELDDEFS['geometry_msgs/msg/TransformStamped'],
    'geometry_msgs/msg/Twist': base.FIELDDEFS['geometry_msgs/msg/Twist'],
    'geometry_msgs/msg/TwistStamped': base.FIELDDEFS['geometry_msgs/msg/TwistStamped'],
    'geometry_msgs/msg/TwistWithCovariance': base.FIELDDEFS[
        'geometry_msgs/msg/TwistWithCovariance'
    ],
    'geometry_msgs/msg/TwistWithCovarianceStamped': base.FIELDDEFS[
        'geometry_msgs/msg/TwistWithCovarianceStamped'
    ],
    'geometry_msgs/msg/Vector3': base.FIELDDEFS['geometry_msgs/msg/Vector3'],
    'geometry_msgs/msg/Vector3Stamped': base.FIELDDEFS['geometry_msgs/msg/Vector3Stamped'],
    'geometry_msgs/msg/Wrench': base.FIELDDEFS['geometry_msgs/msg/Wrench'],
    'geometry_msgs/msg/WrenchStamped': base.FIELDDEFS['geometry_msgs/msg/WrenchStamped'],
    'lifecycle_msgs/msg/State': base.FIELDDEFS['lifecycle_msgs/msg/State'],
    'lifecycle_msgs/msg/Transition': base.FIELDDEFS['lifecycle_msgs/msg/Transition'],
    'lifecycle_msgs/msg/TransitionDescription': base.FIELDDEFS[
        'lifecycle_msgs/msg/TransitionDescription'
    ],
    'lifecycle_msgs/msg/TransitionEvent': base.FIELDDEFS['lifecycle_msgs/msg/TransitionEvent'],
    'nav_msgs/msg/GridCells': base.FIELDDEFS['nav_msgs/msg/GridCells'],
    'nav_msgs/msg/MapMetaData': base.FIELDDEFS['nav_msgs/msg/MapMetaData'],
    'nav_msgs/msg/OccupancyGrid': base.FIELDDEFS['nav_msgs/msg/OccupancyGrid'],
    'nav_msgs/msg/Odometry': base.FIELDDEFS['nav_msgs/msg/Odometry'],
    'nav_msgs/msg/Path': base.FIELDDEFS['nav_msgs/msg/Path'],
    'rcl_interfaces/msg/FloatingPointRange': base.FIELDDEFS[
        'rcl_interfaces/msg/FloatingPointRange'
    ],
    'rcl_interfaces/msg/IntegerRange': base.FIELDDEFS['rcl_interfaces/msg/IntegerRange'],
    'rcl_interfaces/msg/ListParametersResult': base.FIELDDEFS[
        'rcl_interfaces/msg/ListParametersResult'
    ],
    'rcl_interfaces/msg/Log': base.FIELDDEFS['rcl_interfaces/msg/Log'],
    'rcl_interfaces/msg/Parameter': base.FIELDDEFS['rcl_interfaces/msg/Parameter'],
    'rcl_interfaces/msg/ParameterDescriptor': base.FIELDDEFS[
        'rcl_interfaces/msg/ParameterDescriptor'
    ],
    'rcl_interfaces/msg/ParameterEvent': base.FIELDDEFS['rcl_interfaces/msg/ParameterEvent'],
    'rcl_interfaces/msg/ParameterEventDescriptors': base.FIELDDEFS[
        'rcl_interfaces/msg/ParameterEventDescriptors'
    ],
    'rcl_interfaces/msg/ParameterType': base.FIELDDEFS['rcl_interfaces/msg/ParameterType'],
    'rcl_interfaces/msg/ParameterValue': base.FIELDDEFS['rcl_interfaces/msg/ParameterValue'],
    'rcl_interfaces/msg/SetParametersResult': base.FIELDDEFS[
        'rcl_interfaces/msg/SetParametersResult'
    ],
    'rmw_dds_common/msg/NodeEntitiesInfo': base.FIELDDEFS['rmw_dds_common/msg/NodeEntitiesInfo'],
    'rmw_dds_common/msg/ParticipantEntitiesInfo': base.FIELDDEFS[
        'rmw_dds_common/msg/ParticipantEntitiesInfo'
    ],
    'rosbag2_interfaces/msg/ReadSplitEvent': base.FIELDDEFS[
        'rosbag2_interfaces/msg/ReadSplitEvent'
    ],
    'rosbag2_interfaces/msg/WriteSplitEvent': base.FIELDDEFS[
        'rosbag2_interfaces/msg/WriteSplitEvent'
    ],
    'rosgraph_msgs/msg/Clock': base.FIELDDEFS['rosgraph_msgs/msg/Clock'],
    'sensor_msgs/msg/CameraInfo': base.FIELDDEFS['sensor_msgs/msg/CameraInfo'],
    'sensor_msgs/msg/ChannelFloat32': base.FIELDDEFS['sensor_msgs/msg/ChannelFloat32'],
    'sensor_msgs/msg/CompressedImage': base.FIELDDEFS['sensor_msgs/msg/CompressedImage'],
    'sensor_msgs/msg/FluidPressure': base.FIELDDEFS['sensor_msgs/msg/FluidPressure'],
    'sensor_msgs/msg/Illuminance': base.FIELDDEFS['sensor_msgs/msg/Illuminance'],
    'sensor_msgs/msg/Image': base.FIELDDEFS['sensor_msgs/msg/Image'],
    'sensor_msgs/msg/Imu': base.FIELDDEFS['sensor_msgs/msg/Imu'],
    'sensor_msgs/msg/JointState': base.FIELDDEFS['sensor_msgs/msg/JointState'],
    'sensor_msgs/msg/Joy': base.FIELDDEFS['sensor_msgs/msg/Joy'],
    'sensor_msgs/msg/JoyFeedback': base.FIELDDEFS['sensor_msgs/msg/JoyFeedback'],
    'sensor_msgs/msg/JoyFeedbackArray': base.FIELDDEFS['sensor_msgs/msg/JoyFeedbackArray'],
    'sensor_msgs/msg/LaserEcho': base.FIELDDEFS['sensor_msgs/msg/LaserEcho'],
    'sensor_msgs/msg/LaserScan': base.FIELDDEFS['sensor_msgs/msg/LaserScan'],
    'sensor_msgs/msg/MagneticField': base.FIELDDEFS['sensor_msgs/msg/MagneticField'],
    'sensor_msgs/msg/MultiDOFJointState': base.FIELDDEFS['sensor_msgs/msg/MultiDOFJointState'],
    'sensor_msgs/msg/MultiEchoLaserScan': base.FIELDDEFS['sensor_msgs/msg/MultiEchoLaserScan'],
    'sensor_msgs/msg/NavSatFix': base.FIELDDEFS['sensor_msgs/msg/NavSatFix'],
    'sensor_msgs/msg/NavSatStatus': base.FIELDDEFS['sensor_msgs/msg/NavSatStatus'],
    'sensor_msgs/msg/PointCloud': base.FIELDDEFS['sensor_msgs/msg/PointCloud'],
    'sensor_msgs/msg/PointCloud2': base.FIELDDEFS['sensor_msgs/msg/PointCloud2'],
    'sensor_msgs/msg/PointField': base.FIELDDEFS['sensor_msgs/msg/PointField'],
    'sensor_msgs/msg/RegionOfInterest': base.FIELDDEFS['sensor_msgs/msg/RegionOfInterest'],
    'sensor_msgs/msg/RelativeHumidity': base.FIELDDEFS['sensor_msgs/msg/RelativeHumidity'],
    'sensor_msgs/msg/Temperature': base.FIELDDEFS['sensor_msgs/msg/Temperature'],
    'sensor_msgs/msg/TimeReference': base.FIELDDEFS['sensor_msgs/msg/TimeReference'],
    'shape_msgs/msg/Mesh': base.FIELDDEFS['shape_msgs/msg/Mesh'],
    'shape_msgs/msg/MeshTriangle': base.FIELDDEFS['shape_msgs/msg/MeshTriangle'],
    'shape_msgs/msg/Plane': base.FIELDDEFS['shape_msgs/msg/Plane'],
    'shape_msgs/msg/SolidPrimitive': base.FIELDDEFS['shape_msgs/msg/SolidPrimitive'],
    'statistics_msgs/msg/MetricsMessage': base.FIELDDEFS['statistics_msgs/msg/MetricsMessage'],
    'statistics_msgs/msg/StatisticDataPoint': base.FIELDDEFS[
        'statistics_msgs/msg/StatisticDataPoint'
    ],
    'statistics_msgs/msg/StatisticDataType': base.FIELDDEFS[
        'statistics_msgs/msg/StatisticDataType'
    ],
    'std_msgs/msg/Bool': base.FIELDDEFS['std_msgs/msg/Bool'],
    'std_msgs/msg/Byte': base.FIELDDEFS['std_msgs/msg/Byte'],
    'std_msgs/msg/ByteMultiArray': base.FIELDDEFS['std_msgs/msg/ByteMultiArray'],
    'std_msgs/msg/Char': base.FIELDDEFS['std_msgs/msg/Char'],
    'std_msgs/msg/ColorRGBA': base.FIELDDEFS['std_msgs/msg/ColorRGBA'],
    'std_msgs/msg/Empty': base.FIELDDEFS['std_msgs/msg/Empty'],
    'std_msgs/msg/Float32': base.FIELDDEFS['std_msgs/msg/Float32'],
    'std_msgs/msg/Float32MultiArray': base.FIELDDEFS['std_msgs/msg/Float32MultiArray'],
    'std_msgs/msg/Float64': base.FIELDDEFS['std_msgs/msg/Float64'],
    'std_msgs/msg/Float64MultiArray': base.FIELDDEFS['std_msgs/msg/Float64MultiArray'],
    'std_msgs/msg/Header': base.FIELDDEFS['std_msgs/msg/Header'],
    'std_msgs/msg/Int16': base.FIELDDEFS['std_msgs/msg/Int16'],
    'std_msgs/msg/Int16MultiArray': base.FIELDDEFS['std_msgs/msg/Int16MultiArray'],
    'std_msgs/msg/Int32': base.FIELDDEFS['std_msgs/msg/Int32'],
    'std_msgs/msg/Int32MultiArray': base.FIELDDEFS['std_msgs/msg/Int32MultiArray'],
    'std_msgs/msg/Int64': base.FIELDDEFS['std_msgs/msg/Int64'],
    'std_msgs/msg/Int64MultiArray': base.FIELDDEFS['std_msgs/msg/Int64MultiArray'],
    'std_msgs/msg/Int8': base.FIELDDEFS['std_msgs/msg/Int8'],
    'std_msgs/msg/Int8MultiArray': base.FIELDDEFS['std_msgs/msg/Int8MultiArray'],
    'std_msgs/msg/MultiArrayDimension': base.FIELDDEFS['std_msgs/msg/MultiArrayDimension'],
    'std_msgs/msg/MultiArrayLayout': base.FIELDDEFS['std_msgs/msg/MultiArrayLayout'],
    'std_msgs/msg/String': base.FIELDDEFS['std_msgs/msg/String'],
    'std_msgs/msg/UInt16': base.FIELDDEFS['std_msgs/msg/UInt16'],
    'std_msgs/msg/UInt16MultiArray': base.FIELDDEFS['std_msgs/msg/UInt16MultiArray'],
    'std_msgs/msg/UInt32': base.FIELDDEFS['std_msgs/msg/UInt32'],
    'std_msgs/msg/UInt32MultiArray': base.FIELDDEFS['std_msgs/msg/UInt32MultiArray'],
    'std_msgs/msg/UInt64': base.FIELDDEFS['std_msgs/msg/UInt64'],
    'std_msgs/msg/UInt64MultiArray': base.FIELDDEFS['std_msgs/msg/UInt64MultiArray'],
    'std_msgs/msg/UInt8': base.FIELDDEFS['std_msgs/msg/UInt8'],
    'std_msgs/msg/UInt8MultiArray': base.FIELDDEFS['std_msgs/msg/UInt8MultiArray'],
    'stereo_msgs/msg/DisparityImage': base.FIELDDEFS['stereo_msgs/msg/DisparityImage'],
    'tf2_msgs/msg/TF2Error': base.FIELDDEFS['tf2_msgs/msg/TF2Error'],
    'tf2_msgs/msg/TFMessage': base.FIELDDEFS['tf2_msgs/msg/TFMessage'],
    'trajectory_msgs/msg/JointTrajectory': base.FIELDDEFS['trajectory_msgs/msg/JointTrajectory'],
    'trajectory_msgs/msg/JointTrajectoryPoint': base.FIELDDEFS[
        'trajectory_msgs/msg/JointTrajectoryPoint'
    ],
    'trajectory_msgs/msg/MultiDOFJointTrajectory': base.FIELDDEFS[
        'trajectory_msgs/msg/MultiDOFJointTrajectory'
    ],
    'trajectory_msgs/msg/MultiDOFJointTrajectoryPoint': base.FIELDDEFS[
        'trajectory_msgs/msg/MultiDOFJointTrajectoryPoint'
    ],
    'unique_identifier_msgs/msg/UUID': base.FIELDDEFS['unique_identifier_msgs/msg/UUID'],
    'visualization_msgs/msg/ImageMarker': base.FIELDDEFS['visualization_msgs/msg/ImageMarker'],
    'visualization_msgs/msg/InteractiveMarker': base.FIELDDEFS[
        'visualization_msgs/msg/InteractiveMarker'
    ],
    'visualization_msgs/msg/InteractiveMarkerControl': base.FIELDDEFS[
        'visualization_msgs/msg/InteractiveMarkerControl'
    ],
    'visualization_msgs/msg/InteractiveMarkerFeedback': base.FIELDDEFS[
        'visualization_msgs/msg/InteractiveMarkerFeedback'
    ],
    'visualization_msgs/msg/InteractiveMarkerInit': base.FIELDDEFS[
        'visualization_msgs/msg/InteractiveMarkerInit'
    ],
    'visualization_msgs/msg/InteractiveMarkerPose': base.FIELDDEFS[
        'visualization_msgs/msg/InteractiveMarkerPose'
    ],
    'visualization_msgs/msg/InteractiveMarkerUpdate': base.FIELDDEFS[
        'visualization_msgs/msg/InteractiveMarkerUpdate'
    ],
    'visualization_msgs/msg/Marker': base.FIELDDEFS['visualization_msgs/msg/Marker'],
    'visualization_msgs/msg/MarkerArray': base.FIELDDEFS['visualization_msgs/msg/MarkerArray'],
    'visualization_msgs/msg/MenuEntry': base.FIELDDEFS['visualization_msgs/msg/MenuEntry'],
    'visualization_msgs/msg/MeshFile': base.FIELDDEFS['visualization_msgs/msg/MeshFile'],
    'visualization_msgs/msg/UVCoordinate': base.FIELDDEFS['visualization_msgs/msg/UVCoordinate'],
    'rcl_interfaces/msg/LoggerLevel': (
        [
            ('LOG_LEVEL_UNKNOWN', 'uint8', 0),
            ('LOG_LEVEL_DEBUG', 'uint8', 10),
            ('LOG_LEVEL_INFO', 'uint8', 20),
            ('LOG_LEVEL_WARN', 'uint8', 30),
            ('LOG_LEVEL_ERROR', 'uint8', 40),
            ('LOG_LEVEL_FATAL', 'uint8', 50),
        ],
        [
            ('name', (T.BASE, ('string', 0))),
            ('level', (T.BASE, ('uint32', 0))),
        ],
    ),
    'rcl_interfaces/msg/SetLoggerLevelsResult': (
        [],
        [
            ('successful', (T.BASE, ('bool', 0))),
            ('reason', (T.BASE, ('string', 0))),
        ],
    ),
    'rmw_dds_common/msg/Gid': (
        [],
        [
            ('data', (T.ARRAY, ((T.BASE, ('char', 0)), 16))),
        ],
    ),
    'sensor_msgs/msg/BatteryState': (
        [
            ('POWER_SUPPLY_STATUS_UNKNOWN', 'uint8', 0),
            ('POWER_SUPPLY_STATUS_CHARGING', 'uint8', 1),
            ('POWER_SUPPLY_STATUS_DISCHARGING', 'uint8', 2),
            ('POWER_SUPPLY_STATUS_NOT_CHARGING', 'uint8', 3),
            ('POWER_SUPPLY_STATUS_FULL', 'uint8', 4),
            ('POWER_SUPPLY_HEALTH_UNKNOWN', 'uint8', 0),
            ('POWER_SUPPLY_HEALTH_GOOD', 'uint8', 1),
            ('POWER_SUPPLY_HEALTH_OVERHEAT', 'uint8', 2),
            ('POWER_SUPPLY_HEALTH_DEAD', 'uint8', 3),
            ('POWER_SUPPLY_HEALTH_OVERVOLTAGE', 'uint8', 4),
            ('POWER_SUPPLY_HEALTH_UNSPEC_FAILURE', 'uint8', 5),
            ('POWER_SUPPLY_HEALTH_COLD', 'uint8', 6),
            ('POWER_SUPPLY_HEALTH_WATCHDOG_TIMER_EXPIRE', 'uint8', 7),
            ('POWER_SUPPLY_HEALTH_SAFETY_TIMER_EXPIRE', 'uint8', 8),
            ('POWER_SUPPLY_TECHNOLOGY_UNKNOWN', 'uint8', 0),
            ('POWER_SUPPLY_TECHNOLOGY_NIMH', 'uint8', 1),
            ('POWER_SUPPLY_TECHNOLOGY_LION', 'uint8', 2),
            ('POWER_SUPPLY_TECHNOLOGY_LIPO', 'uint8', 3),
            ('POWER_SUPPLY_TECHNOLOGY_LIFE', 'uint8', 4),
            ('POWER_SUPPLY_TECHNOLOGY_NICD', 'uint8', 5),
            ('POWER_SUPPLY_TECHNOLOGY_LIMN', 'uint8', 6),
            ('POWER_SUPPLY_TECHNOLOGY_TERNARY', 'uint8', 7),
            ('POWER_SUPPLY_TECHNOLOGY_VRLA', 'uint8', 8),
        ],
        [
            ('header', (T.NAME, 'std_msgs/msg/Header')),
            ('voltage', (T.BASE, ('float32', 0))),
            ('temperature', (T.BASE, ('float32', 0))),
            ('current', (T.BASE, ('float32', 0))),
            ('charge', (T.BASE, ('float32', 0))),
            ('capacity', (T.BASE, ('float32', 0))),
            ('design_capacity', (T.BASE, ('float32', 0))),
            ('percentage', (T.BASE, ('float32', 0))),
            ('power_supply_status', (T.BASE, ('uint8', 0))),
            ('power_supply_health', (T.BASE, ('uint8', 0))),
            ('power_supply_technology', (T.BASE, ('uint8', 0))),
            ('present', (T.BASE, ('bool', 0))),
            ('cell_voltage', (T.SEQUENCE, ((T.BASE, ('float32', 0)), 0))),
            ('cell_temperature', (T.SEQUENCE, ((T.BASE, ('float32', 0)), 0))),
            ('location', (T.BASE, ('string', 0))),
            ('serial_number', (T.BASE, ('string', 0))),
        ],
    ),
    'sensor_msgs/msg/Range': (
        [
            ('ULTRASOUND', 'uint8', 0),
            ('INFRARED', 'uint8', 1),
        ],
        [
            ('header', (T.NAME, 'std_msgs/msg/Header')),
            ('radiation_type', (T.BASE, ('uint8', 0))),
            ('field_of_view', (T.BASE, ('float32', 0))),
            ('min_range', (T.BASE, ('float32', 0))),
            ('max_range', (T.BASE, ('float32', 0))),
            ('range', (T.BASE, ('float32', 0))),
            ('variance', (T.BASE, ('float32', 0))),
        ],
    ),
    'service_msgs/msg/ServiceEventInfo': (
        [
            ('REQUEST_SENT', 'uint8', 0),
            ('REQUEST_RECEIVED', 'uint8', 1),
            ('RESPONSE_SENT', 'uint8', 2),
            ('RESPONSE_RECEIVED', 'uint8', 3),
        ],
        [
            ('event_type', (T.BASE, ('uint8', 0))),
            ('stamp', (T.NAME, 'builtin_interfaces/msg/Time')),
            ('client_gid', (T.ARRAY, ((T.BASE, ('char', 0)), 16))),
            ('sequence_number', (T.BASE, ('int64', 0))),
        ],
    ),
    'type_description_interfaces/msg/Field': (
        [],
        [
            ('name', (T.BASE, ('string', 0))),
            ('type', (T.NAME, 'type_description_interfaces/msg/FieldType')),
            ('default_value', (T.BASE, ('string', 0))),
        ],
    ),
    'type_description_interfaces/msg/FieldType': (
        [
            ('FIELD_TYPE_NOT_SET', 'uint8', 0),
            ('FIELD_TYPE_NESTED_TYPE', 'uint8', 1),
            ('FIELD_TYPE_INT8', 'uint8', 2),
            ('FIELD_TYPE_UINT8', 'uint8', 3),
            ('FIELD_TYPE_INT16', 'uint8', 4),
            ('FIELD_TYPE_UINT16', 'uint8', 5),
            ('FIELD_TYPE_INT32', 'uint8', 6),
            ('FIELD_TYPE_UINT32', 'uint8', 7),
            ('FIELD_TYPE_INT64', 'uint8', 8),
            ('FIELD_TYPE_UINT64', 'uint8', 9),
            ('FIELD_TYPE_FLOAT', 'uint8', 10),
            ('FIELD_TYPE_DOUBLE', 'uint8', 11),
            ('FIELD_TYPE_LONG_DOUBLE', 'uint8', 12),
            ('FIELD_TYPE_CHAR', 'uint8', 13),
            ('FIELD_TYPE_WCHAR', 'uint8', 14),
            ('FIELD_TYPE_BOOLEAN', 'uint8', 15),
            ('FIELD_TYPE_BYTE', 'uint8', 16),
            ('FIELD_TYPE_STRING', 'uint8', 17),
            ('FIELD_TYPE_WSTRING', 'uint8', 18),
            ('FIELD_TYPE_FIXED_STRING', 'uint8', 19),
            ('FIELD_TYPE_FIXED_WSTRING', 'uint8', 20),
            ('FIELD_TYPE_BOUNDED_STRING', 'uint8', 21),
            ('FIELD_TYPE_BOUNDED_WSTRING', 'uint8', 22),
            ('FIELD_TYPE_NESTED_TYPE_ARRAY', 'uint8', 49),
            ('FIELD_TYPE_INT8_ARRAY', 'uint8', 50),
            ('FIELD_TYPE_UINT8_ARRAY', 'uint8', 51),
            ('FIELD_TYPE_INT16_ARRAY', 'uint8', 52),
            ('FIELD_TYPE_UINT16_ARRAY', 'uint8', 53),
            ('FIELD_TYPE_INT32_ARRAY', 'uint8', 54),
            ('FIELD_TYPE_UINT32_ARRAY', 'uint8', 55),
            ('FIELD_TYPE_INT64_ARRAY', 'uint8', 56),
            ('FIELD_TYPE_UINT64_ARRAY', 'uint8', 57),
            ('FIELD_TYPE_FLOAT_ARRAY', 'uint8', 58),
            ('FIELD_TYPE_DOUBLE_ARRAY', 'uint8', 59),
            ('FIELD_TYPE_LONG_DOUBLE_ARRAY', 'uint8', 60),
            ('FIELD_TYPE_CHAR_ARRAY', 'uint8', 61),
            ('FIELD_TYPE_WCHAR_ARRAY', 'uint8', 62),
            ('FIELD_TYPE_BOOLEAN_ARRAY', 'uint8', 63),
            ('FIELD_TYPE_BYTE_ARRAY', 'uint8', 64),
            ('FIELD_TYPE_STRING_ARRAY', 'uint8', 65),
            ('FIELD_TYPE_WSTRING_ARRAY', 'uint8', 66),
            ('FIELD_TYPE_FIXED_STRING_ARRAY', 'uint8', 67),
            ('FIELD_TYPE_FIXED_WSTRING_ARRAY', 'uint8', 68),
            ('FIELD_TYPE_BOUNDED_STRING_ARRAY', 'uint8', 69),
            ('FIELD_TYPE_BOUNDED_WSTRING_ARRAY', 'uint8', 70),
            ('FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE', 'uint8', 97),
            ('FIELD_TYPE_INT8_BOUNDED_SEQUENCE', 'uint8', 98),
            ('FIELD_TYPE_UINT8_BOUNDED_SEQUENCE', 'uint8', 99),
            ('FIELD_TYPE_INT16_BOUNDED_SEQUENCE', 'uint8', 100),
            ('FIELD_TYPE_UINT16_BOUNDED_SEQUENCE', 'uint8', 101),
            ('FIELD_TYPE_INT32_BOUNDED_SEQUENCE', 'uint8', 102),
            ('FIELD_TYPE_UINT32_BOUNDED_SEQUENCE', 'uint8', 103),
            ('FIELD_TYPE_INT64_BOUNDED_SEQUENCE', 'uint8', 104),
            ('FIELD_TYPE_UINT64_BOUNDED_SEQUENCE', 'uint8', 105),
            ('FIELD_TYPE_FLOAT_BOUNDED_SEQUENCE', 'uint8', 106),
            ('FIELD_TYPE_DOUBLE_BOUNDED_SEQUENCE', 'uint8', 107),
            ('FIELD_TYPE_LONG_DOUBLE_BOUNDED_SEQUENCE', 'uint8', 108),
            ('FIELD_TYPE_CHAR_BOUNDED_SEQUENCE', 'uint8', 109),
            ('FIELD_TYPE_WCHAR_BOUNDED_SEQUENCE', 'uint8', 110),
            ('FIELD_TYPE_BOOLEAN_BOUNDED_SEQUENCE', 'uint8', 111),
            ('FIELD_TYPE_BYTE_BOUNDED_SEQUENCE', 'uint8', 112),
            ('FIELD_TYPE_STRING_BOUNDED_SEQUENCE', 'uint8', 113),
            ('FIELD_TYPE_WSTRING_BOUNDED_SEQUENCE', 'uint8', 114),
            ('FIELD_TYPE_FIXED_STRING_BOUNDED_SEQUENCE', 'uint8', 115),
            ('FIELD_TYPE_FIXED_WSTRING_BOUNDED_SEQUENCE', 'uint8', 116),
            ('FIELD_TYPE_BOUNDED_STRING_BOUNDED_SEQUENCE', 'uint8', 117),
            ('FIELD_TYPE_BOUNDED_WSTRING_BOUNDED_SEQUENCE', 'uint8', 118),
            ('FIELD_TYPE_NESTED_TYPE_UNBOUNDED_SEQUENCE', 'uint8', 145),
            ('FIELD_TYPE_INT8_UNBOUNDED_SEQUENCE', 'uint8', 146),
            ('FIELD_TYPE_UINT8_UNBOUNDED_SEQUENCE', 'uint8', 147),
            ('FIELD_TYPE_INT16_UNBOUNDED_SEQUENCE', 'uint8', 148),
            ('FIELD_TYPE_UINT16_UNBOUNDED_SEQUENCE', 'uint8', 149),
            ('FIELD_TYPE_INT32_UNBOUNDED_SEQUENCE', 'uint8', 150),
            ('FIELD_TYPE_UINT32_UNBOUNDED_SEQUENCE', 'uint8', 151),
            ('FIELD_TYPE_INT64_UNBOUNDED_SEQUENCE', 'uint8', 152),
            ('FIELD_TYPE_UINT64_UNBOUNDED_SEQUENCE', 'uint8', 153),
            ('FIELD_TYPE_FLOAT_UNBOUNDED_SEQUENCE', 'uint8', 154),
            ('FIELD_TYPE_DOUBLE_UNBOUNDED_SEQUENCE', 'uint8', 155),
            ('FIELD_TYPE_LONG_DOUBLE_UNBOUNDED_SEQUENCE', 'uint8', 156),
            ('FIELD_TYPE_CHAR_UNBOUNDED_SEQUENCE', 'uint8', 157),
            ('FIELD_TYPE_WCHAR_UNBOUNDED_SEQUENCE', 'uint8', 158),
            ('FIELD_TYPE_BOOLEAN_UNBOUNDED_SEQUENCE', 'uint8', 159),
            ('FIELD_TYPE_BYTE_UNBOUNDED_SEQUENCE', 'uint8', 160),
            ('FIELD_TYPE_STRING_UNBOUNDED_SEQUENCE', 'uint8', 161),
            ('FIELD_TYPE_WSTRING_UNBOUNDED_SEQUENCE', 'uint8', 162),
            ('FIELD_TYPE_FIXED_STRING_UNBOUNDED_SEQUENCE', 'uint8', 163),
            ('FIELD_TYPE_FIXED_WSTRING_UNBOUNDED_SEQUENCE', 'uint8', 164),
            ('FIELD_TYPE_BOUNDED_STRING_UNBOUNDED_SEQUENCE', 'uint8', 165),
            ('FIELD_TYPE_BOUNDED_WSTRING_UNBOUNDED_SEQUENCE', 'uint8', 166),
        ],
        [
            ('type_id', (T.BASE, ('uint8', 0))),
            ('capacity', (T.BASE, ('uint64', 0))),
            ('string_capacity', (T.BASE, ('uint64', 0))),
            ('nested_type_name', (T.BASE, ('string', 255))),
        ],
    ),
    'type_description_interfaces/msg/IndividualTypeDescription': (
        [],
        [
            ('type_name', (T.BASE, ('string', 255))),
            ('fields', (T.SEQUENCE, ((T.NAME, 'type_description_interfaces/msg/Field'), 0))),
        ],
    ),
    'type_description_interfaces/msg/KeyValue': (
        [],
        [
            ('key', (T.BASE, ('string', 0))),
            ('value', (T.BASE, ('string', 0))),
        ],
    ),
    'type_description_interfaces/msg/TypeDescription': (
        [],
        [
            (
                'type_description',
                (T.NAME, 'type_description_interfaces/msg/IndividualTypeDescription'),
            ),
            (
                'referenced_type_descriptions',
                (
                    T.SEQUENCE,
                    ((T.NAME, 'type_description_interfaces/msg/IndividualTypeDescription'), 0),
                ),
            ),
        ],
    ),
    'type_description_interfaces/msg/TypeSource': (
        [],
        [
            ('type_name', (T.BASE, ('string', 0))),
            ('encoding', (T.BASE, ('string', 0))),
            ('raw_file_contents', (T.BASE, ('string', 0))),
        ],
    ),
}

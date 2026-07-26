#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Iterable
from typing import Iterator
from typing import Sequence


DEFAULT_PREVIEW_LIMIT = 5
DEFAULT_TOF_TOPIC = '/tof_sim/tof/points'
DEFAULT_ODOM_TOPIC = '/odom'
DEFAULT_OUTPUT_DIR = 'datasets'
DEFAULT_MAX_DISTANCE_M = 4.0
DEFAULT_MIN_DISTANCE_M = 0.05
DEFAULT_WIDTH = 8
DEFAULT_HEIGHT = 8


@dataclass(frozen=True)
class BagTopic:
    name: str
    message_type: str
    serialization_format: str


@dataclass(frozen=True)
class SerializedBagRecord:
    topic: str
    timestamp_ns: int
    data: object


@dataclass(frozen=True)
class BagRecord:
    topic: str
    timestamp_ns: int
    message_type: str
    message: object


@dataclass
class TopicSummary:
    topic: BagTopic
    count: int = 0
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None

    def add(self, timestamp_ns: int) -> None:
        self.count += 1
        if self.first_timestamp_ns is None or timestamp_ns < self.first_timestamp_ns:
            self.first_timestamp_ns = timestamp_ns
        if self.last_timestamp_ns is None or timestamp_ns > self.last_timestamp_ns:
            self.last_timestamp_ns = timestamp_ns


@dataclass(frozen=True)
class BagSummary:
    bag_uri: Path
    storage_id: str
    topics: list[TopicSummary]
    total_count: int
    first_timestamp_ns: int | None
    last_timestamp_ns: int | None


@dataclass(frozen=True)
class ToFFrame:
    timestamp_ns: int
    distance_m: object
    valid: object


@dataclass(frozen=True)
class PoseSample:
    timestamp_ns: int
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class DatasetExportResult:
    npz_path: Path
    metadata_path: Path
    frame_count: int
    sample_count: int
    skipped_tof_frames: int
    duplicate_tof_frames: int


class RosbagReadError(RuntimeError):
    pass


class RosbagReader:
    def __init__(self, bag_path: str | Path, storage_id: str | None = None):
        self.bag_uri = normalize_bag_uri(Path(bag_path))
        self.storage_id = storage_id or detect_storage_id(self.bag_uri)

    def topics(self) -> list[BagTopic]:
        reader = self._open_reader()
        return sorted(
            [
                BagTopic(
                    name=topic.name,
                    message_type=topic.type,
                    serialization_format=topic.serialization_format,
                )
                for topic in reader.get_all_topics_and_types()
            ],
            key=lambda topic: topic.name,
        )

    def summarize(self, topics: Iterable[str] | None = None) -> BagSummary:
        available_topics = self.topics()
        selected_topics = normalize_topic_filter(topics)
        validate_topics(selected_topics, available_topics)

        summaries = {
            topic.name: TopicSummary(topic=topic)
            for topic in available_topics
            if selected_topics is None or topic.name in selected_topics
        }
        first_timestamp_ns = None
        last_timestamp_ns = None

        for record in self.iter_serialized_messages(selected_topics):
            summaries[record.topic].add(record.timestamp_ns)
            if first_timestamp_ns is None or record.timestamp_ns < first_timestamp_ns:
                first_timestamp_ns = record.timestamp_ns
            if last_timestamp_ns is None or record.timestamp_ns > last_timestamp_ns:
                last_timestamp_ns = record.timestamp_ns

        topic_summaries = list(summaries.values())
        return BagSummary(
            bag_uri=self.bag_uri,
            storage_id=self.storage_id,
            topics=topic_summaries,
            total_count=sum(summary.count for summary in topic_summaries),
            first_timestamp_ns=first_timestamp_ns,
            last_timestamp_ns=last_timestamp_ns,
        )

    def iter_serialized_messages(
        self,
        topics: Iterable[str] | None = None,
    ) -> Iterator[SerializedBagRecord]:
        selected_topics = normalize_topic_filter(topics)
        validate_topics(selected_topics, self.topics())

        reader = self._open_reader()
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            if selected_topics is not None and topic not in selected_topics:
                continue
            yield SerializedBagRecord(topic=topic, timestamp_ns=int(timestamp_ns), data=data)

    def iter_messages(self, topics: Iterable[str] | None = None) -> Iterator[BagRecord]:
        selected_topics = normalize_topic_filter(topics)
        available_topics = self.topics()
        validate_topics(selected_topics, available_topics)
        topic_types = {topic.name: topic.message_type for topic in available_topics}
        message_classes = {}
        deserialize_message, get_message = import_ros_message_helpers()

        for record in self.iter_serialized_messages(selected_topics):
            message_type = topic_types[record.topic]
            if message_type not in message_classes:
                try:
                    message_classes[message_type] = get_message(message_type)
                except (AttributeError, ModuleNotFoundError, ValueError) as exc:
                    raise RosbagReadError(
                        f'Could not load message type {message_type!r}. '
                        'Source this workspace before reading bags that use local interfaces: '
                        'source install/setup.zsh'
                    ) from exc

            message = deserialize_message(record.data, message_classes[message_type])
            yield BagRecord(
                topic=record.topic,
                timestamp_ns=record.timestamp_ns,
                message_type=message_type,
                message=message,
            )

    def _open_reader(self):
        rosbag2_py = import_rosbag2_py()
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(self.bag_uri),
            storage_id=self.storage_id,
        )
        converter_options = rosbag2_py.ConverterOptions('', '')

        try:
            reader.open(storage_options, converter_options)
        except RuntimeError as exc:
            raise RosbagReadError(
                f'Could not open rosbag at {self.bag_uri} with storage_id={self.storage_id!r}'
            ) from exc

        return reader


def import_rosbag2_py():
    try:
        import rosbag2_py
    except ModuleNotFoundError as exc:
        raise RosbagReadError(
            'rosbag2_py is not available. Source ROS 2 before running this script, '
            'for example: source /opt/ros/humble/setup.zsh'
        ) from exc

    return rosbag2_py


def import_ros_message_helpers():
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ModuleNotFoundError as exc:
        raise RosbagReadError(
            'ROS message helpers are not available. Source ROS 2 before running this script, '
            'for example: source /opt/ros/humble/setup.zsh'
        ) from exc

    return deserialize_message, get_message


def import_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RosbagReadError(
            'numpy is not available. Install tof_dataset dependencies before exporting datasets.'
        ) from exc

    return np


def import_point_cloud2():
    try:
        from sensor_msgs_py import point_cloud2
    except ModuleNotFoundError as exc:
        raise RosbagReadError(
            'sensor_msgs_py is not available. Source ROS 2 before exporting PointCloud2 datasets.'
        ) from exc

    return point_cloud2


def normalize_bag_uri(path: Path) -> Path:
    bag_path = path.expanduser()
    if bag_path.is_file() and (bag_path.name == 'metadata.yaml' or bag_path.suffix in {'.db3', '.mcap'}):
        bag_path = bag_path.parent

    if not bag_path.exists():
        raise RosbagReadError(f'rosbag path does not exist: {bag_path}')

    if not bag_path.is_dir():
        raise RosbagReadError(f'rosbag path must be a bag directory: {bag_path}')

    return bag_path


def detect_storage_id(bag_uri: Path) -> str:
    metadata_path = bag_uri / 'metadata.yaml'
    if not metadata_path.exists():
        return 'sqlite3'

    for line in metadata_path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped.startswith('storage_identifier:'):
            continue

        storage_id = stripped.split(':', 1)[1].strip().strip('"\'')
        if storage_id:
            return storage_id

    return 'sqlite3'


def normalize_topic_filter(topics: Iterable[str] | None) -> set[str] | None:
    if topics is None:
        return None

    normalized_topics = set()
    for topic in topics:
        stripped = topic.strip()
        if not stripped:
            continue
        normalized_topics.add(stripped if stripped.startswith('/') else f'/{stripped}')

    return normalized_topics or None


def validate_topics(selected_topics: set[str] | None, available_topics: Sequence[BagTopic]) -> None:
    if selected_topics is None:
        return

    available_names = {topic.name for topic in available_topics}
    missing_topics = sorted(selected_topics - available_names)
    if missing_topics:
        available = ', '.join(sorted(available_names)) or '(none)'
        missing = ', '.join(missing_topics)
        raise RosbagReadError(f'Unknown topic(s): {missing}. Available topics: {available}')


def timestamp_from_header(message: object, fallback_ns: int) -> int:
    header = getattr(message, 'header', None)
    stamp = getattr(header, 'stamp', None)
    if stamp is None:
        return fallback_ns

    timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return timestamp_ns if timestamp_ns > 0 else fallback_ns


def quaternion_to_yaw(orientation: object) -> float:
    siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
    cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def collect_tof_frames(
    reader: RosbagReader,
    topic: str,
    width: int,
    height: int,
    min_distance_m: float,
    max_distance_m: float,
) -> tuple[list[ToFFrame], int]:
    np = import_numpy()
    point_cloud2 = import_point_cloud2()
    frames = []
    seen_timestamps = set()
    duplicate_count = 0

    for record in reader.iter_messages([topic]):
        message = record.message
        timestamp_ns = timestamp_from_header(message, record.timestamp_ns)
        if timestamp_ns in seen_timestamps:
            duplicate_count += 1
            continue
        seen_timestamps.add(timestamp_ns)

        if int(message.width) != width or int(message.height) != height:
            raise RosbagReadError(
                f'{topic} frame at {timestamp_ns} has size {message.width}x{message.height}; '
                f'expected {width}x{height}'
            )

        points = list(
            point_cloud2.read_points(
                message,
                field_names=('x', 'y', 'z'),
                skip_nans=False,
            )
        )
        expected_points = width * height
        if len(points) != expected_points:
            raise RosbagReadError(
                f'{topic} frame at {timestamp_ns} has {len(points)} points; '
                f'expected {expected_points}'
            )

        raw_points = np.asarray(points)
        if raw_points.dtype.names is not None:
            xyz = np.stack(
                [raw_points['x'], raw_points['y'], raw_points['z']],
                axis=1,
            ).astype(np.float32)
        else:
            xyz = raw_points.astype(np.float32)

        xyz = xyz.reshape(height, width, 3)
        distance = np.linalg.norm(xyz, axis=2).astype(np.float32)
        valid = np.isfinite(distance)
        valid &= distance >= min_distance_m
        valid &= distance <= max_distance_m
        distance = np.where(valid, distance, 0.0).astype(np.float32)
        frames.append(ToFFrame(timestamp_ns=timestamp_ns, distance_m=distance, valid=valid))

    frames.sort(key=lambda frame: frame.timestamp_ns)
    return frames, duplicate_count


def collect_odom_poses(reader: RosbagReader, topic: str) -> list[PoseSample]:
    poses_by_timestamp = {}
    for record in reader.iter_messages([topic]):
        message = record.message
        timestamp_ns = timestamp_from_header(message, record.timestamp_ns)
        pose = message.pose.pose
        poses_by_timestamp[timestamp_ns] = PoseSample(
            timestamp_ns=timestamp_ns,
            x=float(pose.position.x),
            y=float(pose.position.y),
            theta=quaternion_to_yaw(pose.orientation),
        )

    poses = [poses_by_timestamp[key] for key in sorted(poses_by_timestamp)]
    if len(poses) < 2:
        raise RosbagReadError(f'{topic} must contain at least two odometry poses')
    return poses


def interpolate_pose(timestamp_ns: int, poses: Sequence[PoseSample]) -> PoseSample | None:
    if timestamp_ns < poses[0].timestamp_ns or timestamp_ns > poses[-1].timestamp_ns:
        return None

    import bisect

    pose_timestamps = [pose.timestamp_ns for pose in poses]
    index = bisect.bisect_left(pose_timestamps, timestamp_ns)
    if index < len(poses) and poses[index].timestamp_ns == timestamp_ns:
        return poses[index]
    if index == 0 or index >= len(poses):
        return None

    before = poses[index - 1]
    after = poses[index]
    span = after.timestamp_ns - before.timestamp_ns
    if span <= 0:
        return None

    alpha = (timestamp_ns - before.timestamp_ns) / span
    theta_delta = normalize_angle(after.theta - before.theta)
    return PoseSample(
        timestamp_ns=timestamp_ns,
        x=before.x + alpha * (after.x - before.x),
        y=before.y + alpha * (after.y - before.y),
        theta=normalize_angle(before.theta + alpha * theta_delta),
    )


def relative_pose_label(start: PoseSample, end: PoseSample, label_mode: str):
    np = import_numpy()
    dx_world = end.x - start.x
    dy_world = end.y - start.y
    cos_theta = math.cos(start.theta)
    sin_theta = math.sin(start.theta)
    dx_body = cos_theta * dx_world + sin_theta * dy_world
    dy_body = -sin_theta * dx_world + cos_theta * dy_world
    dtheta = normalize_angle(end.theta - start.theta)

    if label_mode == 'se2':
        return np.asarray([dx_body, dy_body, dtheta], dtype=np.float32)
    if label_mode == 'diff_drive':
        return np.asarray([dx_body, dtheta], dtype=np.float32)
    raise RosbagReadError(f'unsupported label mode: {label_mode}')


def build_dataset_arrays(
    frames: Sequence[ToFFrame],
    poses: Sequence[PoseSample],
    max_distance_m: float,
    label_mode: str,
):
    np = import_numpy()
    synced_frames = []
    synced_poses = []
    skipped_frames = 0

    for frame in frames:
        pose = interpolate_pose(frame.timestamp_ns, poses)
        if pose is None:
            skipped_frames += 1
            continue
        synced_frames.append(frame)
        synced_poses.append(pose)

    if len(synced_frames) < 2:
        raise RosbagReadError(
            'not enough synchronized ToF frames with odometry poses to create samples'
        )

    inputs = []
    labels = []
    sample_timestamps_ns = []

    for index in range(1, len(synced_frames)):
        previous_frame = synced_frames[index - 1]
        current_frame = synced_frames[index]
        previous_pose = synced_poses[index - 1]
        current_pose = synced_poses[index]

        previous_distance = (previous_frame.distance_m / max_distance_m).astype(np.float32)
        current_distance = (current_frame.distance_m / max_distance_m).astype(np.float32)
        previous_distance = np.where(previous_frame.valid, previous_distance, 0.0)
        current_distance = np.where(current_frame.valid, current_distance, 0.0)

        sample_input = np.stack(
            [
                previous_distance,
                previous_frame.valid.astype(np.float32),
                current_distance,
                current_frame.valid.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        inputs.append(sample_input)
        labels.append(relative_pose_label(previous_pose, current_pose, label_mode))
        sample_timestamps_ns.append(current_frame.timestamp_ns)

    distance_m = np.stack([frame.distance_m for frame in synced_frames], axis=0).astype(np.float32)
    valid = np.stack([frame.valid for frame in synced_frames], axis=0).astype(bool)
    pose_xytheta = np.asarray(
        [[pose.x, pose.y, pose.theta] for pose in synced_poses],
        dtype=np.float32,
    )
    tof_timestamps_ns = np.asarray(
        [frame.timestamp_ns for frame in synced_frames],
        dtype=np.int64,
    )

    return {
        'inputs': np.stack(inputs, axis=0).astype(np.float32),
        'labels': np.stack(labels, axis=0).astype(np.float32),
        'sample_timestamps_ns': np.asarray(sample_timestamps_ns, dtype=np.int64),
        'tof_timestamps_ns': tof_timestamps_ns,
        'distance_m': distance_m,
        'valid': valid,
        'pose_xytheta': pose_xytheta,
        'channel_names': np.asarray(
            ['prev_distance', 'prev_valid', 'curr_distance', 'curr_valid'],
            dtype='<U32',
        ),
        'label_names': np.asarray(label_names_for_mode(label_mode), dtype='<U32'),
        'skipped_tof_frames': np.asarray([skipped_frames], dtype=np.int64),
    }


def label_names_for_mode(label_mode: str) -> list[str]:
    if label_mode == 'se2':
        return ['delta_x', 'delta_y', 'delta_theta']
    if label_mode == 'diff_drive':
        return ['delta_s', 'delta_theta']
    raise RosbagReadError(f'unsupported label mode: {label_mode}')


def export_dataset(
    reader: RosbagReader,
    output_dir: Path,
    tof_topic: str,
    odom_topic: str,
    width: int,
    height: int,
    min_distance_m: float,
    max_distance_m: float,
    label_mode: str,
    output_name: str | None = None,
) -> DatasetExportResult:
    np = import_numpy()
    topics = reader.topics()
    validate_topics(normalize_topic_filter([tof_topic, odom_topic]), topics)

    tof_frames, duplicate_tof_frames = collect_tof_frames(
        reader,
        tof_topic,
        width,
        height,
        min_distance_m,
        max_distance_m,
    )
    if len(tof_frames) < 2:
        raise RosbagReadError(f'{tof_topic} must contain at least two unique ToF frames')

    poses = collect_odom_poses(reader, odom_topic)
    arrays = build_dataset_arrays(tof_frames, poses, max_distance_m, label_mode)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = output_name or reader.bag_uri.name
    npz_path = output_dir / f'{dataset_name}.npz'
    metadata_path = output_dir / f'{dataset_name}.json'
    np.savez_compressed(npz_path, **arrays)

    metadata = {
        'bag_uri': str(reader.bag_uri),
        'storage_id': reader.storage_id,
        'tof_topic': tof_topic,
        'odom_topic': odom_topic,
        'width': width,
        'height': height,
        'min_distance_m': min_distance_m,
        'max_distance_m': max_distance_m,
        'label_mode': label_mode,
        'channel_names': arrays['channel_names'].tolist(),
        'label_names': arrays['label_names'].tolist(),
        'tof_frame_count_raw_unique': len(tof_frames),
        'tof_frame_count_exported': int(arrays['tof_timestamps_ns'].shape[0]),
        'sample_count': int(arrays['inputs'].shape[0]),
        'skipped_tof_frames': int(arrays['skipped_tof_frames'][0]),
        'duplicate_tof_frames': duplicate_tof_frames,
        'npz_path': str(npz_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    return DatasetExportResult(
        npz_path=npz_path,
        metadata_path=metadata_path,
        frame_count=int(arrays['tof_timestamps_ns'].shape[0]),
        sample_count=int(arrays['inputs'].shape[0]),
        skipped_tof_frames=int(arrays['skipped_tof_frames'][0]),
        duplicate_tof_frames=duplicate_tof_frames,
    )


def format_export_result(result: DatasetExportResult) -> str:
    return '\n'.join(
        [
            f'exported: {result.npz_path}',
            f'metadata: {result.metadata_path}',
            f'tof_frames: {result.frame_count}',
            f'samples: {result.sample_count}',
            f'skipped_tof_frames: {result.skipped_tof_frames}',
            f'duplicate_tof_frames: {result.duplicate_tof_frames}',
        ]
    )


def format_summary(summary: BagSummary) -> str:
    lines = [
        f'bag: {summary.bag_uri}',
        f'storage: {summary.storage_id}',
        f'messages: {summary.total_count}',
    ]

    if summary.first_timestamp_ns is not None and summary.last_timestamp_ns is not None:
        duration = (summary.last_timestamp_ns - summary.first_timestamp_ns) / 1_000_000_000.0
        lines.append(f'duration: {duration:.3f}s')

    lines.append('topics:')
    for topic_summary in summary.topics:
        topic = topic_summary.topic
        window = format_topic_window(topic_summary)
        lines.append(
            f'  {topic.name} [{topic.message_type}] '
            f'count={topic_summary.count}{window}'
        )

    return '\n'.join(lines)


def format_topic_window(summary: TopicSummary) -> str:
    if summary.first_timestamp_ns is None or summary.last_timestamp_ns is None:
        return ''

    start_sec = summary.first_timestamp_ns / 1_000_000_000.0
    end_sec = summary.last_timestamp_ns / 1_000_000_000.0
    return f' first={start_sec:.9f}s last={end_sec:.9f}s'


def format_preview(records: Sequence[BagRecord], first_timestamp_ns: int | None) -> str:
    if not records:
        return 'preview: no messages'

    lines = ['preview:']
    for record in records:
        offset_sec = 0.0
        if first_timestamp_ns is not None:
            offset_sec = (record.timestamp_ns - first_timestamp_ns) / 1_000_000_000.0
        lines.append(
            f'  +{offset_sec:.3f}s {record.topic} '
            f'[{record.message_type}] {describe_message(record.message)}'
        )

    return '\n'.join(lines)


def describe_message(message: object, max_length: int = 180) -> str:
    if hasattr(message, 'linear') and hasattr(message, 'angular'):
        return describe_twist(message)

    if hasattr(message, 'ranges') and hasattr(message, 'angle_min') and hasattr(message, 'angle_max'):
        return describe_laser_scan(message)

    if hasattr(message, 'width') and hasattr(message, 'height') and hasattr(message, 'fields'):
        return describe_point_cloud2(message)

    if hasattr(message, 'pose') and hasattr(message, 'twist') and hasattr(message.pose, 'pose'):
        return describe_odometry(message)

    if hasattr(message, 'transforms'):
        return describe_tf_message(message)

    if hasattr(message, 'clock'):
        clock = message.clock
        return f'Clock {clock.sec}.{clock.nanosec:09d}s'

    if hasattr(message, 'data') and isinstance(message.data, str):
        return trim_text(f'String data={message.data!r}', max_length)

    return trim_text(repr(message).replace('\n', ' '), max_length)


def describe_twist(message: object) -> str:
    return (
        'Twist '
        f'linear=({message.linear.x:.3f}, {message.linear.y:.3f}, {message.linear.z:.3f}) '
        f'angular=({message.angular.x:.3f}, {message.angular.y:.3f}, {message.angular.z:.3f})'
    )


def describe_laser_scan(message: object) -> str:
    finite_ranges = [value for value in message.ranges if math.isfinite(value)]
    range_text = 'range=(empty)'
    if finite_ranges:
        range_text = f'range=({min(finite_ranges):.3f}, {max(finite_ranges):.3f})'

    return (
        'LaserScan '
        f'frame={get_frame_id(message)!r} '
        f'ranges={len(message.ranges)} {range_text} '
        f'angle=({message.angle_min:.3f}, {message.angle_max:.3f})'
    )


def describe_point_cloud2(message: object) -> str:
    field_names = ','.join(field.name for field in message.fields)
    return (
        'PointCloud2 '
        f'frame={get_frame_id(message)!r} '
        f'size={message.width}x{message.height} '
        f'point_step={message.point_step} fields={field_names}'
    )


def describe_odometry(message: object) -> str:
    position = message.pose.pose.position
    linear = message.twist.twist.linear
    angular = message.twist.twist.angular
    return (
        'Odometry '
        f'frame={get_frame_id(message)!r} '
        f'position=({position.x:.3f}, {position.y:.3f}, {position.z:.3f}) '
        f'linear=({linear.x:.3f}, {linear.y:.3f}, {linear.z:.3f}) '
        f'angular_z={angular.z:.3f}'
    )


def describe_tf_message(message: object) -> str:
    pairs = [
        f'{transform.header.frame_id}->{transform.child_frame_id}'
        for transform in message.transforms[:3]
    ]
    suffix = '' if len(message.transforms) <= 3 else f' +{len(message.transforms) - 3} more'
    return f'TFMessage transforms={len(message.transforms)} {"; ".join(pairs)}{suffix}'


def get_frame_id(message: object) -> str:
    header = getattr(message, 'header', None)
    return getattr(header, 'frame_id', '')


def trim_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return f'{text[:max_length - 3]}...'


def collect_preview(
    reader: RosbagReader,
    topics: Iterable[str] | None,
    limit: int,
) -> list[BagRecord]:
    records = []
    for record in reader.iter_messages(topics):
        records.append(record)
        if len(records) >= limit:
            break

    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Inspect or export ROS 2 bag files for offline ToF odometry datasets.',
    )
    parser.add_argument(
        'bag',
        type=Path,
        help='ROS 2 bag directory, metadata.yaml, or storage file path.',
    )
    parser.add_argument(
        '--topic',
        dest='topics',
        action='append',
        help='Topic to read in inspect mode. Can be passed multiple times.',
    )
    parser.add_argument(
        '--storage-id',
        help='rosbag2 storage id. Defaults to metadata.yaml storage_identifier or sqlite3.',
    )
    parser.add_argument(
        '--preview',
        type=int,
        default=DEFAULT_PREVIEW_LIMIT,
        help=f'Number of deserialized messages to print in inspect mode. Default: {DEFAULT_PREVIEW_LIMIT}.',
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export a CNN training dataset from ToF PointCloud2 and odometry topics.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Directory where exported dataset files are written. Default: {DEFAULT_OUTPUT_DIR}.',
    )
    parser.add_argument(
        '--output-name',
        help='Dataset filename stem. Defaults to the bag directory name.',
    )
    parser.add_argument(
        '--tof-topic',
        default=DEFAULT_TOF_TOPIC,
        help=f'PointCloud2 ToF topic used for export. Default: {DEFAULT_TOF_TOPIC}.',
    )
    parser.add_argument(
        '--odom-topic',
        default=DEFAULT_ODOM_TOPIC,
        help=f'Odometry topic used for labels. Default: {DEFAULT_ODOM_TOPIC}.',
    )
    parser.add_argument(
        '--width',
        type=int,
        default=DEFAULT_WIDTH,
        help=f'Expected ToF image width. Default: {DEFAULT_WIDTH}.',
    )
    parser.add_argument(
        '--height',
        type=int,
        default=DEFAULT_HEIGHT,
        help=f'Expected ToF image height. Default: {DEFAULT_HEIGHT}.',
    )
    parser.add_argument(
        '--min-distance',
        type=float,
        default=DEFAULT_MIN_DISTANCE_M,
        help=f'Minimum valid ToF distance in meters. Default: {DEFAULT_MIN_DISTANCE_M}.',
    )
    parser.add_argument(
        '--max-distance',
        type=float,
        default=DEFAULT_MAX_DISTANCE_M,
        help=f'Maximum valid ToF distance in meters and normalization scale. Default: {DEFAULT_MAX_DISTANCE_M}.',
    )
    parser.add_argument(
        '--label-mode',
        choices=['se2', 'diff_drive'],
        default='se2',
        help='Label shape: se2 gives dx,dy,dtheta; diff_drive gives ds,dtheta. Default: se2.',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preview < 0:
        raise RosbagReadError('--preview must be non-negative')
    if args.width <= 0 or args.height <= 0:
        raise RosbagReadError('--width and --height must be positive')
    if args.min_distance < 0.0:
        raise RosbagReadError('--min-distance must be non-negative')
    if args.max_distance <= args.min_distance:
        raise RosbagReadError('--max-distance must be greater than --min-distance')

    reader = RosbagReader(args.bag, storage_id=args.storage_id)

    if args.export:
        result = export_dataset(
            reader=reader,
            output_dir=args.output_dir,
            tof_topic=args.tof_topic,
            odom_topic=args.odom_topic,
            width=args.width,
            height=args.height,
            min_distance_m=args.min_distance,
            max_distance_m=args.max_distance,
            label_mode=args.label_mode,
            output_name=args.output_name,
        )
        print(format_export_result(result))
        return 0

    summary = reader.summarize(args.topics)
    print(format_summary(summary))

    if args.preview:
        print()
        records = collect_preview(reader, args.topics, args.preview)
        print(format_preview(records, summary.first_timestamp_ns))

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except RosbagReadError as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc

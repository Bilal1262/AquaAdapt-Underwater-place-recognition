# Dataset ingestion

The source ROS1 bag and TUM trajectory are immutable inputs. `AnyReader` opens the bag
without ROS, and only candidate image connections are deserialized. Reports are persisted
before extraction. Frame sampling is based on bag timestamps in nanoseconds.

Metadata retains trajectory ID, source topic/type/encoding, source timestamp, dimensions,
image path, nearest pose and time error, position, normalized quaternion, validity, and
split. CSV is authoritative; Parquet is also written when an engine is installed.

For one trajectory, adjacent frames are kept in chronological blocks separated by guard
gaps. For multi-trajectory training, source manifests are concatenated and chronological
train/guard/validation blocks are assigned independently inside each trajectory. The
original source manifests are never modified.

Every row retains `trajectory_id`. Temporal and spatial positive mining explicitly masks
out observations from other trajectories because their pose coordinate systems are not
assumed to share a reference frame. Weighted sampling gives each training trajectory
equal expected contribution even when their frame counts differ.

Held-out evaluation configurations use `splits.policy: all_test`; no frame from that
trajectory enters training or checkpoint selection.

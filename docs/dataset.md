# Dataset ingestion

The source ROS1 bag and TUM trajectory are immutable inputs. `AnyReader` opens the bag
without ROS, and only candidate image connections are deserialized. Reports are persisted
before extraction. Frame sampling is based on bag timestamps in nanoseconds.

Metadata retains trajectory ID, source topic/type/encoding, source timestamp, dimensions,
image path, nearest pose and time error, position, normalized quaternion, validity, and
split. CSV is authoritative; Parquet is also written when an engine is installed.

For one trajectory, adjacent frames are kept in chronological blocks separated by guard
gaps. For multiple trajectories, `trajectory_splits` assigns whole trajectory IDs, so
adding `mclab_2` requires new configuration/manifests rather than a pipeline redesign.


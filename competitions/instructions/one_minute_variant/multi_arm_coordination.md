# Multi-Arm Coordination Protocol

This file fixes the inter-arm coordination protocol for the four-arm Medtronic NeuroSpeed 1.0 in the 1-minute variant. The future Commit 3 control loop must implement the heartbeat, the cross-arm safety zone gating, the cumulative force limit enforcement, and the heartbeat-failure auto-park behavior described below.

## Heartbeat

All 4 arms exchange a 32-byte status frame at 1 kHz over a deterministic real-time bus. The bus is logically a single broadcast channel; physically it can be EtherCAT, TSN Ethernet, or a CAN-FD ring. The simulation uses a logical broadcast queue.

Each 32-byte frame carries:

- arm_id enum (1 byte)
- tick_us little-endian uint32 (4 bytes)
- ee_x, ee_y, ee_z little-endian float32 (12 bytes)
- ee_force_magnitude little-endian float32 (4 bytes)
- safety_zone enum (1 byte)
- robot_state enum (1 byte)
- heartbeat_seq little-endian uint32 (4 bytes)
- crc32 little-endian uint32 (4 bytes)
- reserved (1 byte)

Each arm transmits its own frame at 1 kHz. Each arm receives the other 3 arms' frames at 1 kHz. The frame deadline is 1 ms from transmission to receipt at all 3 sibling arms.

## Cross-Arm Safety Zone Gating

If any arm enters the FORBIDDEN safety zone, all 4 arms emergency-park within 5 ms. The 5 ms budget breaks down as:

- 1 ms heartbeat frame transmission to all 3 sibling arms.
- 1 ms per-arm trajectory replanning to the nearest safe park position.
- 2 ms per-arm motion to the safe park position at maximum braking deceleration.
- 1 ms per-arm settling and force re-zeroing.

The trigger condition is any single frame reporting safety_zone equals FORBIDDEN from any arm. The trigger is broadcast to the other 3 arms by setting heartbeat_ok = 0 in their next outgoing frame and by raising the local estop_state to 1.

## Tool Changeover

Tool changeover is not used in 1-minute resection. All four arms are assigned their tools during the precomputed pre-op window and do not change tools during the 60-second procedure. The 1-minute variant therefore omits the tool changeover state machine that would otherwise apply.

## Force Sharing

Cumulative tip force across all 4 arms must remain under 12 N on the patient frame. Per-arm tip force limit is 5.0 N per `robot_specification_neurospeed.md`. The mathematical relationship is sum over arm = 1 to 4 of arm_force_magnitude must remain less than or equal to 12 N at every tick. Note that 4 times 5.0 N equals 20 N exceeds the 12 N cumulative cap; the protocol therefore enforces the cumulative cap in addition to the per-arm cap.

The cumulative force is computed at 10 kHz from the per-arm 10 kHz force samples. If the cumulative exceeds 11.0 N (a 1.0 N margin below the 12 N cap), each arm receives a clamp_to_force_share command that reduces its commanded velocity proportionally to its current contribution. If the cumulative still exceeds 12 N at the next tick, all 4 arms emergency-park per the cross-arm safety zone gating above.

## Communication Failure

If heartbeat is missed for 3 consecutive frames (3 ms), emergency-park triggers automatically. The miss detection is implemented per arm as a watchdog timer that resets on each successfully received heartbeat from each sibling arm. When the watchdog timer for any sibling exceeds 3 ms the local arm sets heartbeat_ok = 0 in its next outgoing frame and initiates the emergency-park sequence.

The 3 consecutive frame budget is intentionally short. At 1 kHz heartbeat, 3 ms is the minimum window that distinguishes a transient bus glitch (typically 1 to 2 frames missed) from a sustained communication failure. The 5 ms emergency-park budget plus the 3 ms detection window means that a sustained communication failure is fully resolved within 8 ms of the first missed frame.

## Per-Arm Workspace and Collision Avoidance

The four arms share a 0.5 m radius hemisphere workspace centered on the surgical target. Each arm has its own preferred working sector that the trajectory planner enforces:

- Arm 1: front sector facing the surgeon (0 to 90 degrees azimuth from patient anterior).
- Arm 2: front-right sector (270 to 360 degrees azimuth).
- Arm 3: back sector (90 to 270 degrees azimuth, lower hemisphere only to leave room for the imaging arm).
- Arm 4: top sector (upper hemisphere, facing the iMRI bore).

Inter-arm collision is prevented by a per-tick distance check between each pair of end-effector positions. The minimum allowed inter-arm distance is 8 mm (a 1 mm safety margin above the 7 mm physical clearance between the tool housings). If any pair of end-effector positions falls below 8 mm, the trailing arm pauses and the leading arm continues.

## Emergency Arm-Park Trigger Latency

The 100 microsecond emergency arm-park trigger latency is measured from the moment a triggering condition is detected by any arm to the moment all 4 arms have set their commanded velocity to zero. The 100 microsecond budget is part of the 5 ms total emergency-park sequence; it covers the inter-arm broadcast plus the per-arm command write. The remaining 4.9 ms of the 5 ms budget covers the physical motion to the park position.

## Heartbeat Sequence Numbers and Replay Protection

Each frame carries a monotonically increasing heartbeat_seq integer per arm. A receiver that observes a non-monotonic heartbeat_seq from any sibling arm treats it as a protocol violation, sets heartbeat_ok = 0, and initiates emergency-park. This prevents replay of stale frames and prevents a faulty arm from masquerading as a different arm.

## ASCII Coordination Diagram

```
+==========================================================================+
|     4-ARM COORDINATION HEARTBEAT (1 kHz, 32-byte frame, 1 ms deadline)   |
+==========================================================================+
|                                                                          |
|        +-------+  1 kHz broadcast  +-------+                             |
|        | ARM 1 |<----------------->| ARM 2 |                             |
|        | hyb.  |                   | bipol |                             |
|        | u-w-p |                   | + irr |                             |
|        +---+---+                   +---+---+                             |
|            |                           |                                 |
|            | safety_zone, ee_pos,      |                                 |
|            | force_mag, robot_state,   |                                 |
|            | heartbeat_seq, crc32      |                                 |
|            v                           v                                 |
|        +-------+                   +-------+                             |
|        | ARM 3 |<----------------->| ARM 4 |                             |
|        | suct. |  1 kHz broadcast  | iMRI  |                             |
|        | + col |                   | + ALA |                             |
|        +-------+                   +-------+                             |
|                                                                          |
|    Cumulative ee_force_magnitude across 4 arms <= 12 N                   |
|    Per-arm tip force <= 5.0 N (5x tighter than ROSA's 15 N)              |
|    E-stop latency budget 5 ms (10x faster than ROSA's 50 ms)             |
|    Heartbeat miss watchdog 3 ms triggers emergency-park sequence         |
|    Inter-arm minimum distance 8 mm (1 mm above 7 mm tool clearance)      |
+==========================================================================+
```

The diagram above is 28 lines and 78 columns; it conforms to the 80-column 60-line cap defined in `competitions/instructions/ascii_diagram_guide.md`. The future Commit 1 architecture document for the 1-minute output tree must embed a copy of this diagram verbatim in `docs/architecture.md` and must save a standalone copy as `docs/architecture_overview_4arm.txt`.

## Implementation Targets in the Future Output Tree

The future Commit 3 must implement the heartbeat protocol in the C++ control loop and in the Python mapper. Required files:

- `competitions/glioblastoma-1min-trial/src/coordination/arm_heartbeat.cpp`. C++20 single-file implementation of the 1 kHz heartbeat sender and receiver. Uses POSIX shared memory and atomic operations for the in-process simulation; the same code can be deployed against EtherCAT, TSN, or CAN-FD by replacing the transport layer in a follow-on release.
- `competitions/glioblastoma-1min-trial/src/control/robot_loop_4arm.cpp`. C++20 single-file real-time control loop for all 4 arms. Subscribes to the heartbeat broadcast, enforces the cumulative force limit at 10 kHz, and triggers emergency-park within the 5 ms budget when any safety condition is violated.
- `competitions/glioblastoma-1min-trial/src/mapping/sensor_to_xyz_4arm.py`. Python 3.10 reference implementation that mirrors the C++ control loop logic for use in the per-iteration deterministic sweep. The Python and C++ implementations must produce bit-identical command outputs for a fixed seed and a fixed iteration parameter tuple.

## Source Files Cited

- `competitions/instructions/one_minute_variant/robot_specification_neurospeed.md`. Source for the per-arm tip force limit of 5.0 N, the 5 ms E-stop latency limit, and the per-arm tool assignment that fixes which arm is responsible for which task.
- `competitions/instructions/one_minute_variant/sensor_specification_10khz.md`. Source for the 10 kHz force sample rate that drives the cumulative force computation and the heartbeat_ok channel that propagates the emergency-park trigger.
- `competitions/instructions/ascii_diagram_guide.md`. Source for the ASCII diagram format and the 80-column 60-line cap that the coordination diagram above conforms to.
- `competitions/instructions/file_format_conventions.md`. Source for the C++20 and Python 3.10 source code conventions.
- `patient-journey/stage_05_surgery.py`. Source for the IEC 80601-2-77 force limits and the FORBIDDEN safety zone enum value used by the cross-arm safety zone gating.

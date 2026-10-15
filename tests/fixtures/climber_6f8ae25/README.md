Frozen SmallTarget / BigTarget from CCZU-Climber/Climber_Vision_26 commit
6f8ae2537d33f495f55135d1cccf90301e774878 (fetched 2026-10-15).
Namespace/includes, minimal observation type and covariance accessor are adapted.
Scalar finite-value guards are removed to follow this project's requested coding constraint.
The independent randomized RANSAC diagnostic is replaced by a no-op: its result
never affects EKF state, covariance, predict(), point_buff2world(), or validity.
The reference preserves upstream static clocks and SmallTarget loss behavior;
scenarios run in separate processes. Production deliberately uses instance clocks,
geometric validity checks and bounded coasting for both sizes.
The old sp_legacy_target fixtures remain for historical provenance, not acceptance.

Solver and BigTargetGroup sources are also frozen from the same commit.
Group selector methods are excluded; slot association and blade matching are preserved.
Solver only adapts includes and namespace; tests supply identity IMU conjugation
so current camera -> gimbal -> world semantics can be compared directly.

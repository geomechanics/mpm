# Thread-Local P2G Accumulation

## Summary

This branch ports the thread-local P2G prototype to the current
`geomechanics/mpm` repository.

The change replaces the hottest explicit single-phase P2G accumulation paths
with per-thread nodal buffers followed by a parallel node reduction. It keeps
the existing particle, node, cell, and material object model intact.

## Scope

| Path | Status |
|---|---|
| Mass / momentum P2G for `FLIP` and `PIC` | Thread-local |
| Body-force P2G | Thread-local |
| Internal-force P2G | Thread-local |
| Body + internal force traversal | Fused in `MPMScheme` when rotation forces are disabled |
| `APIC`, `ASFLIP`, `TPIC` mass / momentum | Existing locked path |
| Particle traction | Existing locked path |
| Pressure smoothing | Existing locked path |
| Multimaterial and two-phase paths | Existing locked path |

## Correctness Checks

Performed locally on 2026-06-08:

| Command | Result |
|---|---|
| `cmake --build build-master -j2` | Passed |
| `mpirun -np 1 ./build-master/mpmtest_unit "[p2g]"` | Passed, 270 assertions |
| `mpirun -np 1 env OMP_NUM_THREADS=4 ./build-master/mpmtest_unit "[p2g]"` | Passed, 270 assertions |
| `mpirun -np 1 ./build-master/mpmtest_small` | Passed, 1049 assertions in 55 test cases |
| `mpirun -np 1 ./build-master/mpmtest_unit` | Passed, 50672 assertions in 110 test cases |
| `mpirun -np 1 env OMP_NUM_THREADS=1 ./build-master/mpmtest_unit "P2G thread-local performance benchmark"` | Passed |
| `mpirun -np 1 env OMP_NUM_THREADS=4 ./build-master/mpmtest_unit "P2G thread-local performance benchmark"` | Passed |

Direct binary execution without `mpirun` fails in this sandbox because the
MPI-enabled binary calls `MPI_Init` and OpenMPI singleton startup cannot create
its PMIx listener. Running through `mpirun -np 1` works.

## New Test

`tests/p2g_equivalence_test.cc` creates a small 3D mesh, runs the existing
locked P2G path, captures nodal mass, momentum, external force, and internal
force, then resets nodes and runs the new thread-local path on the same
particles. The test compares all nodal fields to `1e-9`.

## Benchmark Hook

`tests/p2g_performance_test.cc` is a hidden Catch2 benchmark (`[.]`). It builds
a regular 32^3-cell / 262144-particle problem and times:

- mass / momentum P2G
- body-force P2G
- internal-force P2G

The hidden tag means this benchmark compiles but does not run in normal test
suites.

Local benchmark output on this VM:

| Threads | Thread-local path | Locked path | Notes |
|---:|---:|---:|---|
| 1 | 294.6 ms/iter | 305.4 ms/iter | Thread-local is roughly parity/slightly faster |
| 4 | 194.0 ms/iter | skipped | Locked multi-thread path intentionally skipped to avoid long contention runs |

These numbers should be treated as a smoke benchmark only. A formal upstream
claim still needs a larger full-step run on a real OpenMP node.

## Design Notes

- Buffers are stored on `Mesh` and reused between calls.
- Buffers are indexed by node id. This is simple and fast for dense structured
  ids, but can waste memory for sparse ids.
- The solver keeps the existing mass/momentum mapper for `APIC`, `ASFLIP`, and
  `TPIC` because those paths use different particle-level velocity
  reconstruction.
- Rotation force mapping remains on the existing particle mapper. The common
  no-rotation path uses the fused body/internal force traversal.

## Remaining Work

1. Run a larger full-step benchmark on a real OpenMP node.
2. Consider compact local node indexing if sparse global node ids are common.
3. Convert pressure smoothing and traction P2G only after the main explicit
   hot path is accepted.
4. Evaluate a separate MPI halo-exchange PR independently; this branch focuses
   on thread-level node-lock contention.

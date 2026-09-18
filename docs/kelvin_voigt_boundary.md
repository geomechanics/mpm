# Kelvin–Voigt Absorbing Boundary

## Introduction

The Kelvin–Voigt absorbing boundary condition is used to reduce artificial wave reflections from the boundaries of a computational domain during dynamic simulations.

The boundary formulation combines a viscous dashpot response with a spring response. The viscous component provides resistance proportional to boundary velocity, while the spring component accounts for boundary displacement.

The absorbing boundary is currently implemented as a **nodal boundary condition**, where the resulting boundary traction is converted into forces applied to the boundary nodes.

---

## Input Definition

The absorbing boundary condition is defined within the `boundary_conditions` section of the input file.

```json
{
  "boundary_conditions": {
    "absorbing_constraints": [
      {
        "nset_id": 1,
        "dir": 0,
        "delta": 100.0,
        "h_min": 20.0,
        "a": 1.0,
        "b": 1.0,
        "position": "edge"
      }
    ]
  }
}
```

### Parameters

| Parameter  | Description                                                   |
| ---------- | ------------------------------------------------------------- |
| `nset_id`  | ID of the node set to which the boundary condition is applied |
| `dir`      | Direction associated with P-wave propagation                  |
| `delta`    | Virtual viscous layer thickness                               |
| `h_min`    | Characteristic cell length                                    |
| `a`        | Dimensionless weighting factor for the P-wave contribution    |
| `b`        | Dimensionless weighting factor for the S-wave contribution    |
| `position` | Position of the node on the boundary                          |

The `position` parameter identifies the geometric location of a boundary
node:

* `corner`
* `edge`
* `face`

`face` is applicable to three-dimensional simulations.

---

## Mathematical Formulation

The absorbing boundary traction consists of a viscous contribution and a
spring contribution.

The general traction can be expressed as

$$
\mathbf{t}_{abs}
=
\rho \,
\mathbf{v}
\odot
\mathbf{c}
+
\mathbf{u}
\odot
\mathbf{k},
$$

where

* $\rho$ is the material density,
* $\mathbf{v}$ is the boundary velocity,
* $\mathbf{u}$ is the boundary displacement,
* $\mathbf{c}$ is the wave-velocity vector,
* $\mathbf{k}$ is the spring-constant vector, and
* $\odot$ denotes component-wise multiplication.

### Wave velocities

The P-wave direction uses the P-wave velocity, while the remaining directions
use the S-wave velocity:

$$
c_{dir} = a c_p,
$$

and

$$
c_i = b c_s,
\qquad i \ne dir.
$$

Here, $c_p$ and $c_s$ are the P-wave and S-wave velocities, respectively.

### Spring constants

The spring constants are defined using the corresponding wave velocity and
virtual layer thickness:

$$
k_p = \frac{\rho c_p^2}{\delta},
$$

$$
k_s = \frac{\rho c_s^2}{\delta}.
$$

Thus, the P-wave direction uses $k_p$, while the remaining directions use
$k_s$.

### Nodal force

The absorbing traction is converted into a nodal force according to the
location of the node on the boundary.

For two-dimensional simulations,

$$
\mathbf{f}_{abs}
=
\begin{cases}
\frac{1}{2}h_{\min}\mathbf{t}_{abs},
& \text{corner},\\[6pt]
h_{\min}\mathbf{t}_{abs},
& \text{edge}.
\end{cases}
$$

For three-dimensional simulations,

$$
\mathbf{f}_{abs}
=
\begin{cases}
\frac{1}{4}h_{\min}^2\mathbf{t}_{abs},
& \text{corner},\\[6pt]
\frac{1}{2}h_{\min}^2\mathbf{t}_{abs},
& \text{edge},\\[6pt]
h_{\min}^2\mathbf{t}_{abs},
& \text{face}.
\end{cases}
$$

The resulting absorbing force is applied to the boundary node in opposition
to the calculated absorbing traction.

---

## Relevant Functions

The following functions participate in the absorbing boundary implementation.

### Initialization

| Function                          | Class       | Purpose                                                             |
| --------------------------------- | ----------- | ------------------------------------------------------------------- |
| `absorbing_boundary_properties()` | `MPMScheme` | Initializes the nodal properties required by the absorbing boundary |
| `map_wave_velocities_to_nodes()`  | `Particle`  | Maps particle wave velocities and density to the associated nodes   |

### Input and Constraint Assignment

| Function                                        | Class         | Purpose                                                           |
| ----------------------------------------------- | ------------- | ----------------------------------------------------------------- |
| `nodal_absorbing_constraints(const Json&, ...)` | `MPMBase`     | Reads absorbing boundary definitions from the input configuration |
| `assign_nodal_absorbing_constraint(...)`        | `Constraints` | Assigns an absorbing constraint to a node set                     |
| `assign_absorbing_id_ptr(...)`                  | `Constraints` | Stores the absorbing constraint and associated node-set ID        |

### Boundary Application

| Function                          | Class     | Purpose                                                                |
| --------------------------------- | --------- | ---------------------------------------------------------------------- |
| `nodal_absorbing_constraints()`   | `MPMBase` | Applies the configured absorbing constraints during the simulation     |
| `apply_absorbing_constraint(...)` | `Node`    | Calculates and applies the absorbing force at an individual node       |
| `update_external_force(...)`      | `Node`    | Updates the node's external force using the calculated absorbing force |

### Configuration Object

The absorbing boundary parameters are stored by:

```cpp
AbsorbingConstraint
```

with accessor functions for:

```cpp
setid()
dir()
delta()
h_min()
a()
b()
position()
```

Together, these functions provide the complete path from the input definition
of an absorbing boundary to its application at individual nodes.

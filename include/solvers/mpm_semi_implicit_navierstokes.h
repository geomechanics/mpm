#ifndef MPM_MPM_SEMI_IMPLICIT_NAVIER_STOKES_H_
#define MPM_MPM_SEMI_IMPLICIT_NAVIER_STOKES_H_

#ifdef USE_GRAPH_PARTITIONING
#include "graph.h"
#endif

#include "mpm_base.h"

#include "assembler_base.h"
#include "solver_base.h"

namespace mpm {

//! MPMSemiImplicit Navier Stokes class
//! \brief A class that implements the fractional step navier-stokes mpm
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MPMSemiImplicitNavierStokes : public MPMBase<Tdim> {
 public:
  //! Default constructor
  MPMSemiImplicitNavierStokes(const std::shared_ptr<IO>& io);

  //! Return matrix assembler pointer
  std::shared_ptr<mpm::AssemblerBase<Tdim>> matrix_assembler() {
    return assembler_;
  }

  //! Solve
  bool solve() override;

  //! Class private functions
 private:
  //! Initialise matrix
  bool initialise_matrix();

  //! Initialise matrix
  bool reinitialise_matrix();

  //! Compute poisson equation
  bool compute_poisson_equation();

  //! Compute corrected velocity
  bool compute_correction_force();

  //! Class private variables
 private:
  // Generate a unique id for the analysis
  using mpm::MPMBase<Tdim>::uuid_;
  //! Time step size
  using mpm::MPMBase<Tdim>::dt_;
  //! Current step
  using mpm::MPMBase<Tdim>::step_;
  //! Number of steps
  using mpm::MPMBase<Tdim>::nsteps_;
  //! Number of steps
  using mpm::MPMBase<Tdim>::nload_balance_steps_;
  //! Output steps
  using mpm::MPMBase<Tdim>::output_steps_;
  //! A unique ptr to IO object
  using mpm::MPMBase<Tdim>::io_;
  //! JSON analysis object
  using mpm::MPMBase<Tdim>::analysis_;
  //! JSON post-process object
  using mpm::MPMBase<Tdim>::post_process_;
  //! Logger
  using mpm::MPMBase<Tdim>::console_;
  //! Stress update
  using mpm::MPMBase<Tdim>::stress_update_;
  //! velocity update
  using mpm::MPMBase<Tdim>::velocity_update_;
  //! FLIP-PIC blending ratio
  using mpm::MPMBase<Tdim>::blending_ratio_;
  //! Gravity
  using mpm::MPMBase<Tdim>::gravity_;
  //! Mesh object
  using mpm::MPMBase<Tdim>::mesh_;
  //! Materials
  using mpm::MPMBase<Tdim>::materials_;
  //! Nonlocal cell neighbourhood
  using mpm::MPMBase<Tdim>::cell_neighbourhood_;
  //! Nonlocal node neighbourhood
  using mpm::MPMBase<Tdim>::node_neighbourhood_;
  //! Pressure smoothing
  bool pressure_smoothing_{false};
  // Projection method parameter (beta)
  double beta_{1};
  //! Assembler object
  std::shared_ptr<mpm::AssemblerBase<Tdim>> assembler_;
  //! Linear solver object
  tsl::robin_map<std::string,
                 std::shared_ptr<mpm::SolverBase<Eigen::SparseMatrix<double>>>>
      linear_solver_;
  //! Method to detect free surface detection
  std::string free_surface_detection_;
  //! Volume tolerance for free surface
  double fs_vol_tolerance_{0.25};

};  // MPMSemiImplicit class
}  // namespace mpm

#include "mpm_semi_implicit_navierstokes.tcc"

#endif  // MPM_MPM_SEMI_IMPLICIT_NAVIER_STOKES_H_

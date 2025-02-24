#ifndef MPM_MPM_SCHEME_NEWMARK_H_
#define MPM_MPM_SCHEME_NEWMARK_H_

#ifdef USE_GRAPH_PARTITIONING
#include "graph.h"
#endif

#include "mpm_scheme.h"

namespace mpm {

//! MPMSchemeNewmark class
//! \brief MPMSchemeNewmark Derived class for Newmark predictor-corrector scheme
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MPMSchemeNewmark : public MPMScheme<Tdim> {
 public:
  //! Default constructor with mesh class
  MPMSchemeNewmark(const std::shared_ptr<mpm::Mesh<Tdim>>& mesh, double dt);

  //! Intialize
  inline void initialise() override;

  //! Compute nodal kinematics - map mass, momentum and inertia to nodes
  //! \param[in] velocity_update Method to update nodal velocity
  //! \param[in] phase Phase to smooth pressure
  inline void compute_nodal_kinematics(mpm::VelocityUpdate velocity_update,
                                       unsigned phase) override;

  //! Compute stress and strain
  //! \param[in] phase Phase to smooth pressure
  //! \param[in] pressure_smoothing Enable or disable pressure smoothing
  inline void compute_stress_strain(unsigned phase,
                                    bool pressure_smoothing) override;

  //! Precompute stress
  //! \param[in] phase Phase to smooth pressure
  //! \param[in] pressure_smoothing Enable or disable pressure smoothing
  inline void precompute_stress_strain(unsigned phase,
                                       bool pressure_smoothing) override;

  //! Postcompute stress
  //! \param[in] phase Phase to smooth pressure
  //! \param[in] pressure_smoothing Enable or disable pressure smoothing
  inline void postcompute_stress_strain(unsigned phase,
                                        bool pressure_smoothing) override;

  //! Compute acceleration velocity position
  //! \param[in] velocity_update Method to update particle velocity
  //! \param[in] blending_ratio FLIP-PIC blending ratio
  //! \param[in] phase Phase of particle
  //! \param[in] damping_type Type of damping
  //! \param[in] damping_factor Value of critical damping
  inline void compute_particle_kinematics(mpm::VelocityUpdate velocity_update,
                                          double blending_ratio, unsigned phase,
                                          const std::string& damping_type,
                                          double damping_factor,
                                          unsigned step) override;

  //! Postcompute nodal kinematics - map mass and momentum to nodes
  //! \param[in] velocity_update Method to update nodal velocity
  //! \param[in] phase Phase to smooth pressure
  inline void postcompute_nodal_kinematics(mpm::VelocityUpdate velocity_update,
                                           unsigned phase) override;

  //! Stress update scheme
  //! \retval scheme Stress update scheme
  inline std::string scheme() const override;

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Compute forces
  //! \ingroup Implicit
  //! \param[in] gravity Acceleration due to gravity
  //! \param[in] step Number of step in solver
  //! \param[in] concentrated_nodal_forces Boolean for if a concentrated force
  //! is applied or not
  //! \param[in] quasi_static Boolean of quasi-static analysis
  inline void compute_forces(const Eigen::Matrix<double, Tdim, 1>& gravity,
                             unsigned phase, unsigned step,
                             bool concentrated_nodal_forces,
                             bool quasi_static) override;

  //! Update nodal kinematics by Newmark scheme
  //! \ingroup Implicit
  //! \param[in] newmark_beta Parameter beta of Newmark scheme
  //! \param[in] newmark_gamma Parameter gamma of Newmark scheme
  //! \param[in] phase Phase to smooth pressure
  inline void update_nodal_kinematics_newmark(unsigned phase,
                                              double newmark_beta,
                                              double newmark_gamma) override;

  // Update particle stress, strain and volume
  //! \ingroup Implicit
  inline void update_particle_stress_strain_volume() override;
  /**@}*/

  /**
   * \defgroup Implicit Functions dealing with implicit thermo-mechanical MPM
   */
  /**@{*/
  //! Initialize nodes, cells and shape functions
  inline void initialise_thermal() override;

  //! Compute nodal temperature and temperature rate
  inline void compute_nodal_temperatures(unsigned phase, 
                                          double dt, Index step) override;

  //! Update nodal kinematics by Newmark scheme
  inline void update_nodal_thermokinematics_newmark(
      unsigned phase, double newmark_beta, double newmark_gamma, 
      Index step) override;

  //! Compute stress and strain by Newmark scheme
  inline void compute_stress_strain_thermal(
      unsigned phase, bool pressure_smoothing) override;

  //! Compute forces
  inline void compute_heats(bool vfm, std::string ftype, 
                            double vfm_param1, double vfm_param2) override;

  // Update particle temperature
  inline void compute_particle_temperature() override;
  /**@}*/

 protected:
  //! Mesh object
  using mpm::MPMScheme<Tdim>::mesh_;
  //! MPI Size
  using mpm::MPMScheme<Tdim>::mpi_size_;
  //! MPI rank
  using mpm::MPMScheme<Tdim>::mpi_rank_;
  //! Time increment
  using mpm::MPMScheme<Tdim>::dt_;

};  // MPMSchemeNewmark class
}  // namespace mpm

#include "mpm_scheme_newmark.tcc"
#include "mpm_scheme_newmark_thermal.tcc"

#endif  // MPM_MPM_SCHEME_NEWMARK_H_

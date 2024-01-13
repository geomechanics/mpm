#ifndef MPM_MPM_SCHEME_USL_H_
#define MPM_MPM_SCHEME_USL_H_

#ifdef USE_GRAPH_PARTITIONING
#include "graph.h"
#endif

#include "mpm_scheme.h"

namespace mpm {

//! MPMSchemeUSL class
//! \brief MPMSchemeUSL Derived class for USL stress update scheme
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MPMSchemeUSL : public MPMScheme<Tdim> {
 public:
  //! Default constructor with mesh class
  MPMSchemeUSL(const std::shared_ptr<mpm::Mesh<Tdim>>& mesh, double dt);

  //! Precompute stress
  //! \param[in] phase Phase to smooth pressure
  //! \param[in] pressure_smoothing Enable or disable pressure smoothing
  //! \param[in] stress_rate Use Cauchy or Jaumann rate of stress
  virtual inline void precompute_stress_strain(
      unsigned phase, bool pressure_smoothing,
      mpm::StressRate stress_rate) override;

  //! Postcompute stress
  //! \param[in] phase Phase to smooth pressure
  //! \param[in] pressure_smoothing Enable or disable pressure smoothing
  //! \param[in] stress_rate Use Cauchy or Jaumann rate of stress
  virtual inline void postcompute_stress_strain(
      unsigned phase, bool pressure_smoothing,
      mpm::StressRate stress_rate) override;

  //! Postcompute nodal kinematics - map mass and momentum to nodes
  //! \param[in] velocity_update Method to update nodal velocity
  //! \param[in] phase Phase to smooth pressure
  virtual inline void postcompute_nodal_kinematics(
      mpm::VelocityUpdate velocity_update, unsigned phase) override;

  //! Stress update scheme
  //! \retval scheme Stress update scheme
  virtual inline std::string scheme() const override;

 protected:
  //! Mesh object
  using mpm::MPMScheme<Tdim>::mesh_;
  //! MPI Size
  using mpm::MPMScheme<Tdim>::mpi_size_;
  //! MPI rank
  using mpm::MPMScheme<Tdim>::mpi_rank_;
  //! Time increment
  using mpm::MPMScheme<Tdim>::dt_;

};  // MPMSchemeUSL class
}  // namespace mpm

#include "mpm_scheme_usl.tcc"

#endif  // MPM_MPM_SCHEME_USL_H_

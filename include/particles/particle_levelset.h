#ifndef MPM_PARTICLE_LEVELSET_H_
#define MPM_PARTICLE_LEVELSET_H_

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "logger.h"
#include "particle.h"

namespace mpm {

//! Levelset subclass
//! \brief subclass that stores the information about levelset particle
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ParticleLevelset : public Particle<Tdim> {

 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Initialise particle levelset properties
  void initialise() override;

  //! Assign nodal Levelset value to particles
  //! \param[in] dt Analysis time step
  void map_particle_contact_force_to_nodes(double dt) override;

 private:
  //! Compute Levelset contact force
  //! \param[in] dt Analysis time step
  //! \param[in] levelset Levelset value at the particle
  //! \param[in] levelset_normal Normal vector towards the levelset //LEDT check
  //! \param[in] levelset_mu Levelset friction
  //! \param[in] barrier_stiffness Barrier stiffness
  //! \param[in] slip_threshold Slip threshold
  //! \param[in] levelset_mp_radius mp radius of influence for contact
  inline VectorDim compute_levelset_contact_force(
      double levelset, const VectorDim& levelset_normal, double levelset_mu,
      double barrier_stiffness, double slip_threshold,
      double levelset_mp_radius, double dt) noexcept;

 private:
  //! coupling force
  Eigen::Matrix<double, Tdim, 1> couple_force_{0.};
  //! levelset value
  double levelset{0.};
  //! levelset normal
  const VectorDim& levelset_normal{0.};
  //! levelset friction
  double levelset_mu{0.};
  //! barrier stiffness
  double barrier_stiffness{0.};
  //! slip threshold
  double slip_threshold{0.};
  //! slip threshold
  double levelset_mp_radius{0.};
  //! cumulative slip magnitude
  double cumulative_slip_mag{0.};
  //! Nodes
  using ParticleBase<Tdim>::nodes_;
  //! Shape functions
  Eigen::VectorXd shapefn_;  // LEDT check: node_shapefn was in particle.tcc
  //! dN/dX
  Eigen::MatrixXd dn_dx_;
  //! Velocity
  Eigen::Matrix<double, Tdim, 1> velocity_;

};  // Particle_Levelset class

}  // namespace mpm

#include "particle_levelset.tcc"

#endif  // MPM_PARTICLE_LEVELSET_H__
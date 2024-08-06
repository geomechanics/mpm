#ifndef MPM_PARTICLE_LEVELSET_H_
#define MPM_PARTICLE_LEVELSET_H_

#include "logger.h"
#include "math_utility.h"
#include "particle.h"

#include <cmath>

namespace mpm {

//! Levelset subclass
//! \brief subclass that stores the information about levelset particle
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ParticleLevelset : public Particle<Tdim> {

 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Construct a levelset particle with id and coordinates
  //! \param[in] id Particle id
  //! \param[in] coord Coordinates of the particles
  ParticleLevelset(Index id, const VectorDim& coord);

  //! Construct a levelset particle with id, coordinates and status
  //! \param[in] id Particle id
  //! \param[in] coord coordinates of the particle
  //! \param[in] status Particle status (active / inactive)
  ParticleLevelset(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~ParticleLevelset() override{};

  //! Delete copy constructor
  ParticleLevelset(const ParticleLevelset<Tdim>&) = delete;

  //! Delete assignment operator
  ParticleLevelset& operator=(const ParticleLevelset<Tdim>&) = delete;

  //! Assign nodal Levelset value to particles
  //! \param[in] dt Analysis time step
  //! \param[in] levelset_damping Levelset damping factor
  void map_particle_contact_force_to_nodes(const double levelset_damping,
                                           double dt) override;

  //! Return the approximate particle diameter
  double diameter() const override;

 private:
  //! Compute Levelset contact force
  //! \param[in] levelset Levelset value at the particle
  //! \param[in] levelset_normal Normal vector towards the levelset
  //! \param[in] levelset_mu Levelset friction
  //! \param[in] levelset_alpha Levelset adhesion coefficient
  //! \param[in] barrier_stiffness Barrier stiffness
  //! \param[in] slip_threshold Slip threshold
  //! \param[in] mp_radius mp radius of influence for contact
  //! \param[in] contact_vel Contact velocity from nodes (PIC)
  //! \param[in] levelset_damping Levelset damping factor
  //! \param[in] dt Analysis time step
  VectorDim compute_levelset_contact_force(
      double levelset, const VectorDim& levelset_normal, double levelset_mu,
      double levelset_alpha, double barrier_stiffness, double slip_threshold,
      const double mp_radius, const VectorDim& contact_vel,
      const double levelset_damping, double dt) noexcept;

 private:
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! coupling force
  VectorDim couple_force_{VectorDim::Zero()};
  //! levelset value
  double levelset{0.};
  //! levelset friction
  double levelset_mu{0.};
  //! levelset adhesion coefficient
  double levelset_alpha{0.};
  //! barrier stiffness
  double barrier_stiffness{0.};
  //! slip threshold
  double slip_threshold{0.};
  //! cumulative slip magnitude
  double cumulative_slip_mag{0.};  // LEDT check not reseting each step
  //! contact velocity
  VectorDim contact_vel{VectorDim::Zero()};
  //! Nodes
  using Particle<Tdim>::nodes_;
  //! Cell
  using Particle<Tdim>::cell_;
  //! Shape functions
  using Particle<Tdim>::shapefn_;
  //! dN/dX
  using Particle<Tdim>::dn_dx_;
  //! Velocity
  using Particle<Tdim>::velocity_;
  //! Volume
  using Particle<Tdim>::volume_;
  //! Mass
  using Particle<Tdim>::mass_;
  //! Size of particle
  using Particle<Tdim>::size_;
  //! particleBase id
  using Particle<Tdim>::id_;
};  // Particle_Levelset class

}  // namespace mpm

#include "particle_levelset.tcc"

#endif  // MPM_PARTICLE_LEVELSET_H__
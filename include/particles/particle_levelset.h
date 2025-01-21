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

  //! Return empty levelset couple vector
  VectorDim levelset_couple() const override { return VectorDim::Zero(); };

  //! Compute particle contact forces and map to nodes
  //! \param[in] dt Analysis time step
  void map_particle_contact_force_to_nodes(double dt) override;

  //! Update time-independent static levelset properties
  //! \param[in] levelset_damping Levelset damping factor
  //! \param[in] levelset_pic Particle in cell method bool for contact velocity
  static void update_levelset_static_properties(double levelset_damping,
                                                bool levelset_pic);

  //! Update time-independent mp levelset properties
  void update_levelset_mp_properties() override;

 protected:
  //! Compute levelset contact force
  //! \param[in] dt Analysis time step
  void compute_levelset_contact_force(double dt) noexcept;

 protected:
  //! Logger
  std::unique_ptr<spdlog::logger> console_;

  //! levelset_damping_
  static double levelset_damping_;
  //! levelset_pic_
  static bool levelset_pic_;
  //! mp radius
  double mp_radius_{0.};
  //! levelset value
  double levelset_{0.};
  //! levelset friction
  double levelset_mu_{0.};
  //! levelset adhesion coefficient
  double levelset_alpha_{0.};
  //! barrier stiffness
  double barrier_stiffness_{0.};
  //! slip threshold
  double slip_threshold_{0.};
  //! cumulative slip magnitude
  double cumulative_slip_mag_{0.};
  //! levelset gradient
  VectorDim levelset_gradient_{VectorDim::Zero()};
  //! contact velocity
  VectorDim contact_vel_{VectorDim::Zero()};
  //! levelset normal
  VectorDim levelset_normal_{VectorDim::Zero()};
  //! levelset tangent
  VectorDim levelset_tangent_{VectorDim::Zero()};
  //! coupling force
  VectorDim couple_force_{VectorDim::Zero()};
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
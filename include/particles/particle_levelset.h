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

  //! Type of particle
  std::string type() const override { return (Tdim == 2) ? "P2DLS" : "P3DLS"; }

  //! Initialise particle levelset
  void initialise() override;

  //! Update contact force due to levelset
  //! \param[in] dt Analysis time step
  //! \param[in] levelset_damping Levelset damping factor
  //! \param[in] levelset_pic Method bool to compute contact velocity
  void levelset_contact_force(double dt, double levelset_damping,
                              bool levelset_pic) override;

  //! Return levelset value
  double levelset() const { return levelset_; }

  //! Return levelset contact force
  VectorDim couple_force() const { return couple_force_; }

 protected:
  //! Map levelset to particle
  void map_levelset_to_particle() noexcept;

  //! Check if particle in contact with levelset
  //! \param[in] init_radius Particle initial radius
  bool is_levelset_contact(double init_radius);

  //! Compute levelset contact force at particle
  //! \param[in] dt Analysis time step
  //! \param[in] init_radius Particle initial radius
  //! \param[in] levelset_damping Levelset damping factor
  //! \param[in] levelset_pic Method bool to compute contact velocity
  void compute_particle_contact_force(double dt, double init_radius,
                                      double levelset_damping,
                                      bool levelset_pic) noexcept;

  //! Map levelset contact force to nodes
  void map_contact_force_to_nodes() noexcept;

 protected:
  //! Logger
  std::unique_ptr<spdlog::logger> console_;

  //! levelset value
  double levelset_{0.};
  //! levelset friction
  double levelset_mu_{0.};
  //! levelset adhesion coefficient
  double levelset_alpha_{0.};
  //! barrier stiffness
  double barrier_stiffness_{0.};
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
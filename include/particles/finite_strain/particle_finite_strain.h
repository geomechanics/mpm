#ifndef MPM_PARTICLE_FINITE_STRAIN_H_
#define MPM_PARTICLE_FINITE_STRAIN_H_

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "logger.h"
#include "particle.h"

namespace mpm {

//! ParticleFiniteStrain class
//! \brief Class for the finite strain single-phase particle
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ParticleFiniteStrain : public mpm::Particle<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Construct a particle using finite strain formulation with id and
  //! coordinates
  //! \param[in] id Particle id
  //! \param[in] coord Coordinates of the
  //! particles
  ParticleFiniteStrain(Index id, const VectorDim& coord);

  //! Construct a particle using finite strain formulation with id, coordinates
  //! and status
  //! \param[in] id Particle id
  //! \param[in] coord coordinates of the particle
  //! \param[in] status Particle status (active / inactive)
  ParticleFiniteStrain(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~ParticleFiniteStrain() override{};

  //! Delete copy constructor
  ParticleFiniteStrain(const ParticleFiniteStrain<Tdim>&) = delete;

  //! Delete assignment operator
  ParticleFiniteStrain& operator=(const ParticleFiniteStrain<Tdim>&) = delete;

  //! Type of particle
  std::string type() const override { return (Tdim == 2) ? "P2DFS" : "P3DFS"; }

  //! Update volume based on centre volumetric strain rate
  virtual void update_volume() noexcept override;

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Compute strain using nodal displacement
  //! \ingroup Implicit
  void compute_strain_newmark() noexcept override;

  //! Compute stress using implicit updating scheme
  //! \ingroup Implicit
  void compute_stress_newmark() noexcept override;
  /**@}*/

 protected:
  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Compute strain increment
  //! \ingroup Implicit
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval strain increment at particle inside a cell
  inline Eigen::Matrix<double, 6, 1> compute_strain_increment(
      const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept override;
  /**@}*/

 protected:
  //! Nodes
  using ParticleBase<Tdim>::nodes_;
  //! State variables
  using ParticleBase<Tdim>::state_variables_;
  //! Material
  using ParticleBase<Tdim>::material_;
  //! Volume
  using Particle<Tdim>::volume_;
  //! Stresses
  using Particle<Tdim>::stress_;
  //! Strains
  using Particle<Tdim>::strain_;
  //! dvolumetric strain
  using Particle<Tdim>::dvolumetric_strain_;
  //! Volumetric strain at centroid
  using Particle<Tdim>::volumetric_strain_centroid_;
  //! Strain rate
  using Particle<Tdim>::strain_rate_;
  //! dstrains
  using Particle<Tdim>::dstrain_;
  //! Velocity
  using Particle<Tdim>::velocity_;
  //! Displacement
  using Particle<Tdim>::displacement_;
  //! dN/dX
  using Particle<Tdim>::dn_dx_;
  //! dN/dX at cell centroid
  using Particle<Tdim>::dn_dx_centroid_;

  //! Logger
  std::unique_ptr<spdlog::logger> console_;

  /**
   * \defgroup ImplicitVariables Variables dealing with implicit MPM
   */
  /**@{*/
  //! Stresses at the last time step
  Eigen::Matrix<double, 6, 1> previous_stress_;
  /**@}*/

  /**
   * \defgroup FiniteStrainVariables Variables for finite strain formulation
   */
  /**@{*/
  //! Deformation gradient
  using Particle<Tdim>::deformation_gradient_;
  //! Deformation gradient increment
  using Particle<Tdim>::deformation_gradient_increment_;
  /**@}*/

};  // ParticleFiniteStrain class
}  // namespace mpm

#include "particle_finite_strain.tcc"

#endif  // MPM_PARTICLE_FINITE_STRAIN_H__

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

  //! Update volume based on deformation gradient increment
  virtual void update_volume() noexcept override;

  //! Compute deformation gradient increment using nodal velocity
  void compute_strain(double dt) noexcept override;

  //! Compute stress and update deformation gradient
  void compute_stress() noexcept override;

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Function to reinitialise consitutive law to be run at the beginning of
  //! each time step
  //! \ingroup Implicit
  void initialise_constitutive_law() noexcept override;

  //! Compute strain using nodal displacement
  //! \ingroup Implicit
  void compute_strain_newmark() noexcept override;

  //! Compute stress using implicit updating scheme
  //! \ingroup Implicit
  void compute_stress_newmark() noexcept override;
  /**@}*/

 protected:
  //! Compute deformation gradient increment using nodal displacement
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval deformaton gradient increment at particle inside a cell
  inline Eigen::Matrix<double, 3, 3> compute_deformation_gradient_increment(
      const Eigen::MatrixXd& dn_dx, unsigned phase, const double dt) noexcept;

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Compute deformation gradient increment using nodal displacement
  //! \ingroup Implicit
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval deformaton gradient increment at particle inside a cell
  inline Eigen::Matrix<double, 3, 3> compute_deformation_gradient_increment(
      const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept;

  //! Compute Hencky strain using deformation gradient
  //! \ingroup Implicit
  inline Eigen::Matrix<double, 6, 1> compute_hencky_strain(
      const Eigen::Matrix<double, 3, 3>& deformation_gradient);

  //! Update stress and strain after convergence of Newton-Raphson iteration
  //! \ingroup Implicit
  void update_stress_strain() noexcept override;
  /**@}*/

 protected:
  //! Nodes
  using ParticleBase<Tdim>::nodes_;
  //! Cell
  using ParticleBase<Tdim>::cell_;
  //! State variables
  using ParticleBase<Tdim>::state_variables_;
  //! Material
  using ParticleBase<Tdim>::material_;
  //! Volume
  using Particle<Tdim>::volume_;
  //! Volumetric mass density (mass / volume)
  using Particle<Tdim>::mass_density_;
  //! Stresses
  using Particle<Tdim>::stress_;
  //! Strains
  // using Particle<Tdim>::strain_;
  //! Velocity
  using Particle<Tdim>::velocity_;
  //! Displacement
  using Particle<Tdim>::displacement_;
  //! dN/dX
  using Particle<Tdim>::dn_dx_;

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

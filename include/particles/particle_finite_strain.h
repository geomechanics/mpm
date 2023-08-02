#ifndef MPM_PARTICLE_FINITE_STRAIN_H_
#define MPM_PARTICLE_FINITE_STRAIN_H_

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "logger.h"
#include "math_utility.h"
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

  //! Compute shape functions of a particle, based on local coordinates
  void compute_shapefn() noexcept override;

  //! Return strain of the particle
  Eigen::Matrix<double, 6, 1> strain() const override {
    const auto& strain = this->compute_hencky_strain();
    return strain;
  }

  //! Return deformation gradient increment of the particle
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment() const {
    return deformation_gradient_increment_;
  }

  //! Update volume based on deformation gradient increment
  //! Note: Volume is updated in compute_strain() and
  //! compute_strain_volume_newmark() for particle with finite strain
  void update_volume() noexcept override{};

  //! Compute deformation gradient
  //! Note: Deformation gradient is updated in update_stress_strain() and
  //! compute stress() for particle with finite strain
  void update_deformation_gradient(const std::string& type,
                                   double dt) noexcept override{};

  //! Compute deformation gradient increment using nodal velocity
  //! \param[in] dt Analysis time step
  void compute_strain(double dt) noexcept override;

  //! Compute stress and update deformation gradient
  //! \param[in] dt Analysis time step
  //! \param[in] stress_rate Use Cauchy or Jaumann rate of stress
  void compute_stress(double dt, mpm::StressRate stress_rate =
                                     mpm::StressRate::None) noexcept override;

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Function to reinitialise consitutive law to be run at the beginning of
  //! each time step
  //! \ingroup Implicit
  void initialise_constitutive_law() noexcept override;

  //! Map mass, material and geometric stiffness matrix to cell
  //! (used in equilibrium equation LHS)
  //! \ingroup Implicit
  //! \param[in] newmark_beta parameter beta of Newmark scheme
  //! \param[in] dt parameter beta of Newmark scheme
  //! \param[in] quasi_static Boolean of quasi-static analysis
  inline bool map_stiffness_matrix_to_cell(double newmark_beta, double dt,
                                           bool quasi_static) override;

  //! Compute deformation gradient and volume using nodal displacement
  //! \ingroup Implicit
  void compute_strain_volume_newmark() noexcept override;

  //! Compute stress using implicit updating scheme
  //! \ingroup Implicit
  void compute_stress_newmark() noexcept override;

  //! Update stress and strain after convergence of Newton-Raphson iteration
  //! \ingroup Implicit
  void update_stress_strain() noexcept override;
  /**@}*/

 protected:
  //! Map geometric stiffness matrix to cell (used in equilibrium equation LHS)
  //! \ingroup Implicit
  inline bool map_geometric_stiffness_matrix_to_cell();

  //! Compute Hencky strain using deformation gradient
  inline Eigen::Matrix<double, 6, 1> compute_hencky_strain() const;

 protected:
  //! particle id
  using ParticleBase<Tdim>::id_;
  //! coordinates
  using ParticleBase<Tdim>::coordinates_;
  //! Status
  using ParticleBase<Tdim>::status_;
  //! Cell
  using ParticleBase<Tdim>::cell_;
  //! Cell id
  using ParticleBase<Tdim>::cell_id_;
  //! Nodes
  using ParticleBase<Tdim>::nodes_;
  //! State variables
  using ParticleBase<Tdim>::state_variables_;
  //! Material
  using ParticleBase<Tdim>::material_;
  //! Material ids
  using ParticleBase<Tdim>::material_id_;
  //! Particle mass
  using Particle<Tdim>::mass_;
  //! Volume
  using Particle<Tdim>::volume_;
  //! Volumetric mass density (mass / volume)
  using Particle<Tdim>::mass_density_;
  //! Stresses
  using Particle<Tdim>::stress_;
  //! Stresses at the previous time step
  using Particle<Tdim>::previous_stress_;
  //! Velocity
  using Particle<Tdim>::velocity_;
  //! Acceleration
  using Particle<Tdim>::acceleration_;
  //! Displacement
  using Particle<Tdim>::displacement_;
  //! dN/dX
  using Particle<Tdim>::dn_dx_;
  //! Size of particle in natural coordinates
  using Particle<Tdim>::natural_size_;
  //! Size of particle
  using Particle<Tdim>::pack_size_;

  //! Logger
  std::unique_ptr<spdlog::logger> console_;

  /**
   * \defgroup FiniteStrainVariables Variables for finite strain formulation
   */
  /**@{*/
  //! Deformation gradient
  using Particle<Tdim>::deformation_gradient_;
  //! Deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment_{
      Eigen::Matrix<double, 3, 3>::Identity()};
  //! Shape function gradient at the reference configuration
  Eigen::MatrixXd reference_dn_dx_;
  /**@}*/

};  // ParticleFiniteStrain class
}  // namespace mpm

#include "particle_finite_strain.tcc"

#endif  // MPM_PARTICLE_FINITE_STRAIN_H__

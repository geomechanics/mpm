#ifndef MPM_PARTICLE_PML_UNSPLIT_H_
#define MPM_PARTICLE_PML_UNSPLIT_H_

#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "logger.h"
#include "math_utility.h"
#include "particle.h"
#include "particle_pml.h"

namespace mpm {

//! ParticleUPML class
//! \brief Class for the Perfectly Matched Layer (PML) particle
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ParticleUPML : public mpm::ParticlePML<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Construct a particle using PML formulation with id and
  //! coordinates
  //! \param[in] id Particle id
  //! \param[in] coord Coordinates of the
  //! particles
  ParticleUPML(Index id, const VectorDim& coord);

  //! Construct a particle using PML formulation with id, coordinates
  //! and status
  //! \param[in] id Particle id
  //! \param[in] coord coordinates of the particle
  //! \param[in] status Particle status (active / inactive)
  ParticleUPML(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~ParticleUPML() override{};

  //! Delete copy constructor
  ParticleUPML(const ParticleUPML<Tdim>&) = delete;

  //! Delete assignment operator
  ParticleUPML& operator=(const ParticleUPML<Tdim>&) = delete;

  //! Type of particle
  std::string type() const override {
    return (Tdim == 2) ? "P2DUPML" : "P3DUPML";
  }

  //! Map particle mass and momentum to nodes
  //! \param[in] velocity_update Method to update nodal velocity
  void map_mass_momentum_to_nodes(
      mpm::VelocityUpdate velocity_update =
          mpm::VelocityUpdate::FLIP) noexcept override;

  // ! Map damped mass vector to nodes
  void map_pml_properties_to_nodes() noexcept override;

  //! Map body force
  //! \param[in] pgravity Gravity of a particle
  void map_body_force(const VectorDim& pgravity) noexcept override;

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Map particle mass, momentum and inertia to nodes
  //! \ingroup Implicit
  void map_mass_momentum_inertia_to_nodes(
      mpm::VelocityUpdate velocity_update =
          mpm::VelocityUpdate::FLIP) noexcept override;

  //! Map inertial force
  //! \ingroup Implicit
  void map_inertial_force(double bossak_alpha = 0.0) noexcept override;

  //! Map internal force
  void map_internal_force(double dt) noexcept override;

  /**@}*/

 protected:
  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Map material stiffness matrix to cell (used in equilibrium equation LHS)
  //! \ingroup Implicit
  //! \param[in] dt time step
  inline bool map_material_stiffness_matrix_to_cell(double dt) override;

  //! Map mass matrix to cell (used in equilibrium equation LHS)
  //! \ingroup Implicit
  //! \param[in] newmark_beta parameter beta of Newmark scheme
  //! \param[in] dt time step
  inline bool map_mass_matrix_to_cell(double newmark_beta, double bossak_alpha,
                                      double dt) override;

  //! Map PML rayleigh damping matrix to cell (used in equilibrium
  //! equation LHS)
  //! \ingroup Implicit
  //! \param[in] newmark_gamma parameter gamma of Newmark scheme
  //! \param[in] newmark_beta parameter beta of Newmark scheme
  //! \param[in] dt parameter beta of Newmark scheme
  //! \param[in] damping_factor Rayleigh damping factor
  inline bool map_rayleigh_damping_matrix_to_cell(
      double newmark_gamma, double newmark_beta, double dt,
      double damping_factor) override;

  //! Compute strain increment
  //! \ingroup Implicit
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval strain increment at particle inside a cell
  inline Eigen::Matrix<double, 6, 1> compute_strain_increment(
      const Eigen::MatrixXd& dn_dx, unsigned phase, double dt) noexcept override;

  /**@}*/

  /**
   * \defgroup AdvancedMapping Functions dealing with advance mapping scheme of
   * MPM
   */
  /**@{*/
  //! Map particle mass and momentum to nodes for affine transformation
  //! \ingroup AdvancedMapping
  void map_mass_momentum_to_nodes_affine() noexcept override;

  //! Map particle mass and momentum to nodes for approximate taylor expansion
  //! \ingroup AdvancedMapping
  void map_mass_momentum_to_nodes_taylor() noexcept override;

  //! Compute PML stiffness matrix
  inline Eigen::MatrixXd compute_pml_stiffness_matrix(double dt) noexcept;

  // Compute PML damping matrix
  inline Eigen::MatrixXd compute_pml_damping_matrix(double dt) noexcept;

  //! Update PML Properties
  //! \param[in] dt parameter beta of Newmark scheme
  void update_pml_properties(double dt) noexcept override;

  //! Function to recompute particle damping functions
  void compute_damping_functions(mpm::dense_map& state_vars) noexcept override;

  //! Compute normal damping functions
  Eigen::VectorXd normal_damping_functions() const noexcept;

  //! Compute evanescent damping functions
  Eigen::VectorXd evanescent_damping_functions() const noexcept;

  //! Map internal force to nodes from displacement component
  //! \param[in] Fe Evanescent damping functions
  //! \param[in] Fp Propagating damping functions
  //! \param[in] dt parameter beta of Newmark scheme
  void map_internal_force_strain(const Eigen::VectorXd& Fe, const Eigen::VectorXd& Fp, double dt) noexcept;

  //! Map internal force to nodes from strain component
  //! \param[in] Fp Propagating damping functions
  void map_internal_force_stress(const Eigen::VectorXd& Fp) noexcept;

  //! Map internal force to nodes from stress component
  //! \param[in] Fp Propagating damping functions
  void map_internal_force_disp(const Eigen::VectorXd& Fp, double dt) noexcept;

  //! Map internal force to nodes from stiffness component
  //! \param[in] dt parameter beta of Newmark scheme
  void map_internal_force_stiffness(double dt) noexcept;

  //! Map PML rayleigh damping force
  //! \ingroup Implicit
  //! \param[in] damping_factor Rayleigh damping factor
  //! \param[in] dt parameter beta of Newmark scheme
  void map_rayleigh_damping_force(double damping_factor,
                                  double dt) noexcept override;

  //! Compute damping matrix for "rayleigh" damping
  //! \param[in] dt parameter beta of Newmark scheme
  inline Eigen::MatrixXd compute_damped_bmatrix(
      const Eigen::VectorXd& damping_functions) noexcept;

  //! Compute damping matrix for "rayleigh" damping
  //! \param[in] dt parameter beta of Newmark scheme
  inline Eigen::MatrixXd compute_combined_bmatrix(
      const Eigen::VectorXd& Fa, const Eigen::VectorXd& Fb) noexcept;

  //! Combine damping factors
  //! \param[in] Fa First set of particle damping functions
  //! \param[in] Fb Second set of particle damping functions
  //! \param[in] identical_index Indicator for identical damping functions
  Eigen::VectorXd combined_damping_functions(
      const Eigen::VectorXd& Fa, const Eigen::VectorXd& Fb,
      bool identical_index) const noexcept;

  //! Convert damping factors to voigt notation
  //! \param[in] damping_functions Particle damping values in vector form
  Eigen::VectorXd voigt_damping_functions(
      const Eigen::VectorXd& damping_functions) const noexcept;

  //! Reduce dimensions for vector properties
  //! \param[in] prop Particle vector property to be reduced
  Eigen::MatrixXd reduce_voigt(Eigen::Matrix<double, 6, 1> prop) noexcept;

  //! Call indicatedtime integrated variable
  //! \param[in] stain_bool Indicator for time integratd strain or stress
  //! variable
  Eigen::Matrix<double, 6, 1> call_state_var(bool strain_bool) noexcept;

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
  //! Shape functions
  using Particle<Tdim>::shapefn_;
  //! dN/dX
  using Particle<Tdim>::dn_dx_;
  //! dN/dX at cell centroid
  using Particle<Tdim>::dn_dx_centroid_;
  //! Strain rate
  using Particle<Tdim>::strain_rate_;
  //! dstrains
  using Particle<Tdim>::dstrain_;
  //! Strains
  using Particle<Tdim>::strain_;
  //! dvolumetric strain
  using Particle<Tdim>::dvolumetric_strain_;
  //! Constitutive Tangent Matrix (dynamic allocation only for implicit scheme)
  using Particle<Tdim>::constitutive_matrix_;
  //! Size of particle in natural coordinates
  using Particle<Tdim>::natural_size_;
  //! Size of particle
  using Particle<Tdim>::pack_size_;
  //! Mapping matrix for advance mapping schemes
  using Particle<Tdim>::mapping_matrix_;

  //! Logger
  std::unique_ptr<spdlog::logger> console_;
};  // ParticleUPML class
}  // namespace mpm

#include "particle_pml_unsplit.tcc"

#endif  // MPM_PARTICLE_PML_UNSPLIT_H__

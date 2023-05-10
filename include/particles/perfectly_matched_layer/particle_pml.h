#ifndef MPM_PARTICLE_PML_H_
#define MPM_PARTICLE_PML_H_

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "logger.h"
#include "math_utility.h"
#include "particle.h"

namespace mpm {

//! ParticlePML class
//! \brief Class for the Perfectly Matched Layer (PML) particle
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ParticlePML : public mpm::Particle<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Construct a particle using PML formulation with id and
  //! coordinates
  //! \param[in] id Particle id
  //! \param[in] coord Coordinates of the
  //! particles
  ParticlePML(Index id, const VectorDim& coord);

  //! Construct a particle using PML formulation with id, coordinates
  //! and status
  //! \param[in] id Particle id
  //! \param[in] coord coordinates of the particle
  //! \param[in] status Particle status (active / inactive)
  ParticlePML(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~ParticlePML() override{};

  //! Delete copy constructor
  ParticlePML(const ParticlePML<Tdim>&) = delete;

  //! Delete assignment operator
  ParticlePML& operator=(const ParticlePML<Tdim>&) = delete;

  //! Type of particle
  std::string type() const override {
    return (Tdim == 2) ? "P2DPML" : "P3DPML";
  }

  //! Map particle mass and momentum to nodes
  //! \param[in] velocity_update Method to update nodal velocity
  virtual void map_mass_momentum_to_nodes(
      mpm::VelocityUpdate velocity_update =
          mpm::VelocityUpdate::FLIP) noexcept override;

  //! Map damped mass vector to nodes
  virtual void map_pml_properties_to_nodes() noexcept override;

  // ! Finalise pml properties
  void finalise_pml_properties(double dt) noexcept override;

  //! Map body force
  //! \param[in] pgravity Gravity of a particle
  virtual void map_body_force(const VectorDim& pgravity) noexcept override;

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Map particle mass, momentum and inertia to nodes
  //! \ingroup Implicit
  virtual void map_mass_momentum_inertia_to_nodes(
      mpm::VelocityUpdate velocity_update =
          mpm::VelocityUpdate::FLIP) noexcept override;

  //! Map inertial force
  //! \ingroup Implicit
  virtual void map_inertial_force(double bossak_alpha = 0.0) noexcept override;

  //! Map internal force
  virtual void map_internal_force(double dt) noexcept override;

  /**@}*/

 protected:
  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Map material stiffness matrix to cell (used in equilibrium equation LHS)
  //! \ingroup Implicit
  //! \param[in] dt time step
  virtual inline bool map_material_stiffness_matrix_to_cell(double dt) override;

  //! Map mass matrix to cell (used in equilibrium equation LHS)
  //! \ingroup Implicit
  //! \param[in] newmark_beta parameter beta of Newmark scheme
  //! \param[in] dt time step
  virtual inline bool map_mass_matrix_to_cell(double newmark_beta,
                                              double bossak_alpha,
                                              double dt) override;

  //! Map PML rayleigh damping force
  //! \ingroup Implicit
  //! \param[in] damping_factor Rayleigh damping factor
  //! \param[in] dt parameter beta of Newmark scheme
  virtual void map_rayleigh_damping_force(double damping_factor,
                                          double dt) noexcept override;

  //! Map PML rayleigh damping matrix to cell (used in equilibrium
  //! equation LHS)
  //! \ingroup Implicit
  //! \param[in] newmark_gamma parameter gamma of Newmark scheme
  //! \param[in] newmark_beta parameter beta of Newmark scheme
  //! \param[in] dt parameter beta of Newmark scheme
  //! \param[in] damping_factor Rayleigh damping factor
  virtual inline bool map_rayleigh_damping_matrix_to_cell(
      double newmark_gamma, double newmark_beta, double dt,
      double damping_factor) override;
  /**@}*/

  /**
   * \defgroup AdvancedMapping Functions dealing with advance mapping scheme of
   * MPM
   */
  /**@{*/
  //! Map particle mass and momentum to nodes for affine transformation
  //! \ingroup AdvancedMapping
  virtual void map_mass_momentum_to_nodes_affine() noexcept override;

  //! Map particle mass and momentum to nodes for approximate taylor expansion
  //! \ingroup AdvancedMapping
  virtual void map_mass_momentum_to_nodes_taylor() noexcept override;

  /**@}*/

  //! Function to recompute particle damping functions
  virtual void compute_damping_functions(mpm::dense_map& state_vars) noexcept;

  //! Function to return mass damping functions
  VectorDim mass_damping_functions() const noexcept;

  //! Compute PML stiffness matrix
  virtual inline Eigen::MatrixXd compute_pml_stiffness_matrix() noexcept;

  //! Function to update displacement functions
  virtual void update_pml_properties(double dt) noexcept;

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

};  // ParticlePML class
}  // namespace mpm

#include "particle_pml.tcc"

#endif  // MPM_PARTICLE_PML_H__

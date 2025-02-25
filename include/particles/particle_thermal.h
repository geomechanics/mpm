#ifndef MPM_PARTICLE_THERMAL_H_
#define MPM_PARTICLE_THERMAL_H_

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "logger.h"
#include "particle.h"

namespace mpm {

//! ThermalParticle class
//! \brief Class that stores the information about thermal functions
//! particles
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ThermalParticle : public mpm::Particle<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Construct a thermal particle with id and coordinates
  //! \param[in] id Particle id
  //! \param[in] coord Coordinates of the particles
  ThermalParticle(Index id, const VectorDim& coord);

  //! Construct a thermal particle with id, coordinates and status
  //! \param[in] id Particle id
  //! \param[in] coord coordinates of the particle
  //! \param[in] status Particle status (active / inactive)
  ThermalParticle(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~ThermalParticle() override{};

  //! Delete copy constructor
  ThermalParticle(const ThermalParticle<Tdim>&) = delete;

  //! Delete assignment operator
  ThermalParticle& operator=(const ThermalParticle<Tdim>&) = delete;

  //! Initialise particle from POD data
  //! \param[in] particle POD data of particle
  //! \retval status Status of reading POD particle
  bool initialise_particle(PODParticle& particle) override;

  //! Initialise particle POD data and material
  //! \param[in] particle POD data of particle
  //! \param[in] materials Material associated with the particle arranged in a
  //! vector
  //! \retval status Status of reading POD particle
  bool initialise_particle(
      PODParticle& particle,
      const std::vector<std::shared_ptr<Material<Tdim>>>& materials) override;

  //! Initialise particle liquid phase on top of the regular solid phase
  void initialise() override;

  //! Return particle data as POD
  //! \retval particle POD of the particle
  std::shared_ptr<void> pod() const override;

  // //! Initialise particle thermal properties
  // void initialise_thermal() override;

  //! Map particle heat capacity and heat to nodes
  void map_heat_to_nodes() override;

  //! Map particle heat capacity and heat to nodes
  void map_heat_to_nodes_newmark() override;

  //! Map heat conduction to nodes
  void map_heat_conduction() override;

  //! Map plastic heat dissipation to nodes
  void map_plastic_heat_dissipation(double dt) override;

  //! Map heat conduction to nodes
  void map_virtual_heat_flux(bool convective, const double vfm_param1,
                                              const double vfm_param2) override;

  //! Compute thermal strain of the particle
  void compute_thermal_strain() override;

  //! Compute updated temperature of the particle
  void update_particle_temperature(double dt) override;

  //! Map specific heat to cell
  bool map_heat_capacity_to_cell(double dt, double newmark_beta, 
                                            double newmark_gamma) override;

  //! Map heat conductivity matrix to cell
  bool map_heat_conductivity_to_cell() override;

  //! Map thermal expansivity matrix to cell
  bool map_thermal_expansivity_to_cell() override;

  //! Compute strain and volume of the particle using nodal displacement and 
  //! temperature increment
  void compute_strain_volume_newmark_thermal() override;

  //! Map transient heat to nodes
  void map_heat_rate_to_nodes() override;

  // Compute updated temperature of the particle
  void compute_updated_temperature_newmark(double dt) override; 

  //! Assign a state variable
  //! \param[in] value Particle temperature to be assigned
  //! \param[in] phase Index to indicate phase
  void assign_temperature(double temperature) override {
    this->temperature_ = temperature;
  }

 protected:
  //! Compute temperature gradient of the particle
  inline Eigen::Matrix<double, Tdim, 1> compute_temperature_gradient(
          unsigned phase) noexcept;

  //! Compute mass gradient of the particle
  inline Eigen::Matrix<double, Tdim, 1> compute_mass_gradient(
          unsigned phase) noexcept;

  /**
   * \defgroup Implicit Functions dealing with implicit thermo-mechanical MPM
   */
  /**@{*/
  //! Compute strain increment
  //! \ingroup Thermal Implicit
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval strain increment at particle inside a cell
  virtual inline Eigen::Matrix<double, 6, 1> compute_strain_increment_thermal(
      const Eigen::MatrixXd& dn_dx, double beta, unsigned phase) noexcept;

  //! Compute deformation gradient increment using nodal displacement
  //! \ingroup Thermal Implicit
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval deformaton gradient increment at particle inside a cell
  inline Eigen::Matrix<double, 3, 3> 
      compute_deformation_gradient_increment_thermal(
      const Eigen::MatrixXd& dn_dx, double beta, unsigned phase) noexcept;
  /**@}*/

 protected:
  //! particle id
  using ParticleBase<Tdim>::id_;
  //! coordinates
  using ParticleBase<Tdim>::coordinates_;
  //! Reference coordinates (in a cell)
  using ParticleBase<Tdim>::xi_;
  //! Cell
  using ParticleBase<Tdim>::cell_;
  //! Cell id
  using ParticleBase<Tdim>::cell_id_;
  //! Nodes
  using ParticleBase<Tdim>::nodes_;
  //! Status
  using ParticleBase<Tdim>::status_;
  //! Material
  using ParticleBase<Tdim>::material_;
  //! Material id
  using ParticleBase<Tdim>::material_id_;
  //! State variables
  using ParticleBase<Tdim>::state_variables_;
  //! Neighbour particles
  using ParticleBase<Tdim>::neighbours_;

  //! Particle mass density
  using Particle<Tdim>::mass_density_;
  //! Particle mass for solid phase
  using Particle<Tdim>::mass_;
  //! Particle total volume
  using Particle<Tdim>::volume_;
  //! Size of particle
  using Particle<Tdim>::size_;
  //! Size of particle in natural coordinates
  using Particle<Tdim>::natural_size_;
  //! Effective stress of soil skeleton
  using Particle<Tdim>::stress_;
  //! Solid skeleton strains
  using Particle<Tdim>::strain_;
  //! dvolumetric strain
  using Particle<Tdim>::dvolumetric_strain_;
  //! Soil skeleton strain rate
  using Particle<Tdim>::strain_rate_;
  //! Soil skeleton dstrain
  using Particle<Tdim>::dstrain_;
  //! Acceleration
  using Particle<Tdim>::acceleration_;
  //! Velocity
  using Particle<Tdim>::velocity_;
  //! Displacement
  using Particle<Tdim>::displacement_;
  //! Particle velocity constraints
  using Particle<Tdim>::particle_velocity_constraints_;
  //! Free surface
  using Particle<Tdim>::free_surface_;
  //! Free surface
  using Particle<Tdim>::normal_;
  //! Set traction
  using Particle<Tdim>::set_traction_;
  //! Surface Traction (given as a stress; force/area)
  using Particle<Tdim>::traction_;
  //! Shape functions
  using Particle<Tdim>::shapefn_;
  //! dN/dX
  using Particle<Tdim>::dn_dx_;
  //! dN/dX at cell centroid
  using Particle<Tdim>::dn_dx_centroid_;
  //! Map of scalar properties
  using Particle<Tdim>::scalar_properties_;
  //! Map of vector properties
  using Particle<Tdim>::vector_properties_;
  //! Map of tensor properties
  using Particle<Tdim>::tensor_properties_;
  //! Size of particle
  using Particle<Tdim>::pack_size_;
  //! Mapping matrix
  using Particle<Tdim>::mapping_matrix_;
  //! Deformation gradient
  using Particle<Tdim>::deformation_gradient_;
  //! Deformation gradient increment
  using Particle<Tdim>::deformation_gradient_increment_;
  //! Stresses at the last time step
  using Particle<Tdim>::previous_stress_;
  //! Constitutive Tangent Matrix (dynamic allocation only for implicit scheme)
  using Particle<Tdim>::constitutive_matrix_;  

  //! Scalar properties
  //! Temperature   
  double temperature_;
  //! PIC temperature 
  double temperature_pic_;
  //! FLIP temperature
  double temperature_flip_;
  //! Incremental volumetric thermal strain
  double dthermal_volumetric_strain_;
  //! Rate of temperature
  double temperature_rate_;  
  //! double dot of temperature
  double temperature_ddot_;    
  //! Temperature increment
  double temperature_increment_;
  //! Heat source
  double heat_source_;
  //! Density
  double density_;

  //! Vector properties
  //! temperature gradient
  Eigen::Matrix<double, Tdim, 1> temperature_gradient_;
  //! Mass gradient
  Eigen::Matrix<double, Tdim, 1> mass_gradient_;  
  //! Heat flux vector
  Eigen::Matrix<double, Tdim, 1> heat_flux_; 
  //! Unit outward normal
  Eigen::Matrix<double, Tdim, 1> outward_normal_;   

  // Tensor properties 
  //! Thermal strain
  Eigen::Matrix<double, 6, 1> thermal_strain_;
  //! Incremental thermal strain
  Eigen::Matrix<double, 6, 1> dthermal_strain_;  

  //! Bool properties
  //! Set heat source
  bool set_heat_source_{false};

  //! Logger
  std::unique_ptr<spdlog::logger> console_;

};  // ThermalParticle class
}  // namespace mpm

#include "particle_thermal.tcc"
#include "particle_thermal_implicit.tcc"

#endif  // MPM_PARTICLE_THERMAL_H__

#ifndef MPM_FLUID_PARTICLE_H_
#define MPM_FLUID_PARTICLE_H_

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "logger.h"
#include "particle.h"

namespace mpm {

//! Fluid Particle class
//! \brief Class with function specific to fluid particles
//! \tparam Tdim Dimension
template <unsigned Tdim>
class FluidParticle : public mpm::Particle<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Construct a particle with id and coordinates
  //! \param[in] id Particle id
  //! \param[in] coord Coordinates of the particles
  FluidParticle(Index id, const VectorDim& coord);

  //! Destructor
  ~FluidParticle() override{};

  //! Delete copy constructor
  FluidParticle(const FluidParticle<Tdim>&) = delete;

  //! Delete assignment operator
  FluidParticle& operator=(const FluidParticle<Tdim>&) = delete;

  //! Compute stress
  //! \param[in] dt Analysis time step
  //! \param[in] stress_rate Use Cauchy or Jaumann rate of stress
  void compute_stress(double dt, mpm::StressRate stress_rate =
                                     mpm::StressRate::None) noexcept override;

  //! Map internal force
  inline void map_internal_force() noexcept override;

  //! Serialize
  //! \retval buffer Serialized buffer data
  std::vector<uint8_t> serialize() override;

  //! Deserialize
  //! \param[in] buffer Serialized buffer data
  //! \param[in] material Particle material pointers
  void deserialize(
      const std::vector<uint8_t>& buffer,
      std::vector<std::shared_ptr<mpm::Material<Tdim>>>& materials) override;

  //! ----------------------------------------------------------------
  //! Semi-Implicit integration functions based on Chorin's Projection
  //! ----------------------------------------------------------------

  //! Assigning beta parameter to particle
  //! \param[in] parameter parameter determining type of projection
  void assign_projection_parameter(double parameter) override {
    this->projection_param_ = parameter;
  };

  //! Return projection parameter
  double projection_parameter() const override {
    return this->projection_param_;
  }

  //! Map laplacian element matrix to cell (used in poisson equation LHS)
  bool map_laplacian_to_cell() override;

  //! Map poisson rhs element matrix to cell (used in poisson equation RHS)
  bool map_poisson_right_to_cell() override;

  //! Map correction matrix element matrix to cell (used to correct velocity)
  bool map_correction_matrix_to_cell() override;

  //! Update pressure after solving poisson equation
  bool compute_updated_pressure() override;

  //! Type of particle
  std::string type() const override {
    return (Tdim == 2) ? "P2DFLUID" : "P3DFLUID";
  }

 protected:
  //! Compute pack size
  //! \retval pack size of serialized object
  int compute_pack_size() const override;

 private:
  //! Compute turbulent stress
  virtual Eigen::Matrix<double, 6, 1> compute_turbulent_stress();

 private:
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
  //! Fluid material
  using ParticleBase<Tdim>::material_;
  //! Material ids
  using ParticleBase<Tdim>::material_id_;
  //! State variables
  using ParticleBase<Tdim>::state_variables_;
  //! Shape functions
  using Particle<Tdim>::shapefn_;
  //! dN/dX
  using Particle<Tdim>::dn_dx_;
  //! Size of particle in natural coordinates
  using Particle<Tdim>::natural_size_;
  //! Deformation gradient
  using Particle<Tdim>::deformation_gradient_;
  //! Fluid strain rate
  using Particle<Tdim>::strain_rate_;
  //! Fluid strain
  using Particle<Tdim>::strain_;
  //! Fluid stress
  using Particle<Tdim>::stress_;
  //! Particle mass density
  using Particle<Tdim>::mass_density_;
  //! Displacement
  using Particle<Tdim>::displacement_;
  //! Velocity
  using Particle<Tdim>::velocity_;
  //! Acceleration
  using Particle<Tdim>::acceleration_;
  //! Particle mass density
  using Particle<Tdim>::mass_;
  //! Particle total volume
  using Particle<Tdim>::volume_;
  //! Size of particle
  using Particle<Tdim>::pack_size_;
  //! Mapping matrix
  using Particle<Tdim>::mapping_matrix_;
  //! Projection parameter for semi-implicit update
  double projection_param_{0.0};
  //! Pressure constraint
  double pressure_constraint_{std::numeric_limits<unsigned>::max()};
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
};  // FluidParticle class
}  // namespace mpm

#include "particle_fluid.tcc"

#endif  // MPM_FLUID_PARTICLE_H__

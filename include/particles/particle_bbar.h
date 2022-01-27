#ifndef MPM_PARTICLE_BBAR_H_
#define MPM_PARTICLE_BBAR_H_

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "logger.h"
#include "particle.h"

namespace mpm {

//! ParticleBbar class
//! \brief Class that stores the information about second-phase (liquid)
//! particles
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ParticleBbar : public mpm::Particle<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Construct a particle using B-bar method with id and coordinates
  //! \param[in] id Particle id
  //! \param[in] coord Coordinates of the particles
  ParticleBbar(Index id, const VectorDim& coord);

  //! Construct a particle using B-bar method with id, coordinates and status
  //! \param[in] id Particle id
  //! \param[in] coord coordinates of the particle
  //! \param[in] status Particle status (active / inactive)
  ParticleBbar(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~ParticleBbar() override{};

  //! Delete copy constructor
  ParticleBbar(const ParticleBbar<Tdim>&) = delete;

  //! Delete assignment operator
  ParticleBbar& operator=(const ParticleBbar<Tdim>&) = delete;

  //! Compute strain
  //! \param[in] dt Analysis time step
  void compute_strain(double dt) noexcept override;

  //! Map internal force
  inline void map_internal_force() noexcept override;

  //! Type of particle
  std::string type() const override {
    return (Tdim == 2) ? "P2DBBAR" : "P3DBBAR";
  }

 protected:
  //! Compute strain rate
  //! \ingroup Implicit
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval strain rate at particle inside a cell
  inline Eigen::Matrix<double, 6, 1> compute_strain_rate(
      const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept;

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
  //! Volumetric mass density (mass / volume)
  using Particle<Tdim>::mass_density_;
  //! Mass
  using Particle<Tdim>::mass_;
  //! Volume
  using Particle<Tdim>::volume_;
  //! Size of particle
  using Particle<Tdim>::size_;
  //! Size of particle in natural coordinates
  using Particle<Tdim>::natural_size_;
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
  //! Pack size
  using Particle<Tdim>::pack_size_;

  //! Logger
  std::unique_ptr<spdlog::logger> console_;

  /**
   * \defgroup ImplicitVariables Variables dealing with implicit MPM
   */
  /**@{*/
  //! Acceleration
  using Particle<Tdim>::acceleration_;
  //! Stresses at the last time step
  using Particle<Tdim>::previous_stress_;
  //! Constitutive Tangent Matrix (dynamic allocation only for implicit scheme)
  using Particle<Tdim>::constitutive_matrix_;
  /**@}*/

};  // ParticleBbar class
}  // namespace mpm

#include "particle_bbar.tcc"

#endif  // MPM_PARTICLE_BBAR_H__

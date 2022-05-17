#ifndef MPM_PARTICLE_FINITE_STRAIN_FBAR_H_
#define MPM_PARTICLE_FINITE_STRAIN_FBAR_H_

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "logger.h"
#include "particle_finite_strain.h"

namespace mpm {

//! ParticleFiniteStrainFbar class
//! \brief Class that introduce F-Bar formulation
//!        to the finite strain single-phase particle
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ParticleFiniteStrainFbar : public mpm::ParticleFiniteStrain<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Construct a particle using B-bar method with id and coordinates
  //! \param[in] id Particle id
  //! \param[in] coord Coordinates of the particles
  ParticleFiniteStrainFbar(Index id, const VectorDim& coord);

  //! Construct a particle using B-bar method with id, coordinates and status
  //! \param[in] id Particle id
  //! \param[in] coord coordinates of the particle
  //! \param[in] status Particle status (active / inactive)
  ParticleFiniteStrainFbar(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~ParticleFiniteStrainFbar() override{};

  //! Delete copy constructor
  ParticleFiniteStrainFbar(const ParticleFiniteStrainFbar<Tdim>&) = delete;

  //! Delete assignment operator
  ParticleFiniteStrainFbar& operator=(const ParticleFiniteStrainFbar<Tdim>&) =
      delete;

  //! Type of particle
  std::string type() const override {
    return (Tdim == 2) ? "P2DFSFBAR" : "P3DFSFBAR";
  }

  //! Compute deformation gradient increment using nodal velocity
  void compute_strain(double dt) noexcept override;

 protected:
  //! dN/dX
  using Particle<Tdim>::dn_dx_;
  //! dN/dX at cell centroid
  using Particle<Tdim>::dn_dx_centroid_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;

  /**
   * \defgroup FiniteStrainVariables Variables for finite strain formulation
   */
  /**@{*/
  //! Deformation gradient
  using ParticleFiniteStrain<Tdim>::deformation_gradient_increment_;

};  // ParticleFiniteStrainFbar class
}  // namespace mpm

#include "particle_finite_strain_fbar.tcc"

#endif  // MPM_PARTICLE_FINITE_STRAIN_FBAR_H__

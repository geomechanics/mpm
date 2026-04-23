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
//! \brief Class that introduce B-Bar formulation to the standard single-phase
//! particle
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

  //! Map internal force
  inline void map_internal_force() noexcept override;

  //! Type of particle
  std::string type() const override {
    return (Tdim == 2) ? "P2DBBAR" : "P3DBBAR";
  }

  /**
   * \defgroup Implicit Functions dealing with implicit MPM
   */
  /**@{*/
  //! Compute B matrix of a particle, based on local coordinates
  //! \ingroup Implicit
  inline Eigen::MatrixXd compute_bmatrix() noexcept override;
  /**@}*/

 protected:
  //! Compute strain rate
  //! \ingroup Implicit
  //! \param[in] dn_dx The spatial gradient of shape function
  //! \param[in] phase Index to indicate phase
  //! \retval strain rate at particle inside a cell
  inline Eigen::Matrix<double, 6, 1> compute_strain_rate(
      const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept override;

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
  //! Volume
  using Particle<Tdim>::volume_;
  //! Stresses
  using Particle<Tdim>::stress_;
  //! Strains
  using Particle<Tdim>::strain_;
  //! dvolumetric strain
  using Particle<Tdim>::dvolumetric_strain_;
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

};  // ParticleBbar class
}  // namespace mpm

#include "particle_bbar.tcc"

#endif  // MPM_PARTICLE_BBAR_H__

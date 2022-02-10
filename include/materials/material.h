#ifndef MPM_MATERIAL_MATERIAL_H_
#define MPM_MATERIAL_MATERIAL_H_

#include <limits>

#include "Eigen/Dense"
#include "json.hpp"

#include "factory.h"
#include "logger.h"
#include "map.h"
#include "material_utility.h"
#include "particle.h"
#include "particle_base.h"

// JSON
using Json = nlohmann::json;

namespace mpm {

// Forward declaration of ParticleBase
template <unsigned Tdim>
class ParticleBase;

//! Material base class
//! \brief Base class that stores the information about materials
//! \details Material class stresses and strains
//! \tparam Tdim Dimension
template <unsigned Tdim>
class Material {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  // Constructor with id
  //! \param[in] id Material id
  Material(unsigned id, const Json& material_properties) : id_{id} {
    //! Logger
    std::string logger = "material::" + std::to_string(id);
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  //! Destructor
  virtual ~Material(){};

  //! Delete copy constructor
  Material(const Material&) = delete;

  //! Delete assignement operator
  Material& operator=(const Material&) = delete;

  //! Return id of the material
  unsigned id() const { return id_; }

  //! Get material property
  //! \tparam Ttype Return type for proerpty
  //! \param[in] key Material property key
  //! \retval result Value of material property
  template <typename Ttype>
  Ttype property(const std::string& key);

  //! Initialise history variables
  virtual mpm::dense_map initialise_state_variables() = 0;

  //! State variables
  virtual std::vector<std::string> state_variables() const = 0;

  //! Initialise material
  //! \brief Function that initialise material to be called at the beginning of
  //! time step
  //! \param[in] state_vars History-dependent state variables
  virtual void initialise(mpm::dense_map* state_vars){};

  //! Compute stress
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of stress
  virtual Vector6d compute_stress(const Vector6d& stress,
                                  const Vector6d& dstrain,
                                  const ParticleBase<Tdim>* ptr,
                                  mpm::dense_map* state_vars) = 0;

  //！ Return Elastic-Plastic stiffness matrix
  //! \param[in] stress Stress
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] the yield status
  //! \retval dp_ Elastic-Plastic stiffness matrix
  virtual Matrix6x6 dp(const Vector6d& stress, mpm::dense_map* state_vars,
                       bool& status) {
    return Matrix6x6::Zero();
  }
  //！ Return Elastic stiffness matrix
  //! \retval de_ Elastic stiffness matrix
  virtual Matrix6x6 de() { return Matrix6x6::Zero(); }

  //! Compute constitutive relations matrix
  //! \ingroup Implicit
  //! \param[in] stress Stress
  //! Compute consistent tangent matrix
  //! \param[in] stress Updated stress
  //! \param[in] prev_stress Stress at the current step
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval dmatrix Constitutive relations mattrix
  virtual Matrix6x6 compute_consistent_tangent_matrix(
      const Vector6d& stress, const Vector6d& prev_stress,
      const Vector6d& dstrain, const ParticleBase<Tdim>* ptr,
      mpm::dense_map* state_vars) {
    auto error = Matrix6x6::Zero();
    throw std::runtime_error(
        "Calling the base class function (compute_consistent_tangent_matrix) "
        "in Material:: illegal operation!");
    return error;
  };

 protected:
  //! material id
  unsigned id_{std::numeric_limits<unsigned>::max()};
  //! Material properties
  Json properties_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
};  // Material class
}  // namespace mpm

#include "material.tcc"

#endif  // MPM_MATERIAL_MATERIAL_H_

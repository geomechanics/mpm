#ifndef MPM_MATERIAL_INFINITESIMAL_ELASTO_PLASTIC_H_
#define MPM_MATERIAL_INFINITESIMAL_ELASTO_PLASTIC_H_

#include <cmath>

#include <limits>

#include "Eigen/Dense"

#include "material.h"

// JSON
using Json = nlohmann::json;

namespace mpm {

//! Elasto-plastic material class of infinitesimal strain
//! \details Base class of all elasto-plastic material model with infinitesimal
//! strain \tparam Tdim Dimension
template <unsigned Tdim>
class InfinitesimalElastoPlastic : public Material<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] material_properties Material properties
  InfinitesimalElastoPlastic(unsigned id, const Json& material_properties)
      : Material<Tdim>(id, material_properties){};

  //! Destructor
  ~InfinitesimalElastoPlastic() override{};

  //! Delete copy constructor
  InfinitesimalElastoPlastic(const InfinitesimalElastoPlastic&) = delete;

  //! Delete assignement operator
  InfinitesimalElastoPlastic& operator=(const InfinitesimalElastoPlastic&) =
      delete;

  //! Compute consistent tangent matrix
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval dmatrix Constitutive relations mattrix
  Matrix6x6 compute_consistent_tangent_matrix(
      const Vector6d& stress, const Vector6d& dstrain,
      const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) override;

 protected:
  //! Compute constitutive relations matrix
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval dmatrix Constitutive relations mattrix
  virtual Matrix6x6 compute_dmatrix(const Vector6d& stress,
                                    const Vector6d& dstrain,
                                    const ParticleBase<Tdim>* ptr,
                                    mpm::dense_map* state_vars) {
    auto error = Matrix6x6::Zero();
    throw std::runtime_error(
        "Calling the base class function (compute_dmatrix) in "
        "Material:: illegal operation!");
    return error;
  };

};  // MohrCoulomb class
}  // namespace mpm

#include "infinitesimal_elasto_plastic.tcc"

#endif  // MPM_MATERIAL_INFINITESIMAL_ELASTO_PLASTIC_H_

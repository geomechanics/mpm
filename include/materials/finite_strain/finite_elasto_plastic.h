#ifndef MPM_MATERIAL_FINITE_ELASTO_PLASTIC_H_
#define MPM_MATERIAL_FINITE_ELASTO_PLASTIC_H_

#include <cmath>

#include <limits>

#include "Eigen/Dense"

#include "material.h"

// JSON
using Json = nlohmann::json;

namespace mpm {

//! Elasto-plastic material class of finite logarithmic (Hencky) strain
//! \details Base class of all elasto-plastic material model with finite
//! strain
//! \tparam Tdim Dimension
template <unsigned Tdim>
class FiniteElastoPlastic : public Material<Tdim> {
 public:
  //! Define a vector of 3 dof
  using Vector3d = Eigen::Matrix<double, 3, 1>;
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 3 x 3
  using Matrix3x3 = Eigen::Matrix<double, 3, 3>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] material_properties Material properties
  FiniteElastoPlastic(unsigned id, const Json& material_properties)
      : Material<Tdim>(id, material_properties){};

  //! Destructor
  ~FiniteElastoPlastic() override{};

  //! Delete copy constructor
  FiniteElastoPlastic(const FiniteElastoPlastic&) = delete;

  //! Delete assignement operator
  FiniteElastoPlastic& operator=(const FiniteElastoPlastic&) = delete;

  //! Compute stress
  //! \param[in] stress Stress
  //! \param[in] deformation_gradient Deformation gradient at the current step
  //! \param[in] deformation_gradient_increment Deformation gradient increment
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of stress
  Vector6d compute_stress(
      const Vector6d& stress,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient_increment,
      const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) override;

  //! Compute consistent tangent matrix
  //! \param[in] stress Updated stress
  //! \param[in] prev_stress Stress at the current step
  //! \param[in] deformation_gradient Deformation gradient at the current step
  //! \param[in] deformation_gradient_increment Deformation gradient increment
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval dmatrix Constitutive relations mattrix
  Matrix6x6 compute_consistent_tangent_matrix(
      const Vector6d& stress, const Vector6d& prev_stress,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient_increment,
      const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) override;

 protected:
  //! Compute updated kirchhoff stress in principal coordinate system by
  //! performing elasto-plastic return mapping
  //! \param[in, out] principal_elastic_hencky_strain Elastic strain in
  //! principal coordinates
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of kirchhoff stress in principal
  //! coordinate
  virtual Vector3d compute_return_mapping(
      Vector3d& principal_elastic_hencky_strain, const ParticleBase<Tdim>* ptr,
      mpm::dense_map* state_vars) {
    return Vector3d::Zero();
  };

  //! Compute constitutive relations matrix for elasto-plastic material
  //! \param[in] stress Stress
  //! \param[in] elastic_left_cauchy_green Updated elastic_left_cauchy_green
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] hardening Boolean to consider hardening, default=true. If
  //! perfect-plastic tensor is needed pass false
  //! \retval dmatrix Constitutive relations mattrix
  virtual Matrix6x6 compute_elasto_plastic_tensor(
      const Vector6d& stress, const Matrix3x3& elastic_left_cauchy_green,
      const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars,
      bool hardening = true) {
    return Matrix6x6::Zero();
  };

};  // FiniteElastoPlastic class
}  // namespace mpm

#include "finite_elasto_plastic.tcc"

#endif  // MPM_MATERIAL_FINITE_ELASTO_PLASTIC_H_

#ifndef MPM_MATERIAL_INFINITESIMAL_ELASTO_PLASTIC_H_
#define MPM_MATERIAL_INFINITESIMAL_ELASTO_PLASTIC_H_

#include <cmath>

#include <limits>

#include "Eigen/Dense"

#include "linear_elastic.h"

// JSON
using Json = nlohmann::json;

namespace mpm {

//! Elasto-plastic material class of infinitesimal strain
//! \details Base class of all elasto-plastic material model with infinitesimal
//! strain \tparam Tdim Dimension
template <unsigned Tdim>
class InfinitesimalElastoPlastic : public LinearElastic<Tdim> {
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
  InfinitesimalElastoPlastic(unsigned id, const Json& material_properties)
      : LinearElastic<Tdim>(id, material_properties){};

  //! Destructor
  ~InfinitesimalElastoPlastic() override{};

  //! Delete copy constructor
  InfinitesimalElastoPlastic(const InfinitesimalElastoPlastic&) = delete;

  //! Delete assignement operator
  InfinitesimalElastoPlastic& operator=(const InfinitesimalElastoPlastic&) =
      delete;

  //! Compute consistent tangent matrix
  //! \param[in] stress Updated stress
  //! \param[in] prev_stress Stress at the current step
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] dt Time step increment
  //! \retval dmatrix Constitutive relations mattrix
  Matrix6x6 compute_consistent_tangent_matrix(const Vector6d& stress,
                                              const Vector6d& prev_stress,
                                              const Vector6d& dstrain,
                                              const ParticleBase<Tdim>* ptr,
                                              mpm::dense_map* state_vars,
                                              double dt) override;

 protected:
  //! Compute trial stress
  //! \param[in] stress Stress (Voigt)
  //! \param[in] dstrain Strain (Voigt)
  //! \param[in] de Elastic constitutive tensor (Voigt)
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of stress
  virtual Vector6d compute_trial_stress(const Vector6d& stress,
                                        const Vector6d& dstrain,
                                        const Matrix6x6& de,
                                        const ParticleBase<Tdim>* ptr,
                                        mpm::dense_map* state_vars);

  //! Compute constitutive relations matrix for elasto-plastic material
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] dt Time step increment
  //! \param[in] hardening Boolean to consider hardening, default=true. If
  //! perfect-plastic tensor is needed pass false
  //! \retval dmatrix Constitutive relations mattrix
  virtual Matrix6x6 compute_elasto_plastic_tensor(const Vector6d& stress,
                                                  const Vector6d& dstrain,
                                                  const ParticleBase<Tdim>* ptr,
                                                  mpm::dense_map* state_vars,
                                                  double dt,
                                                  bool hardening = true) = 0;

  //! Objective stress rate
  using LinearElastic<Tdim>::stress_rate_;

};  // MohrCoulomb class
}  // namespace mpm

#include "infinitesimal_elasto_plastic.tcc"

#endif  // MPM_MATERIAL_INFINITESIMAL_ELASTO_PLASTIC_H_
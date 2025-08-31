#ifndef MPM_MATERIAL_TERRACOTTA_H_
#define MPM_MATERIAL_TERRACOTTA_H_

#include <limits>

#include "Eigen/Dense"

#include "material.h"

namespace mpm {

//! Terracotta model class
//! \brief A simple hydrodynamic model for clay
//! \details A simple hydrodynamic model for clay
//! \tparam Tdim Dimension
template <unsigned Tdim>
class Terracotta : public Material<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a Matrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] id Material ID
  //! \param[in] material_properties Material properties
  Terracotta(unsigned id, const Json& material_properties);

  //! Destructor
  ~Terracotta() override{};

  //! Delete copy constructor
  Terracotta(const Terracotta&) = delete;

  //! Delete assignement operator
  Terracotta& operator=(const Terracotta&) = delete;

  //! Initialise history variables
  //! \retval state_vars State variables with history
  mpm::dense_map initialise_state_variables() override;

  //! State variables
  std::vector<std::string> state_variables() const override;

  //! Compute stress
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of stress
  Vector6d compute_stress(const Vector6d& stress, const Vector6d& dstrain,
                          const ParticleBase<Tdim>* ptr,
                          mpm::dense_map* state_vars, double dt) override;

 protected:
  //! Function that return macaulay bracket of a double input parameter
  //! \param[in] x Input parameter
  //! \retval macaulay_bracket Maccaulay bracket of x
  inline double macaulay(double x) const { return (x > 0.0) ? x : 0.0; }

  //! Function that return Heaviside of a double input parameter
  //! \param[in] x Input parameter
  //! \retval heaviside Heaviside of x
  inline double heaviside(double x) const { return (x > 0.0) ? 1.0 : 0.0; }

  //! material id
  using Material<Tdim>::id_;
  //! Material properties
  using Material<Tdim>::properties_;
  //! Logger
  using Material<Tdim>::console_;

 private:
  //! Grain density
  double grain_density_{std::numeric_limits<double>::max()};
  //! Bulk modulus
  double bulk_modulus_{std::numeric_limits<double>::max()};
  //! Shear modulus
  double shear_modulus_{std::numeric_limits<double>::max()};
  //! Initial packing fraction
  double initial_packing_fraction_{std::numeric_limits<double>::max()};
  //! Minimum packing fraction
  double minimum_packing_fraction_{std::numeric_limits<double>::max()};
  // Parameter lambda
  double lambda_{std::numeric_limits<double>::max()};
  // Reference pressure
  double p1_{std::numeric_limits<double>::max()};
  // Parameter alpha
  double alpha_{std::numeric_limits<double>::max()};
  // Parameter beta
  double beta_{std::numeric_limits<double>::max()};
  // Parameter gamma
  double gamma_{std::numeric_limits<double>::max()};
  // Parameter eta
  double eta_{std::numeric_limits<double>::max()};
  // Parameter omega
  double omega_{std::numeric_limits<double>::max()};
  // Critical state ratio
  double m_{std::numeric_limits<double>::max()};
  // Initial meso-scale temperature
  double initial_tm_{std::numeric_limits<double>::max()};

  // Parameters for return mapping algorithm
  //! Absolute tolerance
  double abs_tol_{1.e-10};
  //! Relative tolerance
  double rel_tol_{1.e-8};
  //! Maximum number of iterations
  unsigned max_iter_{15};

  //! Discrete tolerance
  double tolerance_{1.0e-4};

};  // Terracotta class
}  // namespace mpm

#include "terracotta.tcc"

#endif  // MPM_MATERIAL_TERRACOTTA_H_

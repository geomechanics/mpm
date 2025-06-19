#ifndef MPM_MATERIAL_PM4SAND_H_
#define MPM_MATERIAL_PM4SAND_H_

#include <cmath>
// #include <string>
// #include <vector>
// #include <map> // Assuming std::map or similar for dense_map
#include <stdexcept>  // For runtime_error

#include "Eigen/Dense"
// #include "json.hpp" // Assuming nlohmann/json

// Assuming these base classes exist in the target MPM framework
#include "infinitesimal_elasto_plastic.h"
// #include "particle_base.h" // Forward declaration or include for ParticleBase

namespace mpm {

//! PM4Sand class
//! \brief PM4Sand material model by Boulanger and Ziotopoulou (2015)
//! \details Implementation based on the provided PM4Sand_claude_2.cpp, adapted
//! for MPM. Only valid for 2D Plane Strain problems. \tparam Tdim Dimension
//! (should be 2 for intended use, but framework might pass 3)
template <unsigned Tdim>  // Keep Tdim template for framework compatibility
class PM4Sand : public InfinitesimalElastoPlastic<Tdim> {
 public:
  //! Define vector and matrix types using Eigen
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;
  using Matrix3d = Eigen::Matrix3d;  // For internal 3D calculations
                                     // (representing 2D plane strain)

  //! Constructor with id and material properties
  PM4Sand(unsigned id, const Json& material_properties);

  //! Destructor
  ~PM4Sand() override = default;

  //! Delete copy constructor
  PM4Sand(const PM4Sand&) = delete;

  //! Delete assignement operator
  PM4Sand& operator=(const PM4Sand&) = delete;

  //! Initialise history variables (state variables) required by the model
  mpm::dense_map initialise_state_variables() override;

  //! Returns a list of state variables used by the material model
  std::vector<std::string> state_variables() const override;

  //! Initialise material properties before analysis (if needed beyond
  //! constructor)
  void initialise(mpm::dense_map* state_vars) override;

  //! Compute stress using the PM4Sand model for a given strain increment
  Vector6d compute_stress(const Vector6d& stress, const Vector6d& dstrain,
                          const ParticleBase<Tdim>* ptr,
                          mpm::dense_map* state_vars, double dt) override;

  //! Compute the consistent tangent matrix (returns elastic tangent for now)
  Matrix6x6 compute_elasto_plastic_tensor(const Vector6d& stress,
                                          const Vector6d& dstrain,
                                          const ParticleBase<Tdim>* ptr,
                                          mpm::dense_map* state_vars, double dt,
                                          bool hardening = true) override;

 protected:
  //! material id (inherited)
  using Material<Tdim>::id_;
  //! Material properties (JSON object, inherited)
  using Material<Tdim>::properties_;
  //! Logger (inherited)
  using Material<Tdim>::console_;

  // Alias for the base class compute_elastic_tensor if available and needed
  // using InfinitesimalElastoPlastic<Tdim>::compute_elastic_tensor;

 private:
  //===========================================================================
  // Private helper functions (declarations) - Implemented below or in .tcc
  //===========================================================================

  //! Initialize internal parameters from JSON properties_
  void initialise_parameters(const Json& material_properties);

  //! Get Ksi state parameter
  double getKsi(double Dr, double p);

  //! Get elastic moduli (K and G) - basic version
  void getElasticModuli(const Matrix3d& sigma, double& K, double& G,
                        mpm::dense_map* state_vars);
  //! Get elastic moduli (K, G, Mcur, zcum) - full version
  void getElasticModuli(const Matrix3d& sigma, double& K, double& G,
                        double& Mcur, const double& zcum,
                        mpm::dense_map* state_vars);  // zcum is input

  //! Get the normal to the yield surface
  Matrix3d getNormalToYield(const Matrix3d& stress, const Matrix3d& alpha,
                            mpm::dense_map* state_vars);

  //! Get the yield function value
  double getF(const Matrix3d& stress, const Matrix3d& alpha,
              mpm::dense_map* state_vars);

  //! Get state dependent parameters (n, D, R, K_p, etc.)
  void getStateDependent(
      const Matrix3d& stress, const Matrix3d& alpha, const Matrix3d& alphaIn,
      const Matrix3d& alphaInP, const Matrix3d& fabric,
      const Matrix3d& fabricIn,
      double& G,  // Input G
      const double& zcum, const double& zpeak,
      const double& pzp,   // Input state scalars
      const double& Mcur,  // Input Mcur (const ref OK here if only read)
      const double& dr,    // Input current Dr
      Matrix3d& n, double& D, Matrix3d& R, double& K_p,       // Outputs
      Matrix3d& alphaD, double& Cka, double& h, Matrix3d& b,  // Outputs
      double& alphaAlphaBDotN, mpm::dense_map* state_vars);   // Output

  //! Stress correction procedure
  void stressCorrection(Matrix3d& nextStress, Matrix3d& nextAlpha,
                        const Matrix3d& stress_prev, const Matrix3d& alphaIn,
                        const Matrix3d& alphaInP, const Matrix3d& curFabric,
                        double& nextVoidRatio, mpm::dense_map* state_vars);

  // Integration helpers (might be part of compute_stress directly)
  // void integrateStep(const Matrix3d& dEpsilon, mpm::dense_map* state_vars);
  // void elasticIntegrator(const Matrix3d& dEpsilon, mpm::dense_map*
  // state_vars);

  //! Update state variables map (internal helper, called by compute_stress)
  // void updateStateVariablesMap(const Matrix3d& stress, const Matrix3d& alpha,
  // ... , mpm::dense_map* state_vars);

  // Tensor <-> Voigt conversions
  Matrix3d voigt_to_tensor(const Vector6d& v) const;
  Vector6d tensor_to_voigt(const Matrix3d& m) const;

  // Tensor operations (defined inline here to avoid redefinition errors)
  inline double tensordot(const Matrix3d& A, const Matrix3d& B) const {
    // Element-wise product and sum
    return (A.array() * B.array()).sum();
  }

  inline double tensorNorm(const Matrix3d& A) const {
    // Frobenius norm: sqrt(sum of squares of elements)
    return A.norm();  // Eigen's built-in norm is Frobenius
    // return std::sqrt((A.array() * A.array()).sum()); // Equivalent manual
    // calculation
  }

  inline Matrix3d getDevPart(const Matrix3d& A) const {
    // Deviatoric part: A - trace(A)/3 * I
    return A - (A.trace() / 3.0) * Matrix3d::Identity();
  }

  inline double getTrace(const Matrix3d& A) const {
    // Trace of the matrix
    return A.trace();
  }

  // Macauley bracket functions
  inline double macauley(double x) const { return (x > 0.0) ? x : 0.0; }
  inline double macauleyIndex(double x) const { return (x > 0.0) ? 1.0 : 0.0; }

  // Helper for Voigt Identity Vector [1, 1, 1, 0, 0, 0]
  Vector6d voigtI() const;

  //===========================================================================
  // Private member variables (Parameters) - Initialized from JSON
  //===========================================================================
  double m_Dr;  // Input Dr
  double m_G0;
  double m_hpo;
  double m_P_atm;
  double phi_cv_deg;
  double m_h0;
  double m_emax;
  double m_emin;
  // double m_e_init; // Calculated and stored in state_vars["void_ratio"]
  // initially
  double m_nb;
  double m_nd;
  double m_Ado;    // Can be calculated based on state if input < 0
  double m_z_max;  // Can be calculated based on state if input < 0
  double m_cz;
  double m_ce;  // Can be calculated based on state if input <= 0
  double m_Mc;  // Calculated from phi_cv
  double m_nu;  // Poisson's ratio
  double m_Cgd;
  double m_Cdr;   // Can be calculated based on state if input < 0
  double m_Ckaf;  // Can be calculated based on state if input < 0
  double m_Q;
  double m_R;
  double m_m;
  double m_Fsed_min;  // Can be calculated based on state if input < 0
  double m_p_sedo;    // Can be calculated based on state if input < 0
  double m_Mb = 0.0;  // Bounding surface stress ratio (初期化しておく)
  double m_Md = 0.0;  // Dilatancy surface stress ratio (初期化しておく)

  // Internal constants / precomputed values
  double m_Pmin;   // Minimum allowable mean effective stress (calculated)
  double m_Pmin2;  // Minimum p for Cpzp2 and Cpmin (calculated)

  // Integration scheme choice (can be set via JSON)
  int m_integrationScheme;  // 1: ModEuler (default), 2: ForwEuler

  // Identity tensor (constant)
  const Matrix3d m_I = Matrix3d::Identity();

  // Small tolerance values
  static constexpr double c_tolerance = 1.0e-9;  // Use static constexpr
  // static constexpr double c_maxStrainInc = 1.0e-5; // Handled by MPM
  // framework timestep?
  static constexpr double c_root12 = M_SQRT1_2;  // Use standard math constants
  static constexpr double c_one3 = 1.0 / 3.0;
  static constexpr double c_two3 = 2.0 / 3.0;
  // Precomputed sqrt(2/3) for constexpr
  static constexpr double c_root23 = 0.816496580927726;  // sqrt(2.0 / 3.0)

  //===========================================================================
  // State variables helpers (defined in .cpp or .tcc)
  //===========================================================================
  // Note: Helper functions for accessing state variables (get_scalar,
  // set_scalar, etc.) are typically implemented as free functions in an
  // anonymous namespace within the .cpp/.tcc file to avoid polluting the class
  // interface.

};  // PM4Sand class

}  // namespace mpm

// Include the implementation file (.tcc) for template definitions
#include "pm4sand.tcc"

#endif  // MPM_MATERIAL_PM4SAND_H_

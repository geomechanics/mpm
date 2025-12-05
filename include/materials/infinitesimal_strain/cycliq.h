#ifndef MPM_MATERIAL_CYCLIQ_H_
#define MPM_MATERIAL_CYCLIQ_H_

#include <cmath>

#include "Eigen/Dense"

#include "infinitesimal_elasto_plastic.h"

namespace mpm {

// namespace cycliq {
// //! Failure state
// enum class FailureState { Elastic = 0, Yield = 1 };
// }  // namespace cycliq

//! Cycliq class
//! \brief Cycliq material model
//! \details Cycliq material model with softening
//! \tparam Tdim Dimension
template <unsigned Tdim>
class Cycliq : public InfinitesimalElastoPlastic<Tdim> {
 public:
  //! Define a vector of 6 dof
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  //! Define a dMatrix of 6 x 6
  using Matrix6x6 = Eigen::Matrix<double, 6, 6>;

  //! Constructor with id and material properties
  //! \param[in] material_properties Material properties
  Cycliq(unsigned id, const Json& material_properties);

  //! Destructor
  ~Cycliq() override = default;

  //! Delete copy constructor
  Cycliq(const Cycliq&) = delete;

  //! Delete assignement operator
  Cycliq& operator=(const Cycliq&) = delete;

  //! Initialise history variables
  //! \retval state_vars State variables with history
  mpm::dense_map initialise_state_variables() override;

  //! State variables
  std::vector<std::string> state_variables() const override;

  //! Initialise material
  //! \brief Function that initialise material to be called at the beginning of
  //! time step
  void initialise(mpm::dense_map* state_vars) override {
    (*state_vars).at("yield_state") = 0;
  };

  //! Compute stress
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \retval updated_stress Updated value of stress
  Vector6d compute_stress(const Vector6d& stress, const Vector6d& dstrain,
                          const ParticleBase<Tdim>* ptr,
                          mpm::dense_map* state_vars, double dt) override;
  // //! Compute consistent tangent matrix
  // //! \param[in] stress Current updated stress
  // //! \param[in] prev_stress Previous stress
  // //! \param[in] dstrain Strain increment
  // //! \param[in] ptr Pointer to particle
  // //! \param[in] state_vars State variables
  // //! \retval dmatrix Consistent tangent operator (6x6)
  // Matrix6x6 compute_consistent_tangent_matrix(
  //     const Vector6d& stress, const Vector6d& prev_stress,
  //     const Vector6d& dstrain, const ParticleBase<Tdim>* ptr,
  //     mpm::dense_map* state_vars) override;
 protected:
  //! material id
  using Material<Tdim>::id_;
  //! Material properties
  using Material<Tdim>::properties_;
  //! Logger
  using Material<Tdim>::console_;

 private:
  //! Compute elastic tensor
  //! \param[in] stress Stress
  //! \param[in] state_vars History-dependent state variables
  Eigen::Matrix<double, 6, 6> compute_elastic_tensor(
      const double p, mpm::dense_map* state_vars);

  //! Compute constitutive relations matrix for elasto-plastic material
  //! \param[in] stress Stress
  //! \param[in] dstrain Strain
  //! \param[in] particle Constant point to particle base
  //! \param[in] state_vars History-dependent state variables
  //! \param[in] hardening Boolean to consider hardening, default=true. If
  //! perfect-plastic tensor is needed pass false
  //! \retval dmatrix Constitutive relations mattrix
  Matrix6x6 compute_elasto_plastic_tensor(const Vector6d& stress,
                                          const Vector6d& dstrain,
                                          const ParticleBase<Tdim>* ptr,
                                          mpm::dense_map* state_vars, double dt,
                                          bool hardening = true) override;
  // Conversion between Voigt and tensor forms
  Eigen::Matrix3d voigt_to_tensor(const Vector6d& v) const;
  Vector6d tensor_to_voigt(const Eigen::Matrix3d& m) const;
  
  inline double tensordot(const Eigen::Matrix3d &A, const Eigen::Matrix3d &B){
    return (A.array()*B.array()).sum();
  }
  inline double tensor_norm(const Eigen::Matrix3d &A){
      return std::sqrt((A.array()*A.array()).sum());
  }
  inline double safe_sin3theta(double J2, double J3){
      double t_sin3theta = (J2 == 0.0) ? 0.0 : -0.5 * J3 * std::pow(3.0/J2,1.5);
      if(t_sin3theta > 1.0) t_sin3theta = 1.0;
      if(t_sin3theta < -1.0) t_sin3theta = -1.0;
      return t_sin3theta;
  }
  inline double yield_function(const Eigen::Matrix3d &r, double M_max, double gtheta){
      double t_norm = tensor_norm(r);
      double val = std::pow(2.0/3.0,0.5)*M_max*gtheta - t_norm;
      return val;
  }
inline Eigen::Matrix3d read_tensor_from_state(const mpm::dense_map& sv, const char* prefix) {
  Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
  auto get = [&](const char* name)->double{ auto it=sv.find(name); return (it==sv.end()?0.0:it->second); };
  if(std::string(prefix)=="r_alpha_"){
    T(0,0)=get("r_alpha_xx"); T(1,1)=get("r_alpha_yy"); T(2,2)=get("r_alpha_zz");
    T(0,1)=T(1,0)=get("r_alpha_xy");
    T(1,2)=T(2,1)=get("r_alpha_yz");
    T(0,2)=T(2,0)=get("r_alpha_zx");
  } else { // Fn_
    T(0,0)=get("Fn_xx"); T(1,1)=get("Fn_yy"); T(2,2)=get("Fn_zz");
    T(0,1)=T(1,0)=get("Fn_xy");
    T(1,2)=T(2,1)=get("Fn_yz");
    T(0,2)=T(2,0)=get("Fn_zx");
  }
  return T;
}
inline void write_tensor_to_state(mpm::dense_map* sv, const char* prefix, const Eigen::Matrix3d& T) {
  auto set=[&](const char* k,double v){ (*sv)[k]=v; };
  if(std::string(prefix)=="r_alpha_"){
    set("r_alpha_xx",T(0,0)); set("r_alpha_yy",T(1,1)); set("r_alpha_zz",T(2,2));
    set("r_alpha_xy",T(0,1)); set("r_alpha_yz",T(1,2)); set("r_alpha_zx",T(0,2));
  } else {
    set("Fn_xx",T(0,0)); set("Fn_yy",T(1,1)); set("Fn_zz",T(2,2));
    set("Fn_xy",T(0,1)); set("Fn_yz",T(1,2)); set("Fn_zx",T(0,2));
  }
}
inline Eigen::Matrix3d deviator(const Eigen::Matrix3d& A){
  return (A - (A.trace()/3.0)*Eigen::Matrix3d::Identity());
}
inline double brent_root_find(
    std::function<double(double)> f,
    double ax,
    double bx,
    double tol       = 1.e-12,
    int    max_iter  = 100
)
{
    // Brent法では、まず f(ax) と f(bx) が異符号であることが前提
    double fa = f(ax);
    double fb = f(bx);
    if (fa * fb > 0.0) {
        throw std::runtime_error("Brent: f(a)*f(b)>0 => no bracket");
    }

    double a = ax;
    double b = bx;
    double c = a;
    double fc = fa;
    double d = 0.0, e = 0.0;

    double eps = std::numeric_limits<double>::epsilon();

    for(int iter=0; iter<max_iter; iter++) {
        if(fb*fc > 0.0) {
            c = a;    // 新しい区間を設定
            fc = fa;
            d = e = b - a;
        }
        if(std::fabs(fc) < std::fabs(fb)) {
            a = b; b = c; c = a;
            double ft = fb; fb = fc; fc = ft;
        }

        double tol_act = 2.0*eps*std::fabs(b) + 0.5*tol;
        double m = 0.5*(c - b);

        // 収束判定
        if(std::fabs(m) <= tol_act || fb == 0.0) {
            return b;
        }

        // 試行ステップの決定
        if(std::fabs(e) < tol_act || std::fabs(fa) <= std::fabs(fb)) {
            // 大幅に収束してきたら or secant法が不安定なら bisection
            d = m;
            e = m;
        } else {
            // secant / inverse quadratic interpolation
            double s = fb/fa;
            double p, q;
            if(a == c) {
                // secant
                p = 2.0*m*s;
                q = 1.0 - s;
            } else {
                // inverse quadratic
                double r = fb/fc;
                p = s*(2.0*m*r*(r + s) - (b - a)*(s - 1.0));
                q = (r - 1.0)*(s - 1.0)*(r + s);
            }
            if(p > 0.0) {
                q = -q;
            } else {
                p = -p;
            }
            // 過大ステップの制限
            if(2.0*p < 3.0*m*q - std::fabs(tol_act*q) && p < std::fabs(0.5*e*q)) {
                e = d;
                d = p/q;
            } else {
                d = m;
                e = m;
            }
        }
        a  = b;
        fa = fb;
        if(std::fabs(d) > tol_act) {
            b += d;
        } else {
            b += (m > 0.0 ? +tol_act : -tol_act);
        }
        fb = f(b);
    }

    // 反復数を超えた => 収束しなかった
    throw std::runtime_error("Brent: no convergence after max_iter");
}


  void init_tensors();
  void set_psi(mpm::dense_map* state_vars);
  std::tuple<double,double> set_GK(double p, mpm::dense_map* state_vars);
  std::tuple<double,double,double>fabric_invariants(const Eigen::Matrix3d &Fn,const Eigen::Matrix3d &n_unit);
  void update_fabric(double dgamma,const Eigen::Matrix3d &n_unit,double D_all,double c_cfab, mpm::dense_map* state_vars);

  double set_p(double elast_vol_strain, mpm::dense_map* state_vars);
  double calculate_gtheta(double M_peak, double M_peako, const Eigen::Matrix3d &dev_str);
  double get_gtheta(const Eigen::Matrix3d& dev_str, mpm::dense_map* state_vars);
  void update_M_max(const Eigen::Matrix3d& r, double gtheta, mpm::dense_map* state_vars);
  double get_yield_func(const Eigen::Matrix3d& r, mpm::dense_map* state_vars);
  std::tuple<Eigen::Matrix3d,double> pegasus_procedure(const Eigen::Matrix3d r_pre,mpm::dense_map* state_vars);
  std::tuple<Eigen::Matrix3d,double> set_r_bar(const Eigen::Matrix3d r_pre,mpm::dense_map* state_vars);
  std::tuple<double, double, double> set_dilatancy(const Eigen::Matrix3d r_pre,Eigen::Matrix3d r_bar,mpm::dense_map* state_vars);
  void init_substep_vars(mpm::dense_map* state_vars);
  void init_next_state(mpm::dense_map* state_vars);
  void update_state(double dstrn_vol, mpm::dense_map* state_vars);
  void update_mainstep(mpm::dense_map* state_vars);
  void set_mainstep_next_Strain(const Vector6d& dStrain, mpm::dense_map* state_vars);
  void set_substep_next_Strain(double dstrn_vol, const Eigen::Matrix3d& dStrn_dev, mpm::dense_map* state_vars);
  // void set_p(double elast_vol_strain, mpm::dense_map* state_vars);
  void add_elast_increment(double dstrn_vol_c, const Eigen::Matrix3d& dStrn_dev, mpm::dense_map* state_vars);
  void set_elast_state(mpm::dense_map* state_vars);
  Eigen::Matrix3d elast_sub(Matrix6x6 De,Vector6d dstrain_m);
  Eigen::Matrix3d calc_substep_CP(double a_dstrn_vol, const Eigen::Matrix3d &a_dStrn_dev, const Eigen::Matrix3d r_pre,const Eigen::Matrix3d Strs_dev_pre,mpm::dense_map* state_vars);
  // void update_M_max(const Eigen::Matrix3d &a_dev_str, double gtheta, mpm::dense_map* state_vars);
  // //! Compute stress invariants (p, q, lode_angle and M_theta)
  // //! \param[in] stress Stress
  // //! \param[in|out] p Mean stress
  // //! \param[in|out] q Deviatoric stress
  // //! \param[in|out] lode_angle Lode angle
  // //! \param[in|out] M_theta Critical state M lode angle
  // void compute_stress_invariants(const Vector6d& stress, double* p, double* q,
  //                                double* lode_angle, double* M_theta);


  //! Maximum reasonable (e_max - e_min) is 0.5 for M_image reduction
  double c_ein_;
  double poisson_ratio_;
  double c_G0_;
  double c_kappa_;
  double c_h_;
  double c_M_;
  double c_dre1_;
  double c_dre2_;
  double c_dir_;
  double c_alpha_;
  double c_gammadr_;
  double c_np_;
  double c_nd_;
  double c_lambdac_;
  double c_e0_;
  double c_xi_;
  double c_pat_;
  double c_D1_;
  double c_D2_;
  double c_cfab_;
  double Fn0_norm_;
  double M_peako_;
  double strn_vol_c_0_;
  // double r_dist_ratio;
  double c_pmin_;
  double c_tolerance_pmin_{1e-12};
  double c_tolerance_yield6_{1e-8};
  double c_tolerance_yield_{1e-7};
  double c_tolerance_detan_{0.05};
  double c_tolerance_dgamma_{0.001};
  const int c_max_iteration_ = 50000;
  const double c_1_3 = 1.0/3.0;
  const Eigen::Matrix3d c_I = Eigen::Matrix3d::Identity();
  double IbunI_[3][3][3][3];
  double IIdev_[3][3][3][3];
//   double G;
//   double K;




//   double cap_{0.5};
//   //! Density
//   double density_{std::numeric_limits<double>::max()};
//   //! Poisson ratio
//   double poisson_ratio_{std::numeric_limits<double>::max()};
//   //! Reference pressure pref
//   double reference_pressure_{std::numeric_limits<double>::max()};
//   //! Critical state friction angle
//   double friction_cs_{std::numeric_limits<double>::max()};
//   //! Critical state coefficient M in triaxial compression
//   double Mtc_{std::numeric_limits<double>::max()};
//   //! Critical state coefficient M in triaxial extension
//   double Mte_{std::numeric_limits<double>::max()};
//   //! Use bolton CSL line
//   bool use_bolton_csl_{false};
//   //! Volumetric coupling (dilatancy) parameter N
//   double N_{std::numeric_limits<double>::max()};
//   //! Minimum void ratio
//   double e_min_{std::numeric_limits<double>::max()};
//   //! Maximum void ratio
//   double e_max_{std::numeric_limits<double>::max()};
//   //! Crushing pressure
//   double crushing_pressure_{std::numeric_limits<double>::max()};
//   //! Lambda volumetric
//   double lambda_{std::numeric_limits<double>::max()};
//   //! Kappa swelling volumetric
//   double kappa_{std::numeric_limits<double>::max()};
//   //! Gamma void ratio at reference pressure
//   double gamma_{std::numeric_limits<double>::max()};
//   //! Dilatancy coefficient
//   double chi_{std::numeric_limits<double>::max()};
//   //! Dilatancy coefficient image
//   double chi_image_{std::numeric_limits<double>::max()};
//   //! Hardening modulus
//   double hardening_modulus_{std::numeric_limits<double>::max()};
//   //! Initial void ratio
//   double void_ratio_initial_{std::numeric_limits<double>::max()};
//   //! Initial image pressure
//   double p_image_initial_{std::numeric_limits<double>::max()};
//   //! Flag for bonded model
//   bool bond_model_{false};
//   //! Initial p_cohesion
//   double p_cohesion_initial_{0.};
//   //! Initial p_dilation
//   double p_dilation_initial_{0.};
//   //! Cohesion degradation parameter m upon shearing
//   double m_cohesion_{0.};
//   //! Dilation degradation parameter m upon shearing
//   double m_dilation_{0.};
//   //! Parameter for modulus
//   double m_modulus_{0.};
//   //! Flag to force stress ratio to converge to critical state
//   bool force_critical_state_{false};
//   //! Default tolerance
//   double tolerance_{std::numeric_limits<double>::epsilon()};
//   //! Failure state map
  // std::map<int, mpm::cycliq::FailureState> yield_type_ = {
  //     {0, mpm::cycliq::FailureState::Elastic},
  //     {1, mpm::cycliq::FailureState::Yield}};

};  // Cycliq class
}  // namespace mpm

#include "cycliq.tcc"

#endif  // MPM_MATERIAL_NORSAND_H_

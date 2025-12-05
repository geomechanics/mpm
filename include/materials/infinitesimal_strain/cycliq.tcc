// cycliq.tcc
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <spdlog/spdlog.h>
#include <sstream>

// 数学定数の事前定義
static const double SQRT_2_3 = std::sqrt(2.0 / 3.0);
static const double SQRT_1_5 = std::sqrt(1.5);
static const double SQRT_2_15 = std::sqrt(2.0 / 15.0);

template <unsigned Tdim>
mpm::Cycliq<Tdim>::Cycliq(unsigned id, const Json& material_properties)
    : InfinitesimalElastoPlastic<Tdim>(id, material_properties) {
  try {
    c_ein_ = material_properties.at("c_ein").template get<double>();
    poisson_ratio_ = material_properties.at("poisson_ratio").template get<double>();
    c_G0_ = material_properties.at("c_G0").template get<double>();
    c_kappa_ = material_properties.at("c_kappa").template get<double>();
    c_h_ = material_properties.at("c_h").template get<double>();
    c_M_ = material_properties.at("c_M").template get<double>();
    c_dre1_ = material_properties.at("c_dre1").template get<double>();
    c_dre2_ = material_properties.at("c_dre2").template get<double>();
    c_dir_ = material_properties.at("c_dir").template get<double>();
    c_alpha_ = material_properties.at("c_alpha").template get<double>();
    c_gammadr_ = material_properties.at("c_gammadr").template get<double>();
    c_np_ = material_properties.at("c_np").template get<double>();
    c_nd_ = material_properties.at("c_nd").template get<double>();
    c_lambdac_ = material_properties.at("c_lambdac").template get<double>();
    c_e0_ = material_properties.at("c_e0").template get<double>();
    c_xi_ = material_properties.at("c_xi").template get<double>();
    c_pat_ = material_properties.at("c_pat").template get<double>();
    // Optional parameters cycliqAni
    c_D1_ = material_properties.contains("c_D1")
                ? material_properties.at("c_D1").template get<double>()
                : 0.0;
    c_D2_ = material_properties.contains("c_D2")
                ? material_properties.at("c_D2").template get<double>()
                : 0.0;
    c_cfab_ = material_properties.contains("c_cfab")
                  ? material_properties.at("c_cfab").template get<double>()
                  : 0.0;
    Fn0_norm_ = material_properties.contains("Fn0_norm")
                    ? material_properties.at("Fn0_norm").template get<double>()
                    : -1.0;

    c_pmin_ = material_properties.at("c_pmin").template get<double>();
    double t_sinphi = 3.0 * c_M_ / (6.0 + c_M_);
    double t_tanphi = t_sinphi / std::sqrt(1.0 - t_sinphi * t_sinphi);
    M_peako_ = 2.0 * std::sqrt(3.0) * t_tanphi /
               std::sqrt(3.0 + 4.0 * t_tanphi * t_tanphi);

    properties_ = material_properties;
  } catch (std::exception& except) {
    console_->error("CycLiq parameter not set: {}\n", except.what());
  }
}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::init_tensors() {
  // メモリクリアと初期化
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
          IbunI_[i][j][k][l] = 0.0;
          IIdev_[i][j][k][l] = 0.0;
        }
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    IbunI_[i][i][0][0] = 1.0;
    IbunI_[i][i][1][1] = 1.0;
    IbunI_[i][i][2][2] = 1.0;
  }

  IIdev_[0][0][0][0] = 2. / 3.;
  IIdev_[0][0][1][1] = -1. / 3.;
  IIdev_[0][0][2][2] = -1. / 3.;
  IIdev_[0][1][0][1] = 0.5;
  IIdev_[0][1][1][0] = 0.5;
  IIdev_[0][2][0][2] = 0.5;
  IIdev_[0][2][2][0] = 0.5;
  IIdev_[1][0][0][1] = 0.5;
  IIdev_[1][0][1][0] = 0.5;
  IIdev_[1][1][0][0] = -1. / 3.;
  IIdev_[1][1][1][1] = 2. / 3.;
  IIdev_[1][1][2][2] = -1. / 3.;
  IIdev_[1][2][1][2] = 0.5;
  IIdev_[1][2][2][1] = 0.5;
  IIdev_[2][0][0][2] = 0.5;
  IIdev_[2][0][2][0] = 0.5;
  IIdev_[2][1][1][2] = 0.5;
  IIdev_[2][1][2][1] = 0.5;
  IIdev_[2][2][0][0] = -1. / 3.;
  IIdev_[2][2][1][1] = -1. / 3.;
  IIdev_[2][2][2][2] = 2. / 3.;
}

template <unsigned Tdim>
mpm::dense_map mpm::Cycliq<Tdim>::initialise_state_variables() {
  dense_map state_vars;
  state_vars["yield_state"] = 0.0;
  state_vars["p_pre"] = 0.0;
  state_vars["psi"] = 0.0;
  state_vars["M_max"] = c_tolerance_pmin_;
  state_vars["gammamono"] = 0.0;
  state_vars["strn_vol_pre"] = 0.0;
  state_vars["strn_vol_ir_pre"] = 0.0;
  state_vars["strn_vol_re_pre"] = 0.0;
  state_vars["strn_vol_c_pre"] = 0.0;
  state_vars["strn_vol_ir_reversal"] = 0.0;

  state_vars["r_alpha_xx"] = 0.0;
  state_vars["r_alpha_yy"] = 0.0;
  state_vars["r_alpha_zz"] = 0.0;
  state_vars["r_alpha_xy"] = 0.0;
  state_vars["r_alpha_yz"] = 0.0;
  state_vars["r_alpha_zx"] = 0.0;
  //   Optional parameters cycliqAni
  state_vars["Fn_xx"] = 0.0;
  state_vars["Fn_yy"] = 0.0;
  state_vars["Fn_zz"] = 0.0;
  state_vars["Fn_xy"] = 0.0;
  state_vars["Fn_yz"] = 0.0;
  state_vars["Fn_zx"] = 0.0;
  return state_vars;
}

template <unsigned Tdim>
std::vector<std::string> mpm::Cycliq<Tdim>::state_variables() const {
  return {
      "yield_state",
      "p_pre",
      "psi",
      "M_max",
      "gammamono",
      "strn_vol_pre",
      "strn_vol_ir_pre",
      "strn_vol_re_pre",
      "strn_vol_c_pre",
      "strn_vol_ir_reversal",
      "r_alpha_xx",
      "r_alpha_yy",
      "r_alpha_zz",
      "r_alpha_xy",
      "r_alpha_yz",
      "r_alpha_zx",
        //   Optional parameters cycliqAni
        "Fn_xx",
        "Fn_yy",
        "Fn_zz",
        "Fn_xy",
        "Fn_yz",
        "Fn_zx",
  };
}

template <unsigned Tdim>
typename mpm::Cycliq<Tdim>::Matrix6x6 mpm::Cycliq<Tdim>::compute_elastic_tensor(
    double p, dense_map* state_vars) {
  Matrix6x6 de = Matrix6x6::Zero();

  double G, K;
  double p_pre = (*state_vars)["p_pre"];
  auto GK = set_GK(p_pre, state_vars);
  G = std::get<0>(GK);
  K = std::get<1>(GK);
  double a1 = K + (4.0 / 3.0) * G;
  double a2 = K - (2.0 / 3.0) * G;

  de(0, 0) = a1;
  de(0, 1) = a2;
  de(0, 2) = a2;
  de(1, 0) = a2;
  de(1, 1) = a1;
  de(1, 2) = a2;
  de(2, 0) = a2;
  de(2, 1) = a2;
  de(2, 2) = a1;
  de(3, 3) = 2 * G;
  de(4, 4) = 2 * G;
  de(5, 5) = 2 * G;

  return de;
}

template <unsigned Tdim>
std::tuple<double, double> mpm::Cycliq<Tdim>::set_GK(
    double p, dense_map* /*state_vars*/) {
  double G, K;
  double term = 2.973 - c_ein_;
  double val = (term * term) / (1. + c_ein_); // pow(x,2) -> x*x
  double sqrt_ppat = std::sqrt(c_pat_ * p);
  G = c_G0_ * val * sqrt_ppat;
  K = (1. + c_ein_) / c_kappa_ * sqrt_ppat;
  return std::make_tuple(G, K);
}

template <unsigned Tdim>
double mpm::Cycliq<Tdim>::set_p(double elast_vol_strain,
                                dense_map* state_vars) {
  double p_pre = (*state_vars)["p_pre"];
  double val = std::sqrt(p_pre / c_pat_) +
               (1.0 + c_ein_) * 0.5 / c_kappa_ * elast_vol_strain;
  double p_now = c_pat_ * (val * val);
  return p_now;
}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::set_psi(dense_map* state_vars) {
  double p_pre = (*state_vars)["p_pre"];
  double strn_vol_pre = (*state_vars)["strn_vol_pre"];
  double t_e_pre = (1.0 + c_ein_) * std::exp(-strn_vol_pre) - 1.0;
  double t_ec = c_e0_ - c_lambdac_ * std::pow(p_pre / c_pat_, c_xi_);
  double psi = t_e_pre - t_ec;
  (*state_vars)["psi"] = psi;
}

template <unsigned Tdim>
double mpm::Cycliq<Tdim>::calculate_gtheta(double M_peak, double M_peako,
                                           const Eigen::Matrix3d& dev_str) {
  double J2 = 0.5 * tensordot(dev_str, dev_str);
  double J3 = dev_str.determinant();
  double s3t = safe_sin3theta(J2, J3);
  double s3t2 = s3t * s3t;
  return 1.0 / (1.0 + M_peak * (s3t + s3t * s3t2) / 6.0 +
                (M_peak / M_peako - 1.0) * (1.0 - s3t2));
}

template <unsigned Tdim>
double mpm::Cycliq<Tdim>::get_gtheta(const Eigen::Matrix3d& a_dev_str,
                                     dense_map* /*state_vars*/) {
  double M_peak = c_M_;
  double val = calculate_gtheta(M_peak, M_peako_, a_dev_str);
  return val;
}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::update_M_max(const Eigen::Matrix3d& a_dev_str,
                                     double gtheta, dense_map* state_vars) {
  double t_eta = std::sqrt(1.5 * tensordot(a_dev_str, a_dev_str));
  double t_M_now = t_eta / gtheta;
  double M_max = (*state_vars)["M_max"];
  double psi = (*state_vars)["psi"];
  if (t_M_now > M_max) M_max = t_M_now;
  if (t_M_now > c_M_ * std::exp(-c_np_ * psi) - c_tolerance_pmin_) {
    M_max = t_M_now;
  }
  (*state_vars)["M_max"] = M_max;
}

template <unsigned Tdim>
double mpm::Cycliq<Tdim>::get_yield_func(const Eigen::Matrix3d& a_dev_str,
                                         dense_map* state_vars) {
  // マップアクセス有りのバージョン（後方互換性のため）
  double t_norm = tensor_norm(a_dev_str);
  if(t_norm < 1.0e-14) return 0.0; 

  Eigen::Matrix3d t_Normal = a_dev_str / t_norm;
  double M_max = (*state_vars)["M_max"];
  double t_gtheta = get_gtheta(t_Normal, state_vars);
  return tensordot(
      SQRT_2_3 * M_max * t_gtheta * t_Normal - a_dev_str,
      t_Normal);
}

template <unsigned Tdim>
std::tuple<double, double, double> mpm::Cycliq<Tdim>::fabric_invariants(
    const Eigen::Matrix3d& Fn, const Eigen::Matrix3d& n_unit) {
  const double Fn_norm = std::sqrt(tensordot(Fn,Fn));
  const double An      = tensordot(Fn,n_unit);
  const double Iani    = std::max(Fn_norm - An, 0.0);
  return std::make_tuple(Fn_norm, An, Iani);
}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::update_fabric(double dgamma, const Eigen::Matrix3d& n_unit,
                                     double D_all, double c_cfab,
                                     dense_map* state_vars) {
    if(dgamma<=0.0 || c_cfab<=0.0) return;
    // これは外部からは呼ばず、calc_substep_CP内でローカル変数処理することを推奨
    // 互換性のため残すが、最適化版はcalc_substep_CP内に実装
    Eigen::Matrix3d Fn = read_tensor_from_state(*state_vars,"Fn_");
    Eigen::Matrix3d dFn = c_cfab*dgamma*(n_unit -(1.0+D_all)*Fn);
    Fn = deviator(Fn + dFn);
    write_tensor_to_state(state_vars,"Fn_",Fn);
}

// =========================================================================
// Pegasus Procedure (Optimized)
// =========================================================================
template <unsigned Tdim>
std::tuple<Eigen::Matrix3d, double> mpm::Cycliq<Tdim>::pegasus_procedure(
    const Eigen::Matrix3d r_pre, dense_map* state_vars) {
  
  // 【最適化】ループに入る前にマップから値をキャッシュ
  // 呼び出し元のcalc_substep_CPでMapが最新に更新されていることを前提とする
  const double M_max = (*state_vars)["M_max"];
  const double psi = (*state_vars)["psi"];
  
  // r_alpha の取り出し（Eigen最適化のためにnoalias等は適宜）
  Eigen::Matrix3d r_alpha;
  r_alpha(0, 0) = (*state_vars)["r_alpha_xx"];
  r_alpha(0, 1) = (*state_vars)["r_alpha_xy"];
  r_alpha(0, 2) = (*state_vars)["r_alpha_zx"];
  r_alpha(1, 0) = (*state_vars)["r_alpha_xy"];
  r_alpha(1, 1) = (*state_vars)["r_alpha_yy"];
  r_alpha(1, 2) = (*state_vars)["r_alpha_yz"];
  r_alpha(2, 0) = (*state_vars)["r_alpha_zx"];
  r_alpha(2, 1) = (*state_vars)["r_alpha_yz"];
  r_alpha(2, 2) = (*state_vars)["r_alpha_zz"];

  // 高速化された Yield Function (Mapアクセスなし)
  auto F = [this, M_max](const Eigen::Matrix3d& rbar) -> double {
      double t_norm = tensor_norm(rbar);
      if(t_norm < 1.0e-14) return 0.0;
      Eigen::Matrix3d t_Normal = rbar / t_norm;
      double t_gtheta = this->get_gtheta(t_Normal, nullptr);
      return tensordot(
        SQRT_2_3 * M_max * t_gtheta * t_Normal - rbar,
        t_Normal);
  };

  Eigen::Matrix3d r_bar = Eigen::Matrix3d::Zero();
  double r_dist_ratio = 0.0;

  double r_pre_dot = tensordot(r_pre, r_pre);
  double r_alpha_dot = tensordot(r_alpha, r_alpha);
  double r_diff_dot = tensordot(r_pre - r_alpha, r_pre - r_alpha);

  // ---------------------------------------------------
  // 1. 特別ケース: r_pre, r_alpha がともに極小
  // ---------------------------------------------------
  if (r_pre_dot < c_tolerance_pmin_ && r_alpha_dot < c_tolerance_pmin_) {
    Eigen::Matrix3d dir; 
    dir << 2, 0, 0, 0, -1, 0, 0, 0, -1;
    r_bar = SQRT_2_15 * c_M_ * std::exp(-c_np_ * psi) * dir;
    double rou_bar = std::sqrt(tensordot(r_bar - r_alpha, r_bar - r_alpha));
    double rou = std::sqrt(r_diff_dot);
    r_dist_ratio = (rou < c_tolerance_pmin_) ? 1.0 : (rou_bar / rou);
    return std::make_tuple(r_bar, r_dist_ratio);
  }

  // ---------------------------------------------------
  // 2. 特別ケース: (r_pre - r_alpha) がほぼゼロ
  // ---------------------------------------------------
  if (std::sqrt(r_diff_dot) < c_tolerance_pmin_) {
      double gtheta = get_gtheta(r_pre, nullptr);
      // ゼロ割り防止
      if(r_pre_dot < c_tolerance_pmin_) r_pre_dot = c_tolerance_pmin_;
      
      r_bar = SQRT_2_3 * M_max * gtheta / r_pre_dot * r_pre;
      double rou_bar = std::sqrt(tensordot(r_bar - r_alpha, r_bar - r_alpha));
      double rou = std::sqrt(r_diff_dot);
      r_dist_ratio = (rou < c_tolerance_pmin_) ? 1.0 : (rou_bar / rou);
      return std::make_tuple(r_bar, r_dist_ratio);
  }

  // ---------------------------------------------------
  // 3. 通常ケース: root finding (Mapアクセスなしでループ)
  // ---------------------------------------------------
  double p_beta0 = 1.0;
  double p_beta1 = 1.0;

  Eigen::Matrix3d p_rbar0 = (1.0 - p_beta0) * r_alpha + p_beta0 * r_pre;
  Eigen::Matrix3d p_rbar1 = (1.0 - p_beta1) * r_alpha + p_beta1 * r_pre;
  double p_Fm0 = F(p_rbar0);
  double p_Fm1 = F(p_rbar1);

  if (std::fabs(p_Fm1) < c_tolerance_yield_) {
    return std::make_tuple(p_rbar1, p_beta1);
  }
  if (std::fabs(p_Fm0) < c_tolerance_yield_) {
    return std::make_tuple(p_rbar0, p_beta0);
  }

  // Bracket finding
  const int bracket_max_iter = 10000;
  int i_PP_while1 = 0;
  while (p_Fm0 * p_Fm1 > 0.0) {
    p_beta0 = p_beta1;
    p_beta1 = 2.0 * p_beta1;

    p_rbar0 = (1.0 - p_beta0) * r_alpha + p_beta0 * r_pre;
    p_rbar1 = (1.0 - p_beta1) * r_alpha + p_beta1 * r_pre;
    p_Fm0 = F(p_rbar0);
    p_Fm1 = F(p_rbar1);

    if (++i_PP_while1 > bracket_max_iter) {
      throw std::runtime_error("PP_while1 no convergence: cannot bracket root.");
    }
  }

  auto eval_F_beta = [&](double beta) {
    Eigen::Matrix3d tmp_rbar = (1.0 - beta) * r_alpha + beta * r_pre;
    return F(tmp_rbar);
  };

  // Brent Solve
  auto brent_solve = [&](double a, double b, double fa, double fb, int max_iter) -> double {
    if (fa * fb > 0.0) throw std::runtime_error("Brent: invalid bracket.");
    double c = a, fc = fa, d = 0.0, e = 0.0;

    for (int iter = 0; iter < max_iter; iter++) {
      if (fb * fc > 0.0) { c = a; fc = fa; d = e = b - a; }
      if (std::fabs(fc) < std::fabs(fb)) {
        std::swap(b, c); std::swap(fb, fc);
      }
      double m = 0.5 * (c - b);
      double tol_act = c_tolerance_pmin_ + 1.e-12 * std::fabs(b);

      if (std::fabs(m) <= tol_act || std::fabs(fb) < c_tolerance_pmin_) return b;

      if (std::fabs(e) < tol_act || std::fabs(fa) <= std::fabs(fb)) {
        d = m; e = m;
      } else {
        double s = fb / fa;
        double p, q;
        if (a == c) {
          p = 2.0 * m * s; q = 1.0 - s;
        } else {
          double r = fb / fc;
          p = s * (2.0 * m * r * (r + s) - (b - a) * (s - 1.0));
          q = (r - 1.0) * (s - 1.0) * (r + s);
        }
        if (p > 0.0) q = -q; else p = -p;
        if (2.0 * p < 3.0 * m * q - std::fabs(tol_act * q) && p < std::fabs(0.5 * e * q)) {
          e = d; d = p / q;
        } else {
          d = m; e = m;
        }
      }
      a = b; fa = fb;
      b += (std::fabs(d) > tol_act) ? d : (m > 0 ? tol_act : -tol_act);
      fb = eval_F_beta(b);
    }
    return b;
  };

  double beta_found = std::numeric_limits<double>::quiet_NaN();
  try {
      beta_found = brent_solve(p_beta0, p_beta1, p_Fm0, p_Fm1, c_max_iteration_);
  } catch (...) {}

  Eigen::Matrix3d tmp_rbar = (1.0 - beta_found) * r_alpha + beta_found * r_pre;
  double valF = F(tmp_rbar);

  if (!std::isnan(beta_found) && std::fabs(valF) < c_tolerance_yield6_ * 100) {
    r_bar = tmp_rbar;
    r_dist_ratio = beta_found;
    if (r_dist_ratio == 1.0) r_bar = r_pre;
    return std::make_tuple(r_bar, r_dist_ratio);
  }

  throw std::runtime_error("Pegasus/bisection procedure: no convergence.");
}

template <unsigned Tdim>
std::tuple<Eigen::Matrix3d, double> mpm::Cycliq<Tdim>::set_r_bar(
    const Eigen::Matrix3d r_pre, dense_map* state_vars) {
  // ラッパー関数: 本質的な処理は最適化された pegasus_procedure で行う
  return pegasus_procedure(r_pre, state_vars);
}

template <unsigned Tdim>
std::tuple<double, double, double> mpm::Cycliq<Tdim>::set_dilatancy(
    const Eigen::Matrix3d r_pre, Eigen::Matrix3d r_bar, dense_map* state_vars) {
  // 後方互換性用: マップアクセス有りの呼び出し
  // 実際にはcalc_substep_CP内でインライン化することを推奨
  // ここでは実装を省略せず元のロジックを維持
  double M_max = (*state_vars)["M_max"];
  double psi = (*state_vars)["psi"];
  double gammamono = (*state_vars)["gammamono"];
  double strn_vol_ir_reversal = (*state_vars)["strn_vol_ir_reversal"];
  double strn_vol_re_pre = (*state_vars)["strn_vol_re_pre"];
  double strn_vol_ir_pre = (*state_vars)["strn_vol_ir_pre"];
  double p_pre = (*state_vars)["p_pre"];
  Eigen::Matrix3d Fn = read_tensor_from_state(*state_vars,"Fn_");

  Eigen::Matrix3d r_d = c_M_ * std::exp(c_nd_ * psi) / M_max * r_bar;
  Eigen::Matrix3d Normal = r_bar * (SQRT_1_5 / tensor_norm(r_bar));

  double dila_re = SQRT_2_3 * c_dre1_ * tensordot(r_d - r_pre, Normal);
  
  double chi = 0.0;
  if (strn_vol_ir_pre > c_tolerance_yield_) {
    chi = -c_dir_ * strn_vol_re_pre / strn_vol_ir_pre;
    chi = std::min(chi, 1.0);
  }

  double dila_ir = 0.0;
  double t1 = c_dir_ * std::exp(c_nd_ * psi - c_alpha_ * strn_vol_ir_pre);
  double t2 = SQRT_2_3 * tensordot(r_d - r_pre, Normal) * std::exp(chi);
  double val_tmp = (c_gammadr_ * (1 - std::exp(c_nd_ * psi)));
  double t3 = val_tmp / (val_tmp + gammamono);
  t3 = t3 * t3;

  if (dila_re > 0.0) {
    double denom = (p_pre > 0) ? p_pre : 1.0;
    dila_re = (c_dre2_ * chi) * (c_dre2_ * chi) / denom;
    if (-strn_vol_re_pre < c_tolerance_pmin_) dila_re = 0.0;
  }

  if (dila_re > 0.0) {
    if (psi > 0.0) dila_ir = t1 * t2;
    else dila_ir = t1 * (t2 + t3);
  } else {
    if (psi > 0.0) dila_ir = 0.0;
    else dila_ir = t1 * t3;
  }
  
  if(dila_ir > 0.0){
    auto [Fn_norm, An, Iani] = fabric_invariants(Fn, Normal);
    double fac = std::max(0.0, 1.0 + c_D2_ * Iani);
    dila_ir *= fac;
  }
  return std::make_tuple(dila_re, dila_ir, dila_re + dila_ir);
}

// =========================================================================
// Main Integration Substep (Heavily Optimized)
// =========================================================================
template <unsigned Tdim>
Eigen::Matrix3d mpm::Cycliq<Tdim>::calc_substep_CP(
    double a_dstrn_vol, const Eigen::Matrix3d& a_dStrn_dev,
    const Eigen::Matrix3d Strs_dev_pre, const Eigen::Matrix3d Strs_dev_now,
    dense_map* state_vars) {

  // 1. Mapからのキャッシュ (Cache into local variables)
  double p_pre = (*state_vars)["p_pre"];
  double strn_vol_pre = (*state_vars)["strn_vol_pre"];
  double strn_vol_c_pre = (*state_vars)["strn_vol_c_pre"];
  double strn_vol_ir_pre = (*state_vars)["strn_vol_ir_pre"];
  double strn_vol_re_pre = (*state_vars)["strn_vol_re_pre"];
  double strn_vol_ir_reversal = (*state_vars)["strn_vol_ir_reversal"];
  double gammamono = (*state_vars)["gammamono"];
  double M_max = (*state_vars)["M_max"];
  double psi = (*state_vars)["psi"];
  
  Eigen::Matrix3d r_alpha;
  r_alpha(0,0) = (*state_vars)["r_alpha_xx"]; r_alpha(0,1) = (*state_vars)["r_alpha_xy"]; r_alpha(0,2) = (*state_vars)["r_alpha_zx"];
  r_alpha(1,0) = (*state_vars)["r_alpha_xy"]; r_alpha(1,1) = (*state_vars)["r_alpha_yy"]; r_alpha(1,2) = (*state_vars)["r_alpha_yz"];
  r_alpha(2,0) = (*state_vars)["r_alpha_zx"]; r_alpha(2,1) = (*state_vars)["r_alpha_yz"]; r_alpha(2,2) = (*state_vars)["r_alpha_zz"];

  Eigen::Matrix3d Fn;
  Fn(0,0) = (*state_vars)["Fn_xx"]; Fn(0,1) = (*state_vars)["Fn_xy"]; Fn(0,2) = (*state_vars)["Fn_zx"];
  Fn(1,0) = (*state_vars)["Fn_xy"]; Fn(1,1) = (*state_vars)["Fn_yy"]; Fn(1,2) = (*state_vars)["Fn_yz"];
  Fn(2,0) = (*state_vars)["Fn_zx"]; Fn(2,1) = (*state_vars)["Fn_yz"]; Fn(2,2) = (*state_vars)["Fn_zz"];

  // Helper lambda: set_p (no map access)
  auto set_p_fast = [&](double vol_strain, double p_pre_val) -> double {
      double val = std::sqrt(p_pre_val / c_pat_) + (1.0 + c_ein_) * 0.5 / c_kappa_ * vol_strain;
      return c_pat_ * (val * val);
  };

  Eigen::Matrix3d r_pre = Strs_dev_pre / p_pre;
  Eigen::Matrix3d Strs_dev_now_ = Strs_dev_now;
  double strn_vol_now = strn_vol_pre + a_dstrn_vol;
  double strn_vol_c_now = strn_vol_c_pre + a_dstrn_vol;
  
  double term_k = -2.0 * c_kappa_ / (1.0 + c_ein_);
  double sqrt_p_pat = std::sqrt(p_pre / c_pat_);
  double sqrt_pmin_pat = std::sqrt(c_pmin_ / c_pat_);
  double strn_vol_c_0_trial = term_k * (sqrt_p_pat - sqrt_pmin_pat);

  double p_now = 0.0;
  if (strn_vol_c_now <= strn_vol_c_0_trial + c_tolerance_pmin_) {
    strn_vol_c_now = strn_vol_c_0_trial;
    p_now = c_pmin_;
  } else {
    // ここでのset_pの引数は元コードでは strn_vol_c_now を elast_vol_strain として渡している
    p_now = set_p_fast(strn_vol_c_now, p_pre);
  }

  double G, K;
  // set_GK inline optimization
  {
      double p_avg = (p_pre + p_now) * 0.5;
      double term = 2.973 - c_ein_;
      double val = (term * term) / (1. + c_ein_);
      double sqrt_ppat_avg = std::sqrt(c_pat_ * p_avg);
      G = c_G0_ * val * sqrt_ppat_avg;
      // K = (1. + c_ein_) / c_kappa_ * sqrt_ppat_avg; // K unused in basic flow
  }

  // Update Psi Locally
  {
      double t_e_pre = (1.0 + c_ein_) * std::exp(-strn_vol_pre) - 1.0;
      double t_ec = c_e0_ - c_lambdac_ * std::pow(p_pre / c_pat_, c_xi_);
      psi = t_e_pre - t_ec;
  }

  // Important: pegasus_procedure reads state_vars. We MUST sync needed vars before call.
  // This is unavoidable without changing pegasus signature or duplicating logic.
  (*state_vars)["p_pre"] = p_pre; // Likely unchanged but safe
  (*state_vars)["psi"] = psi;
  // M_max is not updated yet, so cached value is fine to be in map

  auto r_tuple = pegasus_procedure(r_pre, state_vars);
  Eigen::Matrix3d r_bar = std::get<0>(r_tuple);
  double r_dist_ratio = std::get<1>(r_tuple);
  double r_bar_norm = tensor_norm(r_bar);
  
Eigen::Matrix3d Normal;
if (r_bar_norm > 1e-14) {
Normal = r_bar * (SQRT_1_5 / r_bar_norm);
} else {
Normal.setZero();
}

  Eigen::Matrix3d r_now = Strs_dev_now / p_now;
  double s_rn = tensordot(r_pre, Normal);
  
  double phi = tensordot(Strs_dev_now - Strs_dev_pre, Normal) - (p_now - p_pre) * s_rn;
  double t_phi_n = tensordot(r_now - r_pre, Normal);

  double loadindex = 0.0;
  double plast_modul = 0.0;
  double dila_all = 0.0, dila_re = 0.0, dila_ir = 0.0;
  double strn_vol_ir_now = strn_vol_ir_pre;
  double strn_vol_re_now = strn_vol_re_pre;

  double rou = std::sqrt(tensordot(r_now - r_alpha, r_now - r_alpha));

  // ==========================================================================
  // 1) Elastic Step
  // ==========================================================================
  if (phi < c_tolerance_pmin_ || (t_phi_n < c_tolerance_yield_ && rou > 0.05)) {
    gammamono = 0.0;
    r_alpha = r_pre;
    strn_vol_ir_reversal = strn_vol_ir_pre;
  }
  // ==========================================================================
  // 2) Plastic Step (Loop optimized)
  // ==========================================================================
  else {
    double gtheta = get_gtheta(r_pre, nullptr);
    double exp_np_psi = std::exp(-c_np_ * psi);

    plast_modul = (2.0 / 3.0) * c_h_ * gtheta * G * exp_np_psi *
                  ((c_M_ * exp_np_psi / M_max * r_dist_ratio) - 1.0);

    if (0.0 <= plast_modul && plast_modul < c_tolerance_pmin_) {
      plast_modul = c_tolerance_pmin_;
      auto [Fn_norm, An, Iani] = fabric_invariants(Fn, Normal);
      plast_modul *= std::exp(-c_D1_ * std::sqrt(Iani));
    } else if (plast_modul < 0.0 && plast_modul > -c_tolerance_pmin_) {
      plast_modul = -c_tolerance_pmin_;
    }

    // --- Dilatancy Calculation (Inline Optimization) ---
    {
        Eigen::Matrix3d r_d = c_M_ * std::exp(c_nd_ * psi) / M_max * r_bar;
        dila_re = SQRT_2_3 * c_dre1_ * tensordot(r_d - r_pre, Normal);

        double chi = 0.0;
        if (strn_vol_ir_pre > c_tolerance_yield_) {
            chi = -c_dir_ * strn_vol_re_pre / strn_vol_ir_pre;
            chi = std::min(chi, 1.0);
        }

        double t1 = c_dir_ * std::exp(c_nd_ * psi - c_alpha_ * strn_vol_ir_pre);
        double t2 = SQRT_2_3 * tensordot(r_d - r_pre, Normal) * std::exp(chi);
        double val_tmp = (c_gammadr_ * (1 - std::exp(c_nd_ * psi)));
        double t3 = val_tmp / (val_tmp + gammamono);
        t3 = t3 * t3;

        if (dila_re > 0.0) {
            double denom = (p_pre > 0) ? p_pre : 1.0;
            dila_re = (c_dre2_ * chi) * (c_dre2_ * chi) / denom;
            if (-strn_vol_re_pre < c_tolerance_pmin_) dila_re = 0.0;
        }

        if (dila_re > 0.0) {
            dila_ir = (psi > 0.0) ? (t1 * t2) : (t1 * (t2 + t3));
        } else {
            dila_ir = (psi > 0.0) ? 0.0 : (t1 * t3);
        }
        if(dila_ir > 0.0){
            auto [Fn_norm,An,Iani] = fabric_invariants(Fn, Normal);
            double fac = std::max(0.0, 1.0 + c_D2_ * Iani);
            dila_ir *= fac;
        }
        dila_all = dila_re + dila_ir;
    }
    // ---------------------------------------------------

    double strn_vol_c_0_ = term_k * (sqrt_p_pat - sqrt_pmin_pat);

    // Lambda for Phi evaluation (Purely local vars)
    auto eval_phi = [&](double ld) -> double {
      double s_dstrn_vol_p = ld * dila_all;
      Eigen::Matrix3d s_dStrn_dev_p = ld * Normal;

      double tmp_strn_vol_c_now = strn_vol_c_pre + a_dstrn_vol - s_dstrn_vol_p;
      double tmp_p_now;
      
      if (tmp_strn_vol_c_now <= strn_vol_c_0_ + c_tolerance_pmin_) {
        tmp_p_now = c_pmin_;
      } else {
        tmp_p_now = set_p_fast(tmp_strn_vol_c_now, p_pre);
      }

      double tmp_phi = tensordot(Strs_dev_pre + 2.0 * G * (a_dStrn_dev - s_dStrn_dev_p) - Strs_dev_pre, Normal) -
                       (tmp_p_now - p_pre) * s_rn - ld * plast_modul;
      return tmp_phi;
    };

    // Brent Solver
    auto brent_solve = [&](double x_lower, double x_upper, double tol) {
      double fa = eval_phi(x_lower);
      double fb = eval_phi(x_upper);
      
      // Try to expand bracket if needed
      int expand_count = 0;
      while (fa * fb > 0.0 && expand_count < 10) {
           x_upper *= 5.0; 
           fb = eval_phi(x_upper);
           expand_count++;
      }
      if (fa * fb > 0.0) return std::numeric_limits<double>::quiet_NaN();

      double a = x_lower, b = x_upper;
      double c = a, fc = fa;
      bool mflag = true;
      double d = 0.0;

      for (int iter = 0; iter < 100; ++iter) { // Reduced max iter
        if (std::fabs(b - a) < tol) break;
        double fb_val = eval_phi(b);
        double fa_val = eval_phi(a);

        if ((fa_val > 0 && fb_val > 0) || (fa_val < 0 && fb_val < 0)) {
          a = c; fa_val = fc; d = b - c; c = b; fc = fb_val;
        }

        double s = b;
        if (std::fabs(fc - fa_val) > 1.0e-15) {
          s = a - fa_val * ((b - a) / (fb_val - fa_val));
        }

        if ((!((s > std::min(b, a)) && (s < std::max(b, a)))) ||
            (mflag && (std::fabs(s - b) >= (std::fabs(b - c) * 0.5))) ||
            (!mflag && (std::fabs(s - b) >= (std::fabs(c - d) * 0.5)))) {
          s = 0.5 * (a + b);
          mflag = true;
        } else {
          mflag = false;
        }

        double fs = eval_phi(s);
        d = c; c = b; fc = fb_val;

        if (fa_val * fs < 0.0) b = s;
        else { a = s; fa_val = fs; }

        if (std::fabs(eval_phi(b)) < std::fabs(eval_phi(a))) std::swap(a, b);
      }
      return b;
    };

    loadindex = brent_solve(1.0e-14, 1.0e-2, c_tolerance_pmin_);

    if (!std::isfinite(loadindex) || loadindex > 100.0) {
      // Error recovery: keep Elastic prediction
      r_now = Strs_dev_now_ / p_now; // p_now is elastic estimate
      // Write back partial updates (elastic)
      (*state_vars)["p_pre"] = p_now;
      (*state_vars)["strn_vol_pre"] = strn_vol_now;
      (*state_vars)["strn_vol_ir_pre"] = strn_vol_ir_now;
      (*state_vars)["strn_vol_re_pre"] = strn_vol_re_now;
      (*state_vars)["strn_vol_c_pre"] = strn_vol_c_pre - strn_vol_c_now + a_dstrn_vol;
      (*state_vars)["gammamono"] = gammamono;
      
      update_M_max(r_now, get_gtheta(r_now, nullptr), state_vars);
      return Strs_dev_now_;
    }

    // Apply results
    {
      double s_dstrn_vol_p = loadindex * dila_all;
      Eigen::Matrix3d s_dStrn_dev_p = loadindex * Normal;

      strn_vol_c_now = strn_vol_c_pre + a_dstrn_vol - s_dstrn_vol_p;

      if (strn_vol_c_now <= strn_vol_c_0_ + c_tolerance_pmin_) {
        strn_vol_c_now = strn_vol_c_0_;
        p_now = c_pmin_;
      } else {
        p_now = set_p_fast(strn_vol_c_now, p_pre);
        if (p_now <= c_pmin_) {
          p_now = c_pmin_;
          strn_vol_c_now = strn_vol_c_0_;
        }
      }

      Strs_dev_now_ = Strs_dev_pre + 2.0 * G * (a_dStrn_dev - s_dStrn_dev_p);
      strn_vol_ir_now = strn_vol_ir_pre + loadindex * dila_ir;
      strn_vol_re_now = strn_vol_re_pre + loadindex * dila_re;
    }
    
    gammamono += loadindex;
    
    // Update fabric locally
    if(loadindex > 0.0 && c_cfab_ > 0.0){
         Eigen::Matrix3d dFn = c_cfab_ * loadindex * (Normal - (1.0 + dila_all) * Fn);
         Fn = deviator(Fn + dFn);
    }
  }

  // --- Final Update ---
  r_now = Strs_dev_now_ / p_now;
  double eta = std::sqrt(1.5 * tensordot(r_now, r_now));
  const double eta_cap = c_M_ * std::exp(-c_np_ * psi) / (1.0 + c_M_ / 3.0);
  
  if (eta >= eta_cap - c_tolerance_pmin_) {
    double gtheta = get_gtheta(r_now, nullptr);
    double r_now_norm = std::sqrt(tensordot(r_now, r_now));
    if(r_now_norm > 1.0e-14){
        Eigen::Matrix3d r1_ = r_now / r_now_norm;
        Eigen::Matrix3d r1 = SQRT_2_3 * r1_ * c_M_ * std::exp(-c_np_ * psi) * gtheta;
        double r1_dot = tensordot(r1, r1);
        double r_now_dot = tensordot(r_now, r_now);
        
        if (r1_dot - r_now_dot < c_tolerance_yield6_) {
          double ratio = std::sqrt(r_now_dot) / std::sqrt(r1_dot) + c_tolerance_yield6_;
          Strs_dev_now_ = Strs_dev_now_ / ratio;
          r_now = Strs_dev_now_ / p_now;
        }
    }
  }

  // Write Back to Map (Once per call)
  (*state_vars)["p_pre"] = p_now;
  (*state_vars)["strn_vol_pre"] = strn_vol_now;
  (*state_vars)["strn_vol_ir_pre"] = strn_vol_ir_now;
  (*state_vars)["strn_vol_re_pre"] = strn_vol_re_now;
  (*state_vars)["strn_vol_c_pre"] = strn_vol_c_pre - strn_vol_c_now + a_dstrn_vol - loadindex * dila_all;
  (*state_vars)["gammamono"] = gammamono;
  (*state_vars)["strn_vol_ir_reversal"] = strn_vol_ir_reversal;

  (*state_vars)["r_alpha_xx"] = r_alpha(0,0); (*state_vars)["r_alpha_xy"] = r_alpha(0,1); (*state_vars)["r_alpha_zx"] = r_alpha(0,2);
  (*state_vars)["r_alpha_yy"] = r_alpha(1,1); (*state_vars)["r_alpha_yz"] = r_alpha(1,2); (*state_vars)["r_alpha_zz"] = r_alpha(2,2);
  
  (*state_vars)["Fn_xx"] = Fn(0,0); (*state_vars)["Fn_xy"] = Fn(0,1); (*state_vars)["Fn_zx"] = Fn(0,2);
  (*state_vars)["Fn_yy"] = Fn(1,1); (*state_vars)["Fn_yz"] = Fn(1,2); (*state_vars)["Fn_zz"] = Fn(2,2);

  // set_psi already called locally, just write final value logic if changed again
  // but we need to update M_max
  update_M_max(r_now, get_gtheta(r_now, nullptr), state_vars);

  return Strs_dev_now_;
}

template <unsigned Tdim>
Eigen::Matrix3d mpm::Cycliq<Tdim>::elast_sub(Matrix6x6 De, Vector6d dstrain_m) {
  Eigen::Matrix3d dStress;
  Vector6d dStress_vec = De * dstrain_m;
  dStress << dStress_vec(0), dStress_vec(3), dStress_vec(5), dStress_vec(3),
      dStress_vec(1), dStress_vec(4), dStress_vec(5), dStress_vec(4),
      dStress_vec(2);

  return dStress;
}

template <unsigned Tdim>
typename mpm::Cycliq<Tdim>::Vector6d mpm::Cycliq<Tdim>::compute_stress(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* /*ptr*/, dense_map* state_vars,double dt) {
  
  Vector6d stress_m = -1 * stress / 1000;
  Vector6d dstrain_m = -1 * dstrain;
  dstrain_m(3) = 0.5 * dstrain_m(3);
  dstrain_m(4) = 0.5 * dstrain_m(4);
  dstrain_m(5) = 0.5 * dstrain_m(5);
  double dvol = dstrain_m(0) + dstrain_m(1) + dstrain_m(2);
  double strn_vol_c_now = (*state_vars)["strn_vol_c_pre"] + dvol;
  Eigen::Matrix3d Stress_prev;
  Stress_prev << stress_m(0), stress_m(3), stress_m(5), stress_m(3),
      stress_m(1), stress_m(4), stress_m(5), stress_m(4), stress_m(2);
  double p_pre = (Stress_prev.trace()) / 3.0;

  if ((*state_vars)["p_pre"] == 0.0) {
    (*state_vars)["p_pre"] = p_pre;

    // 異方圧密状態での初期化
    Eigen::Matrix3d Strs_dev_init = Stress_prev - p_pre * Eigen::Matrix3d::Identity();
    double p_denom = (p_pre == 0.0) ? 1.0 : p_pre;
    Eigen::Matrix3d r_init = Strs_dev_init / p_denom;

    double eta_init = std::sqrt(1.5 * tensordot(r_init, r_init));
    (*state_vars)["M_max"] = std::max(eta_init * 1.1, c_M_ * 0.7);

    Eigen::Matrix3d r_alpha_init;
    if (eta_init > 0.3) {
      r_alpha_init = r_init * 0.7;
    } else {
      r_alpha_init = r_init * 0.0;
    }

    (*state_vars)["r_alpha_xx"] = r_alpha_init(0, 0);
    (*state_vars)["r_alpha_yy"] = r_alpha_init(1, 1);
    (*state_vars)["r_alpha_zz"] = r_alpha_init(2, 2);
    (*state_vars)["r_alpha_xy"] = r_alpha_init(0, 1);
    (*state_vars)["r_alpha_yz"] = r_alpha_init(1, 2);
    (*state_vars)["r_alpha_zx"] = r_alpha_init(2, 0);
    
    Eigen::Matrix3d Fn_init = Eigen::Matrix3d::Zero();
    double nrm = std::sqrt(tensordot(r_init, r_init));
    if(nrm > 0.0){
        Fn_init = r_init/nrm;
        double target = (Fn0_norm_ > 0.0 ? Fn0_norm_ : std::min(1.0, eta_init / std::max(c_M_, c_tolerance_pmin_)));
        Fn_init = deviator(Fn_init * target);
    }
    write_tensor_to_state(state_vars, "Fn_", Fn_init);
    set_psi(state_vars);
  }
  
  Eigen::Matrix3d Strs_dev_pre = Stress_prev - p_pre * Eigen::Matrix3d::Identity();
  
  double term_k = -2.0 * c_kappa_ / (1.0 + c_ein_);
  double strn_vol_c_0_ = term_k * (std::sqrt(p_pre / c_pat_) - std::sqrt(c_pmin_ / c_pat_));
  
  double p_now = 0.0;
  if (strn_vol_c_now <= strn_vol_c_0_ + c_tolerance_pmin_) {
    strn_vol_c_now = strn_vol_c_0_;
    p_now = c_pmin_;
  } else {
    p_now = set_p(strn_vol_c_now, state_vars);
  }
  
  double G, K;
  auto GK = set_GK((p_pre + p_now) / 2.0, state_vars);
  G = std::get<0>(GK);

  Eigen::Matrix3d dStrn_dev;
  dStrn_dev << dstrain_m(0) - dvol / 3.0, dstrain_m(3), dstrain_m(5),
      dstrain_m(3), dstrain_m(1) - dvol / 3.0, dstrain_m(4), dstrain_m(5),
      dstrain_m(4), dstrain_m(2) - dvol / 3.0;
      
  Eigen::Matrix3d dev_trial = Strs_dev_pre + dStrn_dev * 2.0 * G;
  Eigen::Matrix3d r_pre = Strs_dev_pre / ((p_pre==0)?1.0:p_pre);
  
  Eigen::Matrix3d t_increment = dev_trial / p_now - r_pre;
  int t_nsub1 = std::ceil(std::sqrt(1.5 * tensordot(t_increment, t_increment)) / c_tolerance_detan_);
  int t_nsub2 = std::ceil(std::sqrt(2.0 / 3.0 * tensordot(dStrn_dev, dStrn_dev)) / c_tolerance_dgamma_);
  int nsub = std::max(1, std::max(t_nsub1, t_nsub2));
  
  if (nsub > 100) nsub = 100;

  Eigen::Matrix3d Strs_dev_now = Strs_dev_pre;
  
  // Substepping loop
  for (int i = 0; i < nsub; i++) {
    Strs_dev_now = calc_substep_CP(dvol / nsub, dStrn_dev / nsub, Strs_dev_pre,
                                   dev_trial, state_vars);
    Strs_dev_pre = Strs_dev_now;
    p_pre = (*state_vars)["p_pre"];
    
    strn_vol_c_0_ = term_k * (std::sqrt(p_pre / c_pat_) - std::sqrt(c_pmin_ / c_pat_));
    strn_vol_c_now = (*state_vars)["strn_vol_c_pre"] + dvol / nsub;
    
    if (strn_vol_c_now <= strn_vol_c_0_ + c_tolerance_pmin_) {
      strn_vol_c_now = strn_vol_c_0_;
      p_now = c_pmin_;
    } else {
      p_now = set_p(strn_vol_c_now, state_vars);
    }
    
    auto GK_sub = set_GK((p_pre + p_now) / 2.0, state_vars);
    G = std::get<0>(GK_sub);
    dev_trial = Strs_dev_now + dStrn_dev / nsub * 2.0 * G;
  }
  
  p_now = (*state_vars)["p_pre"];
  Eigen::Matrix3d updated_stress = Strs_dev_now + p_now * Eigen::Matrix3d::Identity();

  Vector6d updated_stress_vec;
  updated_stress_vec << updated_stress(0, 0), updated_stress(1, 1),
      updated_stress(2, 2), updated_stress(0, 1), updated_stress(1, 2),
      updated_stress(2, 0);

  return -updated_stress_vec * 1000;
}

template <unsigned Tdim>
typename mpm::Cycliq<Tdim>::Matrix6x6
    mpm::Cycliq<Tdim>::compute_elasto_plastic_tensor(
        const Vector6d& stress, const Vector6d& dstrain,
        const ParticleBase<Tdim>* /*ptr*/, dense_map* state_vars, double dt,
        bool /*hardening*/) {
  double p = stress(0) + stress(1) + stress(2);
  Matrix6x6 dep = compute_elastic_tensor(p, state_vars);
  return dep;
}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::add_elast_increment(double dstrn_vol_c,
                                            const Eigen::Matrix3d& dStrn_dev,
                                            dense_map* state_vars) {
  double p_pre = (*state_vars)["p_pre"];
  double strn_vol_c_now = (*state_vars)["strn_vol_c_pre"] + dstrn_vol_c;
  double p_now = set_p(strn_vol_c_now, state_vars);
  (*state_vars)["p_pre"] = p_now;
}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::set_substep_next_Strain(
    double dstrn_vol, const Eigen::Matrix3d& dStrn_dev, dense_map* state_vars) {
  double strn_vol_now = (*state_vars)["strn_vol_pre"] + dstrn_vol;
  (*state_vars)["strn_vol_pre"] = strn_vol_now;
}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::set_mainstep_next_Strain(const Vector6d& dStrain,
                                                 dense_map* state_vars) {
  double dvol = dStrain(0) + dStrain(1) + dStrain(2);
  double strn_vol_now = (*state_vars)["strn_vol_pre"] + dvol;
  (*state_vars)["strn_vol_pre"] = strn_vol_now;
}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::init_substep_vars(dense_map* /*state_vars*/) {}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::init_next_state(dense_map* /*state_vars*/) {}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::update_state(double /*dstrn_vol*/,
                                     dense_map* /*state_vars*/) {}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::update_mainstep(dense_map* /*state_vars*/) {}

template <unsigned Tdim>
void mpm::Cycliq<Tdim>::set_elast_state(dense_map* /*state_vars*/) {}
//! Assign nodal connectivity property for bspline elements
template <unsigned Tdim, unsigned Tpolynomial>
void mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::
    initialise_bspline_connectivity_properties(
        const Eigen::MatrixXd& nodal_coordinates,
        const std::vector<std::vector<unsigned>>& nodal_properties,
        bool kernel_correction, unsigned kc_niteration, double kc_tol) {
  assert(nodal_coordinates.rows() == nodal_properties.size());

  this->nconnectivity_ = nodal_coordinates.rows();
  this->nodal_coordinates_ = nodal_coordinates;
  this->node_type_ = nodal_properties;

  //! Uniform spacing length in 2D
  this->spacing_length_ =
      std::abs(nodal_coordinates(1, 0) - nodal_coordinates(0, 0));
  
  // 【最適化】逆数を事前計算して除算を回避
  if (this->spacing_length_ > std::numeric_limits<double>::epsilon()) {
      this->inv_spacing_length_ = 1.0 / this->spacing_length_;
  } else {
      this->inv_spacing_length_ = 0.0;
  }

  //! Kernel correction boolean and niteration
  this->kernel_correction_ = kernel_correction;
  this->kc_niteration_ = kc_niteration;
  this->kc_tol_ = kc_tol;
}

//! Return shape functions of a Quadrilateral BSpline Element
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::VectorXd
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::shapefn(
        const Eigen::Matrix<double, Tdim, 1>& xi,
        Eigen::Matrix<double, Tdim, 1>& particle_size,
        const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {

  if (this->nconnectivity_ == 4)
    return mpm::QuadrilateralElement<Tdim, 4>::shapefn(xi, particle_size,
                                                       deformation_gradient);

  // Initialize shape function vector
  Eigen::VectorXd shapefn = Eigen::VectorXd::Ones(this->nconnectivity_);

  //! Convert local coordinates to real coordinates
  auto local_shapefn = this->shapefn_local(xi, particle_size, deformation_gradient);
  
  Eigen::Matrix<double, Tdim, 1> pcoord = Eigen::Matrix<double, Tdim, 1>::Zero();
  for (unsigned i = 0; i < 4; ++i) { 
      pcoord.noalias() += local_shapefn(i) * nodal_coordinates_.row(i).transpose();
  }

  if (!kernel_correction_) {
    //! Compute shape function following a multiplicative rule
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      
      // 【最適化】キャッシュの導入 (grad_shapefnと同様のアプローチ)
      // 各次元のカーネル値を計算して保持
      double kernels[Tdim];
      for (unsigned i = 0; i < Tdim; ++i) {
         kernels[i] = this->kernel(pcoord[i], nodal_coordinates_(n, i),
                                   this->node_type_[n][i], Tpolynomial);
         
         // 境界条件(node_type 1, 4)の追加項処理
         unsigned ntype = this->node_type_[n][i];
         if (ntype == 1) {
             kernels[i] += this->kernel(pcoord[i], nodal_coordinates_(n, i), 5, Tpolynomial);
         } else if (ntype == 4) {
             kernels[i] += this->kernel(pcoord[i], nodal_coordinates_(n, i), 6, Tpolynomial);
         }
      }

      // キャッシュした値を乗算
      for (unsigned i = 0; i < Tdim; ++i) {
        shapefn[n] *= kernels[i];
      }
    }
  }
  else {
    // カーネル補正ありの場合
    // (Close-form equation版は元々高速なので、そのままループ構造だけ維持)
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      for (unsigned i = 0; i < Tdim; ++i) {
        double N = this->kernel(pcoord[i], nodal_coordinates_(n, i));
        shapefn[n] *= N;
      }
    }

    if (std::abs(shapefn.sum() - 1.0) > kc_tol_) {
      bool apply_kernel_correction = true;
      unsigned it = 0;
      
      Eigen::Matrix<double, Tdim + 1, Tdim + 1> M;
      Eigen::Matrix<double, Tdim + 1, 1> p;

      while (apply_kernel_correction && it < kc_niteration_) {
        M.setZero();
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          p(0) = 1.0;
          p.template tail<Tdim>() = nodal_coordinates_.row(n).transpose() - pcoord;
          M.noalias() += shapefn(n) * (p * p.transpose());
        }

        const auto M_inv = M.inverse();
        const double C1 = M_inv(0, 0);
        const auto C2 = M_inv.template block<Tdim, 1>(1, 0);

        apply_kernel_correction = false;
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          double dot_prod = C2.dot(nodal_coordinates_.row(n).transpose() - pcoord);
          shapefn(n) *= (C1 + dot_prod);
          
          if (kc_niteration_ > 1 && (shapefn(n) < 0.0)) {
            shapefn(n) = 0.0;
            apply_kernel_correction = true;
          }
        }
        it++;
      }
    }
  }
  return shapefn;
}

//! Return gradient of shape functions
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::MatrixXd
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::grad_shapefn(
        const Eigen::Matrix<double, Tdim, 1>& xi,
        Eigen::Matrix<double, Tdim, 1>& particle_size,
        const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {

  Eigen::MatrixXd grad_shapefn(this->nconnectivity_, Tdim); 
  
  if (this->nconnectivity_ == 4)
    return mpm::QuadrilateralElement<Tdim, 4>::grad_shapefn(
        xi, particle_size, deformation_gradient);

  auto local_shapefn = this->shapefn_local(xi, particle_size, deformation_gradient);
  Eigen::Matrix<double, Tdim, 1> pcoord = Eigen::Matrix<double, Tdim, 1>::Zero();
  for (unsigned i = 0; i < 4; ++i) {
      pcoord.noalias() += local_shapefn(i) * nodal_coordinates_.row(i).transpose();
  }

  if (!kernel_correction_) {
      for (unsigned n = 0; n < this->nconnectivity_; ++n) {
        
        double kernels[Tdim];
        for(unsigned k=0; k<Tdim; ++k) {
            kernels[k] = this->kernel(pcoord[k], nodal_coordinates_(n, k), this->node_type_[n][k], Tpolynomial);
             unsigned ntype = this->node_type_[n][k];
             if (ntype == 1) kernels[k] += this->kernel(pcoord[k], nodal_coordinates_(n, k), 5, Tpolynomial);
             else if (ntype == 4) kernels[k] += this->kernel(pcoord[k], nodal_coordinates_(n, k), 6, Tpolynomial);
        }

        for (unsigned i = 0; i < Tdim; ++i) {
          double dN_dx = this->gradient(pcoord[i], nodal_coordinates_(n, i),
                                        (this->node_type_[n])[i], Tpolynomial);
          unsigned ntype = this->node_type_[n][i];
          if (ntype == 1) dN_dx += this->gradient(pcoord[i], nodal_coordinates_(n, i), 5, Tpolynomial);
          else if (ntype == 4) dN_dx += this->gradient(pcoord[i], nodal_coordinates_(n, i), 6, Tpolynomial);

          for (unsigned j = 0; j < Tdim; ++j) {
            if (j != i) {
              dN_dx *= kernels[j]; 
            }
          }
          grad_shapefn(n, i) = dN_dx;
        }
      }
  } else {
      Eigen::VectorXd shapefn = Eigen::VectorXd::Ones(this->nconnectivity_);
      for (unsigned n = 0; n < this->nconnectivity_; ++n) {
        for (unsigned i = 0; i < Tdim; ++i) {
          shapefn[n] *= this->kernel(pcoord[i], nodal_coordinates_(n, i));
        }
      }

      bool apply_kernel_correction = (std::abs(shapefn.sum() - 1.0) > kc_tol_);

      if (!apply_kernel_correction) {
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          for (unsigned i = 0; i < Tdim; ++i) {
            double dN_dx = this->gradient(pcoord[i], nodal_coordinates_(n, i));
            for (unsigned j = 0; j < Tdim; ++j) {
              if (j != i) {
                 dN_dx *= this->kernel(pcoord[j], nodal_coordinates_(n, j));
              }
            }
            grad_shapefn(n, i) = dN_dx;
          }
        }
      } else {
        Eigen::Matrix<double, Tdim + 1, Tdim + 1> M;
        M.setZero();
        Eigen::Matrix<double, Tdim + 1, 1> p;

        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          p(0) = 1.0;
          // 【修正】template を追加
          p.template tail<Tdim>() = nodal_coordinates_.row(n).transpose() - pcoord;
          M.noalias() += shapefn(n) * (p * p.transpose());
        }
        
        const auto M_inv = M.inverse();
        // 【修正】template を追加
        const auto C2 = M_inv.template block<Tdim, 1>(1, 0);
        // 【修正】template を追加
        const auto C3 = M_inv.template block<Tdim, Tdim>(1, 1);

        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
            Eigen::Matrix<double, Tdim, 1> diff = nodal_coordinates_.row(n).transpose() - pcoord;
            grad_shapefn.row(n) = shapefn(n) * (C2 + C3 * diff).transpose();
        }
      }
    }
  return grad_shapefn;
}

// ... (Other functions like dn_dx, bmatrix usually delegate to above, so optimizations there propagate)

// ==========================================
// 【重要】スカラ演算化したKernel関数群
// ==========================================

// ==========================================
// 2. 反復版 De Boor アルゴリズム (再帰除去・座標変換による高速化)
// ==========================================
//! Compute B-Spline Basis Function (Iterative De Boor)
template <unsigned Tdim, unsigned Tpolynomial>
double mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::kernel(
    double point_coord, double nodal_coord, unsigned node_type,
    unsigned poly_order, unsigned index) const {
    
  // 【超高速化】物理座標を「ノット空間」へ正規化
  // これにより、ループ内での spacing_length_ の乗除算が不要になる
  // u_hat = (x - x_node) / L
  double u_hat = (point_coord - nodal_coord) * inv_spacing_length_;

  const std::vector<double>& kv = this->knot(node_type);

  // 作業用バッファ（スタック確保）
  // Tpolynomial はコンパイル時定数なので固定長配列でOK
  double N[Tpolynomial + 1];

  // 1. Order 0 の計算 (初期化)
  // 基底関数 N_{index, 0}, N_{index+1, 0}, ... を計算
  // 目的は N_{index, poly_order} なので、必要な台(support)の範囲だけ計算
  for (unsigned j = 0; j <= poly_order; ++j) {
      // ノット値はそのまま配列から取得（u_hatと比較）
      double t_curr = kv[index + j];
      double t_next = kv[index + j + 1];

      if (u_hat >= t_curr && u_hat < t_next) {
          N[j] = 1.0;
      } else {
          N[j] = 0.0;
      }
  }

  // 2. Order 1 から poly_order までボトムアップ計算 (反復)
  for (unsigned d = 1; d <= poly_order; ++d) {
      // 三角形の左側へ畳み込み
      // N_{i, d} = term1 * N_{i, d-1} + term2 * N_{i+1, d-1}
      // 配列 N の 0番目から埋めていく
      for (unsigned j = 0; j <= poly_order - d; ++j) {
          double t_left = kv[index + j];
          double t_right = kv[index + j + d + 1];
          
          double t_left_next = kv[index + j + d];
          double t_right_prev = kv[index + j + 1];

          double value = 0.0;

          // 第1項 alpha
          double den1 = t_left_next - t_left;
          if (den1 > std::numeric_limits<double>::epsilon()) {
              value += ((u_hat - t_left) / den1) * N[j];
          }

          // 第2項 beta
          double den2 = t_right - t_right_prev;
          if (den2 > std::numeric_limits<double>::epsilon()) {
              value += ((t_right - u_hat) / den2) * N[j + 1];
          }

          N[j] = value;
      }
  }

  // 最終結果は N[0] に格納されている
  return N[0];
}

//! Compute B-Spline Basis Function (Closed-form) - OPTIMIZED
template <unsigned Tdim, unsigned Tpolynomial>
double mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::kernel(
    double point_coord, double nodal_coord) const {
  
  // 【最適化】除算を乗算に変更
  double xi = std::abs((point_coord - nodal_coord) * inv_spacing_length_);

  double value = 0.0;
  // 分岐予測を考慮するとif-else chainが標準的だが、範囲チェックを先に簡易化
  if (xi < 0.5) {
      value = -xi * xi + 0.75; // 3./4. -> 0.75
  } else if (xi < 1.5) {
      // 0.5 * xi^2 - 1.5 * xi + 1.125
      value = 0.5 * xi * xi - 1.5 * xi + 1.125; 
  }
  return value;
}

//! Compute B-Spline Gradient (Iterative)
template <unsigned Tdim, unsigned Tpolynomial>
double mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::gradient(
    double point_coord, double nodal_coord, unsigned node_type,
    unsigned poly_order, unsigned index) const {

  // B-Splineの微分公式:
  // N'_{i,p}(u) = p / (t_{i+p} - t_i) * N_{i, p-1}(u) - p / (t_{i+p+1} - t_{i+1}) * N_{i+1, p-1}(u)
  // ここで N_{...} は正規化座標系での値。
  // 最後に物理座標系への連鎖律 (d/dx = d/du_hat * du_hat/dx) で inv_spacing_length_ を掛ける

  const std::vector<double>& kv = this->knot(node_type);

  // カーネル計算 (Order = p - 1)
  // 再帰ではなく、先ほどの高速化された kernel 関数を呼び出す（p-1 なのでコストは低い）
  // 必要な2つの項を計算
  
  double term1 = 0.0;
  double den1 = kv[index + poly_order] - kv[index];
  if (std::abs(den1) > std::numeric_limits<double>::epsilon()) {
      term1 = (double)poly_order / den1 * this->kernel(point_coord, nodal_coord, node_type, poly_order - 1, index);
  }

  double term2 = 0.0;
  double den2 = kv[index + poly_order + 1] - kv[index + 1];
  if (std::abs(den2) > std::numeric_limits<double>::epsilon()) {
      term2 = (double)poly_order / den2 * this->kernel(point_coord, nodal_coord, node_type, poly_order - 1, index + 1);
  }

  // dN/du_hat * du_hat/dx
  return (term1 - term2) * inv_spacing_length_;
}
//! Compute B-Spline Gradient (Closed-form) - OPTIMIZED
template <unsigned Tdim, unsigned Tpolynomial>
double mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::gradient(
    double point_coord, double nodal_coord) const {
  
  // 【最適化】除算を乗算に変更
  double xi = (point_coord - nodal_coord) * inv_spacing_length_;
  double signxi = (xi >= 0.) ? 1.0 : -1.0;
  double absxi = std::abs(xi);

  double value = 0.0;
  if (absxi < 0.5) {
    value = -2.0 * xi * inv_spacing_length_;
  } else if (absxi < 1.5) {
    value = (absxi - 1.5) * signxi * inv_spacing_length_;
  }

  return value;
}

//! Compute dn_dx
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::MatrixXd
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::dn_dx(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  // そのまま grad_shapefn に委譲
  return this->grad_shapefn(xi, particle_size, deformation_gradient);
}

//! Compute local dn_dx
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::MatrixXd
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::dn_dx_local(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  
  // grad_shapefn は物理勾配 dN/dx を返す実装になっているため、そのまま返します。
  // 必要であればアサーションでサイズ確認を行います。
  Eigen::MatrixXd dn_dx = this->grad_shapefn(
      xi, particle_size, deformation_gradient);

  assert(dn_dx.rows() == this->nconnectivity_);
  assert(dn_dx.cols() == Tdim);
  
  return dn_dx;
}

//! Return the B-matrix of a Quadrilateral Element
template <unsigned Tdim, unsigned Tpolynomial>
inline std::vector<Eigen::MatrixXd>
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::bmatrix(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {

  // Get gradient shape functions
  Eigen::MatrixXd grad_shapefn =
      this->grad_shapefn(xi, particle_size, deformation_gradient);

  // B-Matrix
  std::vector<Eigen::MatrixXd> bmatrix;
  bmatrix.reserve(this->nconnectivity_);

  // 【最適化】try-catch を削除し、assert に変更
  assert(grad_shapefn.rows() == nodal_coordinates.rows());
  assert(xi.rows() == nodal_coordinates.cols());

  // ループ内での行列確保コストを下げるため、biをループ外で作るか、
  // コンパイラの最適化に任せる形になりますが、ここでは可読性維持のためそのままにします。
  // (std::vector::push_back のコストの方が支配的です)
  for (unsigned i = 0; i < this->nconnectivity_; ++i) {
    Eigen::Matrix<double, 3, Tdim> bi;
    // clang-format off
          bi(0, 0) = grad_shapefn(i, 0); bi(0, 1) = 0.;
          bi(1, 0) = 0.;                 bi(1, 1) = grad_shapefn(i, 1);
          bi(2, 0) = grad_shapefn(i, 1); bi(2, 1) = grad_shapefn(i, 0);
    // clang-format on
    bmatrix.push_back(bi);
  }
  return bmatrix;
}

//! Return local shape functions
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::VectorXd
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::shapefn_local(
        const VectorDim& xi, VectorDim& particle_size,
        const MatrixDim& deformation_gradient) const {
  // 親クラスの処理を呼び出し
  return mpm::QuadrilateralElement<Tdim, 4>::shapefn(xi, particle_size,
                                                     deformation_gradient);
}

//! Compute Jacobian
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::jacobian(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Jacobian dx_i/dxi_j local
  // block処理はEigenのViewを使うためコストは低いです
  return this->jacobian_local(xi, nodal_coordinates.block(0, 0, 4, 2),
                              particle_size, deformation_gradient);
}

//! Compute Jacobian local
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::jacobian_local(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Jacobian dx_i/dxi_j
  return mpm::QuadrilateralElement<2, 4>::jacobian(
      xi, nodal_coordinates, particle_size, deformation_gradient);
}

//! Compute natural coordinates of a point (analytical)
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::Matrix<double, Tdim, 1> mpm::QuadrilateralBSplineElement<
    Tdim, Tpolynomial>::natural_coordinates_analytical(const VectorDim& point,
                                                       const Eigen::MatrixXd&
                                                           nodal_coordinates)
    const {
  // これは例外を投げる仕様のまま維持（呼び出されるべきではないため）
  throw std::runtime_error(
      "Analytical solution for QuadBSpline<Tdim, Tpolynomial> has "
      "not been implemented");
  
  Eigen::Matrix<double, 2, 1> xi;
  xi.fill(std::numeric_limits<double>::max());
  return xi;
}
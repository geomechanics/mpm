//! Assign nodal connectivity property for bspline elements
template <unsigned Tdim, unsigned Tpolynomial>
void mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::
    initialise_bspline_connectivity_properties(
        const Eigen::MatrixXd& nodal_coordinates,
        const std::vector<std::vector<unsigned>>& nodal_properties,
        bool kernel_correction) {
  assert(nodal_coordinates.rows() == nodal_properties.size());

  this->nconnectivity_ = nodal_coordinates.rows();
  this->nodal_coordinates_ = nodal_coordinates;
  this->node_type_ = nodal_properties;

  //! Uniform spacing length in 3D
  this->spacing_length_ =
      std::abs(nodal_coordinates(1, 0) - nodal_coordinates(0, 0));

  //! Identify if element is at boundary to see if kernel correction is
  //! necessary
  if (kernel_correction) {
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      for (unsigned i = 0; i < Tdim; ++i) {
        if (this->node_type_[n][i] == 1 || this->node_type_[n][i] == 4) {
          this->kernel_correction_ = true;
          goto breakout;
        }
      }
    }
  breakout:;
  }
}

//! Return shape functions of a Hexahedron BSpline Element at a given
//! local coordinate
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::VectorXd
    mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::shapefn(
        const Eigen::Matrix<double, Tdim, 1>& xi,
        Eigen::Matrix<double, Tdim, 1>& particle_size,
        const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {

  //! To store shape functions
  Eigen::VectorXd shapefn =
      Eigen::VectorXd::Constant(this->nconnectivity_, 1.0);

  if (this->nconnectivity_ == 8)
    return mpm::HexahedronElement<Tdim, 8>::shapefn(xi, particle_size,
                                                    deformation_gradient);

  try {
    //! Check if we need to apply kernel correction based on particle position
    //! with respective to local element nodes
    bool apply_kernel_correction = false;
    if (kernel_correction_) {
      apply_kernel_correction = this->kernel_correction_region(xi);
    }

    //! Convert local coordinates to real coordinates
    Eigen::Matrix<double, Tdim, 1> pcoord;
    pcoord.setZero();
    auto local_shapefn =
        this->shapefn_local(xi, particle_size, deformation_gradient);
    for (unsigned i = 0; i < local_shapefn.size(); ++i)
      pcoord.noalias() +=
          local_shapefn(i) * nodal_coordinates_.row(i).transpose();

    //! Compute shape function following a multiplicative rule
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      //! Loop over dimension
      for (unsigned i = 0; i < Tdim; ++i) {
        double N = this->kernel(pcoord[i], nodal_coordinates_.row(n)[i],
                                this->node_type_[n][i], Tpolynomial);
        switch (this->node_type_[n][i]) {
          case 1:
            N += this->kernel(pcoord[i], nodal_coordinates_.row(n)[i], 5,
                              Tpolynomial);
            break;
          case 4:
            N += this->kernel(pcoord[i], nodal_coordinates_.row(n)[i], 6,
                              Tpolynomial);
            break;
        }

        shapefn[n] = shapefn[n] * N;
      }
    }

    //! If kernel correction is needed
    if (apply_kernel_correction) {
      // Compute M inverse matrix
      Eigen::Matrix<double, Tdim + 1, Tdim + 1> M =
          Eigen::Matrix<double, Tdim + 1, Tdim + 1>::Zero();
      for (unsigned n = 0; n < this->nconnectivity_; ++n) {
        Eigen::Matrix<double, Tdim + 1, 1> p;
        p << 1.0, (nodal_coordinates_.row(n).transpose() - pcoord);
        M.noalias() += shapefn(n) * p * p.transpose();
      }
      const auto& M_inv = M.inverse();
      const double C1 = M_inv(0, 0);
      const VectorDim& C2 = M_inv.block(1, 0, Tdim, 1);

      // Corrected shape function calculation
      for (unsigned n = 0; n < this->nconnectivity_; ++n) {
        shapefn(n) *=
            (C1 + C2.dot(nodal_coordinates_.row(n).transpose() - pcoord));
      }
    }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return shapefn;
  }
  return shapefn;
}

//! Return gradient of shape functions of a Hexahedron BSpline Element at a
//! given local coordinate
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::MatrixXd
    mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::grad_shapefn(
        const Eigen::Matrix<double, Tdim, 1>& xi,
        Eigen::Matrix<double, Tdim, 1>& particle_size,
        const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {

  //! To store grad shape functions
  Eigen::MatrixXd grad_shapefn(this->nconnectivity_, Tdim);

  if (this->nconnectivity_ == 8)
    return mpm::HexahedronElement<Tdim, 8>::grad_shapefn(xi, particle_size,
                                                         deformation_gradient);

  try {
    //! Check if we need to apply kernel correction based on particle position
    //! with respective to local element nodes
    bool apply_kernel_correction = false;
    if (kernel_correction_) {
      apply_kernel_correction = this->kernel_correction_region(xi);
    }

    //! Convert local coordinates to real coordinates
    Eigen::Matrix<double, Tdim, 1> pcoord;
    pcoord.setZero();
    auto local_shapefn =
        this->shapefn_local(xi, particle_size, deformation_gradient);
    for (unsigned i = 0; i < local_shapefn.size(); ++i)
      pcoord.noalias() +=
          local_shapefn(i) * nodal_coordinates_.row(i).transpose();

    if (!apply_kernel_correction) {
      //! Compute the shape function gradient following a multiplicative rule
      for (unsigned n = 0; n < this->nconnectivity_; ++n)
        //! Loop over dimension
        for (unsigned i = 0; i < Tdim; ++i) {
          double dN_dx = this->gradient(pcoord[i], nodal_coordinates_.row(n)[i],
                                        (this->node_type_[n])[i], Tpolynomial);
          switch (this->node_type_[n][i]) {
            case 1:
              dN_dx += this->gradient(pcoord[i], nodal_coordinates_.row(n)[i],
                                      5, Tpolynomial);
              break;
            case 4:
              dN_dx += this->gradient(pcoord[i], nodal_coordinates_.row(n)[i],
                                      6, Tpolynomial);
              break;
          }

          for (unsigned j = 0; j < Tdim; ++j) {
            if (j != i) {
              double N = this->kernel(pcoord[j], nodal_coordinates_.row(n)[j],
                                      this->node_type_[n][j], Tpolynomial);
              switch (this->node_type_[n][j]) {
                case 1:
                  N += this->kernel(pcoord[j], nodal_coordinates_.row(n)[j], 5,
                                    Tpolynomial);
                  break;
                case 4:
                  N += this->kernel(pcoord[j], nodal_coordinates_.row(n)[j], 6,
                                    Tpolynomial);
                  break;
              }
              dN_dx = dN_dx * N;
            }
          }

          grad_shapefn(n, i) = dN_dx;
        }
    } else {
      Eigen::VectorXd shapefn =
          Eigen::VectorXd::Constant(this->nconnectivity_, 1.0);

      //! Compute shape function following a multiplicative rule
      for (unsigned n = 0; n < this->nconnectivity_; ++n) {
        //! Loop over dimension
        for (unsigned i = 0; i < Tdim; ++i) {
          double N = this->kernel(pcoord[i], nodal_coordinates_.row(n)[i],
                                  this->node_type_[n][i], Tpolynomial);
          switch (this->node_type_[n][i]) {
            case 1:
              N += this->kernel(pcoord[i], nodal_coordinates_.row(n)[i], 5,
                                Tpolynomial);
              break;
            case 4:
              N += this->kernel(pcoord[i], nodal_coordinates_.row(n)[i], 6,
                                Tpolynomial);
              break;
          }

          shapefn[n] = shapefn[n] * N;
        }
      }

      // Compute M inverse matrix
      Eigen::Matrix<double, Tdim + 1, Tdim + 1> M =
          Eigen::Matrix<double, Tdim + 1, Tdim + 1>::Zero();
      for (unsigned n = 0; n < this->nconnectivity_; ++n) {
        Eigen::Matrix<double, Tdim + 1, 1> p;
        p << 1.0, (nodal_coordinates_.row(n).transpose() - pcoord);
        M.noalias() += shapefn(n) * p * p.transpose();
      }
      const auto& M_inv = M.inverse();
      const VectorDim& C2 = M_inv.block(1, 0, Tdim, 1);
      const MatrixDim& C3 = M_inv.block(1, 1, Tdim, Tdim);

      // Corrected shape function gradient calculation
      for (unsigned n = 0; n < this->nconnectivity_; ++n) {
        grad_shapefn.row(n) =
            shapefn(n) *
            (C2 + C3 * (nodal_coordinates_.row(n).transpose() - pcoord))
                .transpose();
      }
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return grad_shapefn;
  }
  return grad_shapefn;
}

//! Return local shape functions of a BSpline Hexahedron Element at a given
//! Return local shape functions of a Hexahedron Element at a given local
//! coordinate, with particle size and deformation gradient
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::VectorXd
    mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::shapefn_local(
        const Eigen::Matrix<double, Tdim, 1>& xi,
        Eigen::Matrix<double, Tdim, 1>& particle_size,
        const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {
  return mpm::HexahedronElement<Tdim, 8>::shapefn(xi, particle_size,
                                                  deformation_gradient);
}

//! Compute Jacobian
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::jacobian(
        const Eigen::Matrix<double, 3, 1>& xi,
        const Eigen::MatrixXd& nodal_coordinates,
        Eigen::Matrix<double, 3, 1>& particle_size,
        const Eigen::Matrix<double, 3, 3>& deformation_gradient) const {
  // Jacobian dx_i/dxi_j local
  return this->jacobian_local(xi, nodal_coordinates.block(0, 0, 8, 3),
                              particle_size, deformation_gradient);
}

//! Compute dn_dx
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::MatrixXd mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::dn_dx(
    const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
    VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  return this->grad_shapefn(xi, particle_size, deformation_gradient);
}

//! Compute local dn_dx
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::MatrixXd
    mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::dn_dx_local(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  Eigen::MatrixXd grad_sf = mpm::HexahedronElement<Tdim, 8>::grad_shapefn(
      xi, particle_size, deformation_gradient);

  // Jacobian dx_i/dxi_j
  Eigen::Matrix<double, Tdim, Tdim> jacobian =
      (grad_sf.transpose() * nodal_coordinates.block(0, 0, 8, Tdim));

  // Gradient shapefn of the cell
  // dN/dx = [J]^-1 * dN/dxi
  const Eigen::MatrixXd dn_dx = grad_sf * (jacobian.inverse()).transpose();

  // Compute dN/dx local
  Eigen::MatrixXd dn_dx_local(this->nconnectivity_, Tdim);
  dn_dx_local.setZero();
  dn_dx_local.block(0, 0, 8, Tdim) = dn_dx;

  return dn_dx_local;
}

//! Compute Jacobian local with particle size and deformation gradient
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::jacobian_local(
        const Eigen::Matrix<double, 3, 1>& xi,
        const Eigen::MatrixXd& nodal_coordinates,
        Eigen::Matrix<double, 3, 1>& particle_size,
        const Eigen::Matrix<double, 3, 3>& deformation_gradient) const {
  // Jacobian dx_i/dxi_j
  return mpm::HexahedronElement<Tdim, 8>::jacobian(
      xi, nodal_coordinates, particle_size, deformation_gradient);
}
//! Compute Bmatrix
template <unsigned Tdim, unsigned Tpolynomial>
inline std::vector<Eigen::MatrixXd>
    mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::bmatrix(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  Eigen::MatrixXd grad_shapefn =
      this->grad_shapefn(xi, particle_size, deformation_gradient);

  // B-Matrix
  std::vector<Eigen::MatrixXd> bmatrix;
  bmatrix.reserve(this->nconnectivity_);

  try {
    // Check if matrices dimensions are correct
    if ((grad_shapefn.rows() != nodal_coordinates.rows()) ||
        (xi.rows() != nodal_coordinates.cols()))
      throw std::runtime_error(
          "BMatrix - Jacobian calculation: Incorrect dimension of xi and "
          "nodal_coordinates");
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return bmatrix;
  }

  for (unsigned i = 0; i < this->nconnectivity_; ++i) {
    // clang-format off
    Eigen::Matrix<double, 6, Tdim> bi;
    bi(0, 0) = grad_shapefn(i, 0); bi(0, 1) = 0.;                 bi(0, 2) = 0.;
    bi(1, 0) = 0.;                 bi(1, 1) = grad_shapefn(i, 1); bi(1, 2) = 0.;
    bi(2, 0) = 0.;                 bi(2, 1) = 0.;                 bi(2, 2) = grad_shapefn(i, 2);
    bi(3, 0) = grad_shapefn(i, 1); bi(3, 1) = grad_shapefn(i, 0); bi(3, 2) = 0.;
    bi(4, 0) = 0.;                 bi(4, 1) = grad_shapefn(i, 2); bi(4, 2) = grad_shapefn(i, 1);
    bi(5, 0) = grad_shapefn(i, 2); bi(5, 1) = 0.;                 bi(5, 2) = grad_shapefn(i, 0);
    // clang-format on
    bmatrix.push_back(bi);
  }
  return bmatrix;
}

//! Compute B-Spline Basis Function using the recursive De Boor's algorithm
//! for single direction
template <unsigned Tdim, unsigned Tpolynomial>
double mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::kernel(
    double point_coord, double nodal_coord, unsigned node_type,
    unsigned poly_order, unsigned index) const {
  double value = 0.0;
  //! Compute knot coordinate
  Eigen::VectorXd knot_vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      (this->knot(node_type)).data(), (this->knot(node_type)).size());
  const Eigen::VectorXd one =
      Eigen::VectorXd::Constant(knot_vector.size(), 1.0);
  const Eigen::VectorXd knot_coord =
      nodal_coord * one + spacing_length_ * knot_vector;

  if (poly_order == 0) {
    if (point_coord >= knot_coord[index] &&
        point_coord < knot_coord[index + 1]) {
      value = 1.0;
    } else {
      value = 0.0;
    }
  } else {
    const double den_a = (knot_coord[index + poly_order] - knot_coord[index]);
    double a = (point_coord - knot_coord(index)) / den_a;
    if (den_a < std::numeric_limits<double>::epsilon()) a = 0;

    const double den_b =
        (knot_coord[index + poly_order + 1] - knot_coord[index + 1]);
    double b = (knot_coord[index + poly_order + 1] - point_coord) / den_b;
    if (den_b < std::numeric_limits<double>::epsilon()) b = 0;

    value = a * this->kernel(point_coord, nodal_coord, node_type,
                             poly_order - 1, index) +
            b * this->kernel(point_coord, nodal_coord, node_type,
                             poly_order - 1, index + 1);
  }
  return value;
}

//! Compute B-Spline Basis Function Gradient using the recursive De Boor's
//! algorithm for single direction
template <unsigned Tdim, unsigned Tpolynomial>
double mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::gradient(
    double point_coord, double nodal_coord, unsigned node_type,
    unsigned poly_order, unsigned index) const {
  double value = 0;
  Eigen::VectorXd knot_vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      (this->knot(node_type)).data(), (this->knot(node_type)).size());
  const Eigen::VectorXd one =
      Eigen::VectorXd::Constant(knot_vector.size(), 1.0);
  const Eigen::VectorXd knot_coord =
      nodal_coord * one + spacing_length_ * knot_vector;

  const double den_a = (knot_coord[index + poly_order] - knot_coord[index]);
  double a = poly_order / den_a;
  if (den_a < std::numeric_limits<double>::epsilon()) a = 0;

  const double den_b =
      (knot_coord[index + poly_order + 1] - knot_coord[index + 1]);
  double b = poly_order / den_b;
  if (den_b < std::numeric_limits<double>::epsilon()) b = 0;

  value = a * this->kernel(point_coord, nodal_coord, node_type, poly_order - 1,
                           index) -
          b * this->kernel(point_coord, nodal_coord, node_type, poly_order - 1,
                           index + 1);

  return value;
}

//! Function to check if particle is lying on the region where kernel
//! correction is necessary
template <unsigned Tdim, unsigned Tpolynomial>
bool mpm::HexahedronBSplineElement<Tdim, Tpolynomial>::kernel_correction_region(
    const VectorDim& xi) const {
  bool status = false;
  // Exit if kernel_correction_ is false
  if (!kernel_correction_) return status;

  // Loop over dimension
  for (unsigned i = 0; i < Tdim; ++i) {
    // First loop over local nodes
    for (unsigned n = 0; n < this->nfunctions_local(); ++n) {
      // If it is located at the lower or uppermost boundary
      if (this->node_type_[n][i] == 1 || this->node_type_[n][i] == 4) {
        status = true;
        goto breakout;
      }
    }

    // Second loop over local nodes
    for (unsigned n = 0; n < this->nfunctions_local(); ++n) {
      // If it is located at the lower intermediate boundary
      if (this->node_type_[n][i] == 2) {
        // Check if particle is located at the negative region
        if (xi[i] < 0.0) {
          status = true;
          goto breakout;
        }
      }
      // If it is located at the lower intermediate boundary
      else if (this->node_type_[n][i] == 3) {
        // Check if particle is located at the positive region
        if (xi[i] > 0.0) {
          status = true;
          goto breakout;
        }
      }
    }
  }

breakout:
  return status;
}
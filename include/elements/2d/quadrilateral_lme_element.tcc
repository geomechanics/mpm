//! Assign nodal connectivity property for LME elements
template <unsigned Tdim>
void mpm::QuadrilateralLMEElement<Tdim>::initialise_lme_connectivity_properties(
    double beta, const Eigen::MatrixXd& nodal_coordinates) {
  assert(nodal_coordinates.rows() == nodal_properties.size());

  this->nconnectivity_ = nodal_coordinates.rows();
  this->nodal_coordinates_ = nodal_coordinates;
  this->beta_ = beta;
  this->lambda_.setZero();
}

//! Return shape functions of a Quadrilateral BSpline Element at a given
//! local coordinate
template <unsigned Tdim>
inline Eigen::VectorXd mpm::QuadrilateralLMEElement<Tdim>::shapefn(
    const Eigen::Matrix<double, Tdim, 1>& xi,
    const Eigen::Matrix<double, Tdim, 1>& particle_size,
    const Eigen::Matrix<double, Tdim, 1>& deformation_gradient) const {

  //! To store shape functions
  Eigen::VectorXd shapefn = Eigen::VectorXd::Constant(nconnectivity_, 1.0);

  if (nconnectivity_ == 4)
    return mpm::QuadrilateralElement<Tdim, 4>::shapefn(xi, particle_size,
                                                       deformation_gradient);

  try {
    //! Convert local coordinates to real coordinates
    Eigen::Matrix<double, Tdim, 1> pcoord;
    pcoord.setZero();
    auto local_shapefn =
        this->shapefn_local(xi, particle_size, deformation_gradient);
    for (unsigned i = 0; i < local_shapefn.size(); ++i)
      pcoord += local_shapefn(i) * nodal_coordinates_.row(i).transpose();

    //! Compute functional f in each connectivity
    Eigen::Matrix<double, nconnectivity_, 1> func_f =
        Eigen::Matrix<double, nconnectivity_, 1>::Zero();
    for (unsigned n = 0; n < nconnectivity_; ++n) {
      const auto& ncoord = nodal_coordinates_.row(n).transpose();
      func_f(n) =
          -beta_ * (pcoord - ncoord).norm() + lambda_.dot(pcoord - ncoord);
    }

    //! Compute p in each connectivity
    Eigen::Matrix<double, nconnectivity_, 1> func_p =
        func_f.exp() / (func_f.exp()).sum();

    //! Compute vector r
    VectorDim r = VectorDim::Zero();
    for (unsigned n = 0; n < nconnectivity_; ++n) {
      const auto& ncoord = nodal_coordinates_.row(n).transpose();
      r += func_p(n) * (pcoord - ncoord);
    }

    //! Begin Newton-Raphson iteration
    const double tolerance = 1.e-12;
    if (r.norm() > tolerance) {
      bool convergence = false;
      unsigned it = 1;
      unsigned max_it = 10;
      while (!convergence) {
        //! Compute matrix J
        Eigen::Matrix<double, Tdim, Tdim> J =
            Eigen::Matrix<double, Tdim, Tdim>::Zero();
        for (unsigned n = 0; n < nconnectivity_; ++n) {
          const auto& ncoord = nodal_coordinates_.row(n).transpose();
          J += func_p(n) * (pcoord - ncoord) * (pcoord - ncoord).transpose();
        }
        J += -r * r.transpose();

        //! Compute Delta Lambda
        VectorDim dlambda = J.inverse() * (-r);
        lambda_ = lambda_ + dlambda;

        //! Reevaluate func_f, func_p, and r
        //! Compute functional func_f in each connectivity
        for (unsigned n = 0; n < nconnectivity_; ++n) {
          const auto& ncoord = nodal_coordinates_.row(n).transpose();
          func_f(n) =
              -beta_ * (pcoord - ncoord).norm() + lambda_.dot(pcoord - ncoord);
        }

        //! Compute func_p in each connectivity
        func_p = func_f.exp() / (func_f.exp()).sum();

        //! Compute vector r
        r.setZero();
        for (unsigned n = 0; n < nconnectivity_; ++n) {
          const auto& ncoord = nodal_coordinates_.row(n).transpose();
          r += func_p(n) * (pcoord - ncoord);
        }

        //! Check convergence
        if (r.norm() < tolerance || it == max_it) convergence = true;
        it++;
      }
    }

    // Assign shape function
    shapefn = func_p;

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return shapefn;
  }
  return shapefn;
}

//! Return gradient of shape functions of a Quadrilateral BSpline Element at a
//! given local coordinate
template <unsigned Tdim>
inline Eigen::MatrixXd mpm::QuadrilateralLMEElement<Tdim>::grad_shapefn(
    const Eigen::Matrix<double, Tdim, 1>& xi,
    const Eigen::Matrix<double, Tdim, 1>& particle_size,
    const Eigen::Matrix<double, Tdim, 1>& deformation_gradient) const {

  //! To store grad shape functions
  Eigen::MatrixXd grad_shapefn(nconnectivity_, Tdim);

  if (nconnectivity_ == 4)
    return mpm::QuadrilateralElement<Tdim, 4>::grad_shapefn(
        xi, particle_size, deformation_gradient);

  try {
    //! Convert local coordinates to real coordinates
    Eigen::Matrix<double, Tdim, 1> pcoord;
    pcoord.setZero();
    auto local_shapefn =
        this->shapefn_local(xi, particle_size, deformation_gradient);
    for (unsigned i = 0; i < local_shapefn.size(); ++i)
      pcoord += local_shapefn(i) * nodal_coordinates_.row(i).transpose();

    //! Compute functional f in each connectivity
    Eigen::Matrix<double, nconnectivity_, 1> func_f =
        Eigen::Matrix<double, nconnectivity_, 1>::Zero();
    for (unsigned n = 0; n < nconnectivity_; ++n) {
      const auto& ncoord = nodal_coordinates_.row(n).transpose();
      func_f(n) =
          -beta_ * (pcoord - ncoord).norm() + lambda_.dot(pcoord - ncoord);
    }

    //! Compute p in each connectivity
    Eigen::Matrix<double, nconnectivity_, 1> func_p =
        func_f.exp() / (func_f.exp()).sum();

    //! Compute vector r
    VectorDim r = VectorDim::Zero();
    for (unsigned n = 0; n < nconnectivity_; ++n) {
      const auto& ncoord = nodal_coordinates_.row(n).transpose();
      r += func_p(n) * (pcoord - ncoord);
    }

    //! Compute matrix J
    Eigen::Matrix<double, Tdim, Tdim> J =
        Eigen::Matrix<double, Tdim, Tdim>::Zero();
    for (unsigned n = 0; n < nconnectivity_; ++n) {
      const auto& ncoord = nodal_coordinates_.row(n).transpose();
      J += func_p(n) * (pcoord - ncoord) * (pcoord - ncoord).transpose();
    }
    J += -r * r.transpose();

    //! Begin Newton-Raphson iteration
    const double tolerance = 1.e-12;
    if (r.norm() > tolerance) {
      bool convergence = false;
      unsigned it = 1;
      unsigned max_it = 5;
      while (!convergence) {
        //! Compute Delta Lambda
        VectorDim dlambda = J.inverse() * (-r);
        lambda_ = lambda_ + dlambda;

        //! Reevaluate func_f, func_p, and r
        //! Compute functional func_f in each connectivity
        for (unsigned n = 0; n < nconnectivity_; ++n) {
          const auto& ncoord = nodal_coordinates_.row(n).transpose();
          func_f(n) =
              -beta_ * (pcoord - ncoord).norm() + lambda_.dot(pcoord - ncoord);
        }

        //! Compute func_p in each connectivity
        func_p = func_f.exp() / (func_f.exp()).sum();

        //! Compute vector r
        r.setZero();
        for (unsigned n = 0; n < nconnectivity_; ++n) {
          const auto& ncoord = nodal_coordinates_.row(n).transpose();
          r += func_p(n) * (pcoord - ncoord);
        }

        //! Compute matrix J
        J.setZero();
        for (unsigned n = 0; n < nconnectivity_; ++n) {
          const auto& ncoord = nodal_coordinates_.row(n).transpose();
          J += func_p(n) * (pcoord - ncoord) * (pcoord - ncoord).transpose();
        }
        J += -r * r.transpose();

        //! Check convergence
        if (r.norm() <= tolerance || it == max_it) convergence = true;
        it++;
      }
    }

    // Compute shape function gradient
    for (unsigned n = 0; n < nconnectivity_; ++n) {
      const auto& ncoord = nodal_coordinates_.row(n).transpose();
      const VectorDim grad_p = -func_p(n) * J.inverse() * (pcoord - ncoord);
      grad_shapefn.row(n) = grad_p.transpose();
    }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return grad_shapefn;
  }
  return grad_shapefn;
}

//! Compute dn_dx
template <unsigned Tdim>
inline Eigen::MatrixXd mpm::QuadrilateralLMEElement<Tdim>::dn_dx(
    const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
    const VectorDim& particle_size,
    const VectorDim& deformation_gradient) const {
  // Get gradient shape functions
  return this->grad_shapefn(xi, particle_size, deformation_gradient);
}

//! Return the B-matrix of a Quadrilateral Element at a given local
//! coordinate for a real cell
template <unsigned Tdim>
inline std::vector<Eigen::MatrixXd> mpm::QuadrilateralLMEElement<Tdim>::bmatrix(
    const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
    const VectorDim& particle_size,
    const VectorDim& deformation_gradient) const {

  // Get gradient shape functions
  Eigen::MatrixXd grad_sf =
      this->grad_shapefn(xi, particle_size, deformation_gradient);

  // B-Matrix
  std::vector<Eigen::MatrixXd> bmatrix;
  bmatrix.reserve(nconnectivity_);

  try {
    // Check if matrices dimensions are correct
    if ((grad_sf.rows() != nodal_coordinates.rows()) ||
        (xi.rows() != nodal_coordinates.cols()))
      throw std::runtime_error(
          "BMatrix - Jacobian calculation: Incorrect dimension of xi and "
          "nodal_coordinates");
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return bmatrix;
  }

  // Jacobian dx_i/dxi_j
  Eigen::Matrix<double, Tdim, Tdim> jacobian =
      (grad_sf.transpose() * nodal_coordinates);

  // Gradient shapefn of the cell
  // dN/dx = [J]^-1 * dN/dxi
  Eigen::MatrixXd grad_shapefn = grad_sf * (jacobian.inverse()).transpose();

  for (unsigned i = 0; i < nconnectivity_; ++i) {
    Eigen::Matrix<double, 3, Tdim> bi;
    // clang-format off
          bi(0, 0) = grad_shapefn(i, 0); bi(0, 1) = 0.;
          bi(1, 0) = 0.;                 bi(1, 1) = grad_shapefn(i, 1);
          bi(2, 0) = grad_shapefn(i, 1); bi(2, 1) = grad_shapefn(i, 0);
          bmatrix.push_back(bi);
    // clang-format on
  }
  return bmatrix;
}

//! Return local shape functions of a BSpline Quadrilateral Element at a given
//! local coordinate, with particle size and deformation gradient
template <unsigned Tdim>
inline Eigen::VectorXd mpm::QuadrilateralLMEElement<Tdim>::shapefn_local(
    const VectorDim& xi, const VectorDim& particle_size,
    const VectorDim& deformation_gradient) const {
  return mpm::QuadrilateralElement<Tdim, 4>::shapefn(xi, particle_size,
                                                     deformation_gradient);
}

//! Compute Jacobian with particle size and deformation gradient
template <unsigned Tdim>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::QuadrilateralLMEElement<Tdim>::jacobian(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        const VectorDim& particle_size,
        const VectorDim& deformation_gradient) const {

  // Get gradient shape functions
  const Eigen::MatrixXd grad_shapefn =
      this->grad_shapefn(xi, particle_size, deformation_gradient);

  try {
    // Check if matrices dimensions are correct
    if ((grad_shapefn.rows() != nodal_coordinates.rows()) ||
        (xi.size() != nodal_coordinates.cols()))
      throw std::runtime_error(
          "Jacobian calculation: Incorrect dimension of xi and "
          "nodal_coordinates");
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return Eigen::Matrix<double, Tdim, Tdim>::Zero();
  }

  // Jacobian dx_i/dxi_j
  return (grad_shapefn.transpose() * nodal_coordinates);
}

//! Compute Jacobian local with particle size and deformation gradient
template <unsigned Tdim>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::QuadrilateralLMEElement<Tdim>::jacobian_local(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        const VectorDim& particle_size,
        const VectorDim& deformation_gradient) const {
  // Jacobian dx_i/dxi_j
  return mpm::QuadrilateralElement<2, 4>::jacobian(
      xi, nodal_coordinates, particle_size, deformation_gradient);
}

//! Compute natural coordinates of a point (analytical)
template <unsigned Tdim>
inline Eigen::Matrix<double, Tdim, 1>
    mpm::QuadrilateralLMEElement<Tdim>::natural_coordinates_analytical(
        const VectorDim& point,
        const Eigen::MatrixXd& nodal_coordinates) const {
  // Local point coordinates
  Eigen::Matrix<double, 2, 1> xi;
  xi.fill(std::numeric_limits<double>::max());
  throw std::runtime_error(
      "Analytical solution for QuadLME<Tdim> has "
      "not been "
      "implemented");
  return xi;
}

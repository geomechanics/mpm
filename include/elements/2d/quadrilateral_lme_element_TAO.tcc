
#include <petscmat.h>
#include <petsctao.h>

// Auxiliar functions to compute the shape functions
typedef struct {
  PetscInt N_a;
  PetscScalar beta;
  PetscScalar R_a;
  PetscScalar * rel_coordinates;
  PetscScalar * distance;
  PetscScalar * p;
} LME_ctx;

//! Evaluates the function and corresponding gradient.
//! \param[in] tao the Tao context
//! \param[in,out] lambda  Lagrange multiplier
//! \param[out] log_Z
//! \param[out] r
//! \param[in] logZ_ctx
//! \retval PetscErrorCode
static PetscErrorCode __function_gradient_log_Z(Tao tao, Vec lambda,
                                                PetscScalar* log_Z, Vec r,
                                                void* logZ_ctx);

//! Evaluates the function hessian
//! \param[in] tao the Tao context
//! \param[in,out] lambda  Lagrange multiplier
//! \param[out] H Hessian of the log(Z) functional
//! \param[out] Hpre Preconditioner of the Hessian
//! \param[in] logZ_ctx
//! \retval PetscErrorCode
static PetscErrorCode __hessian_log_Z(Tao tao, Vec lambda, Mat H, Mat Hpre,
                                      void* logZ_ctx);

//! Assign nodal connectivity property for LME elements
template <unsigned Tdim>
void mpm::QuadrilateralLMEElement<Tdim>::initialise_lme_connectivity_properties(
    double beta, double radius, bool anisotropy,
    const Eigen::MatrixXd& nodal_coordinates) {
  this->nconnectivity_ = nodal_coordinates.rows();
  this->nodal_coordinates_ = nodal_coordinates;
  this->beta_ = beta_;
  this->anisotropy_ = anisotropy;
  this->support_radius_ = radius;
}

//! Return shape functions of a Quadrilateral LME Element at a given
//! local coordinate
template <unsigned Tdim>
inline Eigen::VectorXd mpm::QuadrilateralLMEElement<Tdim>::shapefn(
    const Eigen::Matrix<double, Tdim, 1>& xi,
    Eigen::Matrix<double, Tdim, 1>& lambda,
    const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {

  //! To store shape functions
  Eigen::VectorXd shapefn =
      Eigen::VectorXd::Constant(this->nconnectivity_, 1.0);

  if (this->nconnectivity_ == 4)
    return mpm::QuadrilateralElement<Tdim, 4>::shapefn(xi, lambda,
                                                       deformation_gradient);

  try {
    //! Convert local coordinates to real coordinates
    Eigen::Matrix<double, Tdim, 1> pcoord =
        Eigen::Matrix<double, Tdim, 1>::Zero();
    auto local_shapefn = this->shapefn_local(xi, lambda, deformation_gradient);
    for (unsigned i = 0; i < local_shapefn.size(); ++i)
      pcoord.noalias() +=
          local_shapefn(i) * nodal_coordinates_.row(i).transpose();

    //! Create relative coordinate vector
    Eigen::VectorXd rel_coordinates =
        (-nodal_coordinates_.transpose()).colwise() + pcoord;

    //! Create metric tensor
    Eigen::Matrix<double, Tdim, Tdim> metric =
        Eigen::Matrix<double, Tdim, Tdim>::Identity();

    if (anisotropy_) {
      //! Anisotropic metric tensor
      const auto& inverse_deformation_gradient = deformation_gradient.inverse();
      metric = inverse_deformation_gradient.transpose() *
               inverse_deformation_gradient;
    }

    //! Compute particle-node distance once as a vector
    Eigen::VectorXd distance =
        Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      distance(n) = std::sqrt((rel_coordinates.col(n)).transpose() *
                              (metric * rel_coordinates.col(n)));
    }

    //! Compute functional f in each connectivity
    Eigen::VectorXd f = Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    double sum_exp_f = 0.;
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      if (distance(n) < this->support_radius_) {
        f(n) = -beta_ * distance(n) * distance(n) +
               lambda.dot(rel_coordinates.col(n));
        sum_exp_f += std::exp(f(n));
      }
    }

    //! Compute p in each connectivity
    Eigen::VectorXd p = Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      if (distance(n) < this->support_radius_)
        p(n) = std::exp(f(n)) / sum_exp_f;
    }

    //! Compute vector r
    VectorDim r = VectorDim::Zero();
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      r.noalias() += p(n) * (rel_coordinates.col(n));
    }

    //! Begin Newton-Raphson iteration
    const double tolerance = 1.e-12;
    if (r.norm() > tolerance) {

      /* Definition of some parameters */
      LME_ctx user_ctx;
      PetscInt MaxIter = 10;

      const PetscInt ix[2] = {0, 1};

      /* Create user-defined variable */
      user_ctx.beta = this->beta_;
      user_ctx.distance = distance.data();
      user_ctx.rel_coordinates = rel_coordinates.data(); 
      user_ctx.R_a = this->support_radius_;
      user_ctx.N_a = this->nconnectivity_;
      user_ctx.p = p.data();

      /* Create lagrange multiplier */
      Eigen::Vector2d lambda = Eigen::Vector2d::Zero();
      Vec lambda_aux;
      VecCreate(PETSC_COMM_SELF, &lambda_aux);
      VecSetSizes(lambda_aux, PETSC_DECIDE, 2);
      VecSetFromOptions(lambda_aux);
      VecSetValues(lambda_aux, Tdim, ix, lambda.data(), INSERT_VALUES);
      VecAssemblyBegin(lambda_aux);
      VecAssemblyEnd(lambda_aux);

      /* Create Hessian */
      Mat H;
      MatCreateSeqAIJ(PETSC_COMM_SELF, 2, 2, 2, NULL, &H);
      MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE);
      MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE);
      MatSetFromOptions(H);

      /* Create TAO solver with desired solution method */
      Tao tao;
      TaoCreate(PETSC_COMM_SELF, &tao);
      TaoSetMaximumIterations(tao, MaxIter);
      TaoSetType(tao, TAONTL);
      TaoSetFromOptions(tao);

      /* Set solution vec */
      TaoSetSolution(tao, lambda_aux);

      /* Set routines for function, gradient, hessian evaluation */
      TaoSetObjectiveAndGradient(tao, NULL, __function_gradient_log_Z,
                                 &user_ctx);
      TaoSetHessian(tao, H, H, __hessian_log_Z, &user_ctx);

      /* Solve the system */
      TaoSolve(tao);

      /* Update values of the lagrange multiplier */
      VecGetValues(lambda_aux, 2, ix, lambda.data());

      /* Destroy auxiliar variables */
      TaoDestroy(&tao);
      MatDestroy(&H);
      VecDestroy(&lambda_aux);
    }

    // Assign shape function
    shapefn = p;
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return shapefn;
  }
  return shapefn;
}

//! Return gradient of shape functions of a Quadrilateral LME Element at a
//! given local coordinate
template <unsigned Tdim>
inline Eigen::MatrixXd mpm::QuadrilateralLMEElement<Tdim>::grad_shapefn(
    const Eigen::Matrix<double, Tdim, 1>& xi,
    Eigen::Matrix<double, Tdim, 1>& lambda,
    const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {

  //! To store grad shape functions
  Eigen::MatrixXd grad_shapefn(this->nconnectivity_, Tdim);

  if (this->nconnectivity_ == 4)
    return mpm::QuadrilateralElement<Tdim, 4>::grad_shapefn(
        xi, lambda, deformation_gradient);

  try {
    //! Convert local coordinates to real coordinates
    Eigen::Matrix<double, Tdim, 1> pcoord =
        Eigen::Matrix<double, Tdim, 1>::Zero();
    auto local_shapefn = this->shapefn_local(xi, lambda, deformation_gradient);
    for (unsigned i = 0; i < local_shapefn.size(); ++i)
      pcoord.noalias() +=
          local_shapefn(i) * nodal_coordinates_.row(i).transpose();

    //! Create relative coordinate vector
    Eigen::VectorXd rel_coordinates =
        (-nodal_coordinates_.transpose()).colwise() + pcoord;

    //! Create metric tensor
    Eigen::Matrix<double, Tdim, Tdim> metric =
        Eigen::Matrix<double, Tdim, Tdim>::Identity();

    if (anisotropy_) {
      //! Anisotropic metric tensor
      const auto& inverse_deformation_gradient = deformation_gradient.inverse();
      metric = inverse_deformation_gradient.transpose() *
               inverse_deformation_gradient;
    }

    //! Compute particle-node distance once as a vector
    Eigen::VectorXd distance =
        Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      distance(n) = std::sqrt((rel_coordinates.col(n)).transpose() *
                              (metric * rel_coordinates.col(n)));
    }

    //! Compute functional f in each connectivity
    Eigen::VectorXd f = Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    double sum_exp_f = 0.;
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      if (distance(n) < this->support_radius_) {
        f(n) = -beta_ * distance(n) * distance(n) +
               lambda.dot(rel_coordinates.col(n));
        sum_exp_f += std::exp(f(n));
      }
    }

    //! Compute p in each connectivity
    Eigen::VectorXd p = Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      if (distance(n) < this->support_radius_)
        p(n) = std::exp(f(n)) / sum_exp_f;
    }

    //! Compute vector r
    VectorDim r = VectorDim::Zero();
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      r.noalias() += p(n) * (rel_coordinates.col(n));
    }

    //! Compute matrix J
    Eigen::Matrix2d J = -r * r.transpose();
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      J.noalias() += p(n) * ((rel_coordinates.col(n)) *
                             (rel_coordinates.col(n)).transpose());
    }

    //! Add preconditioner for J (Mathieu Foca, PhD Thesis)
    J.diagonal().array() += r.norm();

    //! Begin Newton-Raphson iteration
    const double tolerance = 1.e-12;
    if (r.norm() > tolerance) {

      /* Definition of some parameters */
      LME_ctx user_ctx;
      PetscInt MaxIter = 10;

      const PetscInt ix[2] = {0, 1};

      /* Create user-defined variable */
      user_ctx.beta = this->beta_;
      user_ctx.distance = distance.data();
      user_ctx.rel_coordinates = rel_coordinates.data(); 
      user_ctx.R_a = this->support_radius_;
      user_ctx.N_a = this->nconnectivity_;
      user_ctx.p = p.data();

      /* Create lagrange multiplier */
      Eigen::Vector2d lambda = Eigen::Vector2d::Zero();
      Vec lambda_aux;
      VecCreate(PETSC_COMM_SELF, &lambda_aux);
      VecSetSizes(lambda_aux, PETSC_DECIDE, 2);
      VecSetFromOptions(lambda_aux);
      VecSetValues(lambda_aux, Tdim, ix, lambda.data(), INSERT_VALUES);
      VecAssemblyBegin(lambda_aux);
      VecAssemblyEnd(lambda_aux);

      /* Create Hessian */
      Mat H;
      MatCreateSeqAIJ(PETSC_COMM_SELF, 2, 2, 2, NULL, &H);
      MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE);
      MatSetOption(H, MAT_SYMMETRIC, PETSC_TRUE);
      MatSetFromOptions(H);

      /* Create TAO solver with desired solution method */
      Tao tao;
      TaoCreate(PETSC_COMM_SELF, &tao);
      TaoSetMaximumIterations(tao, MaxIter);
      TaoSetType(tao, TAONTL);
      TaoSetFromOptions(tao);

      /* Set solution vec */
      TaoSetSolution(tao, lambda_aux);

      /* Set routines for function, gradient, hessian evaluation */
      TaoSetObjectiveAndGradient(tao, NULL, __function_gradient_log_Z,
                                 &user_ctx);
      TaoSetHessian(tao, H, H, __hessian_log_Z, &user_ctx);

      /* Solve the system */
      TaoSolve(tao);

      /* Update values of the lagrange multiplier */
      VecGetValues(lambda_aux, 2, ix, lambda.data());

      /* Destroy auxiliar variables */
      TaoDestroy(&tao);
      MatDestroy(&H);
      VecDestroy(&lambda_aux);
    }

    // Compute shape function gradient
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      const VectorDim grad_p = -p(n) * J.inverse() * (rel_coordinates.col(n));
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
    VectorDim& lambda, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  return this->grad_shapefn(xi, lambda, deformation_gradient);
}

//! Return the B-matrix of a Quadrilateral Element at a given local
//! coordinate for a real cell
template <unsigned Tdim>
inline std::vector<Eigen::MatrixXd> mpm::QuadrilateralLMEElement<Tdim>::bmatrix(
    const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
    VectorDim& lambda, const MatrixDim& deformation_gradient) const {

  // Get gradient shape functions
  Eigen::MatrixXd grad_sf =
      this->grad_shapefn(xi, lambda, deformation_gradient);

  // B-Matrix
  std::vector<Eigen::MatrixXd> bmatrix;
  bmatrix.reserve(this->nconnectivity_);

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

  for (unsigned i = 0; i < this->nconnectivity_; ++i) {
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

//! Return local shape functions of a LME Quadrilateral Element at a given
//! local coordinate, with particle size and deformation gradient
template <unsigned Tdim>
inline Eigen::VectorXd mpm::QuadrilateralLMEElement<Tdim>::shapefn_local(
    const VectorDim& xi, VectorDim& lambda,
    const MatrixDim& deformation_gradient) const {
  return mpm::QuadrilateralElement<Tdim, 4>::shapefn(xi, lambda,
                                                     deformation_gradient);
}

//! Compute Jacobian with particle size and deformation gradient
template <unsigned Tdim>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::QuadrilateralLMEElement<Tdim>::jacobian(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& lambda, const MatrixDim& deformation_gradient) const {

  // Get gradient shape functions
  const Eigen::MatrixXd grad_shapefn =
      this->grad_shapefn(xi, lambda, deformation_gradient);

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
        VectorDim& lambda, const MatrixDim& deformation_gradient) const {
  // Jacobian dx_i/dxi_j
  return mpm::QuadrilateralElement<2, 4>::jacobian(
      xi, nodal_coordinates, lambda, deformation_gradient);
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
      "not been implemented");
  return xi;
}

/*****************************************************/

static PetscErrorCode __function_gradient_log_Z(Tao tao, Vec lambda,
                                                PetscScalar* logZ, 
                                                Vec grad_logZ,
                                                void* logZ_ctx) {

  /* Definition of some parameters */
  PetscErrorCode STATUS = EXIT_SUCCESS;

  /* Get constants */
  PetscInt Ndim = 2;
  PetscScalar beta = ((LME_ctx*)logZ_ctx)->beta;
  PetscScalar R_a = ((LME_ctx*)logZ_ctx)->R_a;
  PetscInt N_a = ((LME_ctx*)logZ_ctx)->N_a;

  /* Read auxiliar variables */
  PetscScalar * p = ((LME_ctx*)logZ_ctx)->p;
  PetscScalar * distance = ((LME_ctx*)logZ_ctx)->distance;
  const PetscScalar* l_a = ((LME_ctx*)logZ_ctx)->rel_coordinates;

  PetscScalar * r_ptr;
  VecGetArray(grad_logZ, &r_ptr);
  PetscScalar * lambda_ptr;
  VecGetArray(lambda, &lambda_ptr);

  //! Reevaluate f, p, and r
  //! Compute functional f in each connectivity
  Eigen::VectorXd f = Eigen::VectorXd::Constant(N_a, 0.0);
  PetscScalar Z = 0.;
  for (unsigned int n = 0; n < N_a; ++n) {
    if (distance[n] < R_a) {
      for (PetscInt i = 0; i < Ndim; i++)
      {      
        f[n] = -beta * distance[n] * distance[n] +
              lambda_ptr[i]*(l_a[n*2 + i]);
        Z += std::exp(f[n]);
      }
    }
  }

  //! Divide by Z and get the final value of the shape function
  PetscScalar Z_m1 = 1.0 / Z;

  //! Compute p in each connectivity
  for (unsigned n = 0; n < N_a; ++n) {
    if (distance[n] < R_a) p[n] = std::exp(f[n]) * Z_m1;
  }

  //! Evaluate objective function 
  *logZ = std::log(Z);

  //! Compute vector r
  for (unsigned n = 0; n < N_a; ++n) {
    for (PetscInt i = 0; i < Ndim; i++) {
      r_ptr[i] += p[n] * l_a[n*2 + i];
    }
  }

  //! Restore auxiliar pointer 
  VecRestoreArray(grad_logZ, &r_ptr);
  VecRestoreArray(lambda, &lambda_ptr);

  return STATUS;
}

/*****************************************************/

static PetscErrorCode __hessian_log_Z(Tao tao, Vec lambda, Mat H, Mat Hpre,
                                      void* logZ_ctx) {

  /* Definition of some parameters */
  PetscErrorCode STATUS = EXIT_SUCCESS;

  /* Get constants */
  PetscInt Ndim = 2;
  PetscInt N_a = ((LME_ctx*)logZ_ctx)->N_a;
  PetscScalar H_a[4] = {0.0, 0.0, 0.0, 0.0};
  const PetscInt idx[2] = {0, 1};

  /* Read auxiliar variables */
  PetscScalar* p_a = ((LME_ctx*)logZ_ctx)->p;
  const PetscScalar* l_a = ((LME_ctx*)logZ_ctx)->rel_coordinates;

  /* Get the value of the gradient of log(Z) */
  Vec r;
  TaoGetGradient(tao, &r, NULL, NULL);
  const PetscScalar* r_ptr;
  VecGetArrayRead(r, &r_ptr);

  /* Fill the Hessian */
  for (PetscInt i = 0; i < Ndim; i++) {
    for (PetscInt j = 0; j < Ndim; j++) {
      /* First component */
      for (PetscInt a = 0; a < N_a; a++) {
        H_a[i * Ndim + j] += p_a[a] * l_a[a * Ndim + i] * l_a[a * Ndim + j];
      }
      /* Second component */
      H_a[i * Ndim + j] += -r_ptr[i] * r_ptr[j];
    }
  }

  MatSetValues(H, Ndim, idx, Ndim, idx, H_a, INSERT_VALUES);

  /* Restore auxiliar pointers */
  VecRestoreArrayRead(r, &r_ptr);

  MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);
  if (H != Hpre) {
    MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY);
  }

  return STATUS;
}

/*****************************************************/
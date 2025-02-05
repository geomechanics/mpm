//! Initialise element thermal matrix
template <unsigned Tdim>
bool mpm::Cell<Tdim>::initialise_element_thermal_matrix() {
  bool status = true;
  if (this->status()) {
    try {
      // Initialse heat capacity matrix N x N
      heat_capacity_matrix_.resize(nnodes_, nnodes_);
      heat_capacity_matrix_.setZero();      
      // Initialse thermal conductivity matrix N x N
      thermal_conductivity_matrix_.resize(nnodes_, nnodes_);
      thermal_conductivity_matrix_.setZero();
      // Initialse thermal expansivity matrix (N * Tdim) x N
      thermal_expansivity_matrix_.resize(nnodes_ * Tdim, nnodes_);
      thermal_expansivity_matrix_.setZero();
    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
      status = false;
    }
  }
  return status;
}

//! Compute local heat capacity stiffness matrix
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_local_heat_capacity_matrix(
    const Eigen::VectorXd& shapefn, const double pvolume, 
    const double multiplier) noexcept {

  std::lock_guard<std::mutex> guard(cell_mutex_);
  // Heat capacity matrix N x N
  // Consistent matrix
  // thermal_conductivity_matrix_ +=
  //     shapefn * (shapefn.transpose()) * pvolume * multiplier;
  // Lumped matrix
  for (unsigned i = 0; i < this->nnodes_; ++i) 
    thermal_conductivity_matrix_(i, i) += shapefn(i) * multiplier * pvolume;
}

//! Compute local thermal conductivity matrix
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_local_thermal_conductivity_matrix(
    const Eigen::MatrixXd& grad_shapefn, double pvolume,
    double multiplier) noexcept {

  std::lock_guard<std::mutex> guard(cell_mutex_);
  // Thermal conductivity matrix N x N
  thermal_conductivity_matrix_ +=
    grad_shapefn * (grad_shapefn.transpose()) * pvolume * multiplier;;
}

//! Compute local thermal expansivity matrix
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_local_thermal_expansivity_matrix(
    const Eigen::VectorXd& shapefn, const Eigen::MatrixXd& bmatrix, 
    const Eigen::MatrixXd& dmatrix,
    const Eigen::VectorXd& identity_vector,
    double pvolume, double multiplier) noexcept {

  std::lock_guard<std::mutex> guard(cell_mutex_);
  // Thermal expansivity matrix (N * Tdim) * N
  // Tdim * (Tdim + 1) / 2 = M, M=1, 3, 6 for 1D, 2D, 3D
  // bmatrix: M * (N * Tdim), bmatrix.transpose(): (N * Tdim) * M
  // dmatrix: M * M
  // identity_vector: M * 1
  // shapefn: N * 1, shapefn.transpose(): 1 * N
  thermal_expansivity_matrix_.noalias() +=
      -bmatrix.transpose() * dmatrix * identity_vector * shapefn.transpose() * 
                                                   multiplier * pvolume;
}


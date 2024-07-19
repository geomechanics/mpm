//! Initialise element stiffness matrix
template <unsigned Tdim>
bool mpm::Cell<Tdim>::initialise_element_stiffness_matrix() {
  bool status = true;
  if (this->status()) {
    try {
      // Initialse stiffness matrix ((N*Tdim)x(N*Tdim))
      stiffness_matrix_.resize(nnodes_ * Tdim, nnodes_ * Tdim);
      stiffness_matrix_.setZero();

    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
      status = false;
    }
  }
  return status;
}

//! Compute local material stiffness matrix
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_local_material_stiffness_matrix(
    const Eigen::MatrixXd& bmatrix, const Eigen::MatrixXd& dmatrix,
    double pvolume, double multiplier) noexcept {

  std::lock_guard<std::mutex> guard(cell_mutex_);
  stiffness_matrix_.noalias() +=
      bmatrix.transpose() * dmatrix * bmatrix * multiplier * pvolume;
}

//! Compute local geometric stiffness matrix
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_local_geometric_stiffness_matrix(
    const Eigen::MatrixXd& geometric_stiffness, double pvolume,
    double multiplier) noexcept {

  std::lock_guard<std::mutex> guard(cell_mutex_);
  stiffness_matrix_.noalias() += geometric_stiffness * multiplier * pvolume;
}

//! Compute local mass matrix
template <unsigned Tdim>
inline void mpm::Cell<Tdim>::compute_local_mass_matrix(
    const Eigen::VectorXd& shapefn, double pvolume,
    double multiplier) noexcept {

  std::lock_guard<std::mutex> guard(cell_mutex_);
  for (unsigned i = 0; i < this->nnodes_; ++i) {
    for (unsigned k = 0; k < Tdim; ++k) {
      stiffness_matrix_(Tdim * i + k, Tdim * i + k) +=
          shapefn(i) * multiplier * pvolume;
    }
  }
}

//! Compute local stiffness matrix (Used in equilibrium equation)
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_local_stiffness_matrix_block(
    unsigned row_start, unsigned col_start, const Eigen::MatrixXd& stiffness,
    double pvolume, double multiplier) noexcept {

  std::lock_guard<std::mutex> guard(cell_mutex_);
  stiffness_matrix_
      .block(row_start, col_start, stiffness.rows(), stiffness.cols())
      .noalias() += stiffness * multiplier * pvolume;
}
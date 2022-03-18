//! Function to check convergence
inline bool mpm::ConvergenceCriterionResidual::check_convergence(
    const Eigen::VectorXd& residual_vector, bool initial) {
  bool convergence = false;
  try {
    // Check mpi rank and size
    int mpi_rank = 0;
    int mpi_size = 1;

#ifdef USE_MPI
    // Get MPI rank
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    // Get number of MPI ranks
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

    // Residual norm
    double residual_norm;

    // Perform norm computation: using PETSC Vector for parallel case
    if (mpi_size > 1) {
#if USE_PETSC
      // Initiate PETSC residual vector across the ranks
      Vec petsc_res;
      VecCreateMPI(MPI_COMM_WORLD, PETSC_DECIDE, this->global_active_dof_,
                   &petsc_res);

      // Copying local residual vector to petsc vector
      VecSetValues(petsc_res, this->rank_global_mapper_.size(),
                   this->rank_global_mapper_.data(), residual_vector.data(),
                   ADD_VALUES);
      VecAssemblyBegin(petsc_res);
      VecAssemblyEnd(petsc_res);

      // Compute PETSC Vector norm in all rank
      PetscScalar res_norm;
      VecNorm(petsc_res, NORM_2, &res_norm);
      residual_norm = res_norm;

      // Destroy vector
      VecDestroy(&petsc_res);
#endif
    } else {
      residual_norm = residual_vector.norm();
    }

    // Save if this is the initial iteration
    if (initial) this->initial_residual_norm_ = residual_norm;

    // Convergence check
    if (residual_norm < this->abs_tolerance_) convergence = true;

    // Convergence check with relative residual norm
    double relative_residual_norm = residual_norm / initial_residual_norm_;
    if (relative_residual_norm < this->tolerance_) convergence = true;

    if (mpi_rank == 0 && this->verbosity_ >= 2) {
      console_->info("Residual norm: {}.", residual_norm);
      console_->info("Relative residual norm: {}.", relative_residual_norm);
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return convergence;
}
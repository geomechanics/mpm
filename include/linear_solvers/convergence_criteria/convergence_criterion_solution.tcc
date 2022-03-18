//! Function to check convergence
inline bool mpm::ConvergenceCriterionSolution::check_convergence(
    const Eigen::VectorXd& solution_vector, bool save_settings) {
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
    double solution_norm;

    // Perform norm computation: using PETSC Vector for parallel case
    if (mpi_size > 1) {
#if USE_PETSC
      // Initiate PETSC solution vector across the ranks
      Vec petsc_sol;
      VecCreateMPI(MPI_COMM_WORLD, PETSC_DECIDE, this->global_active_dof_,
                   &petsc_sol);

      // Copying local residual vector to petsc vector
      VecSetValues(petsc_sol, this->rank_global_mapper_.size(),
                   this->rank_global_mapper_.data(), solution_vector.data(),
                   INSERT_VALUES);
      VecAssemblyBegin(petsc_sol);
      VecAssemblyEnd(petsc_sol);

      // Compute PETSC Vector norm in all rank
      PetscScalar sol_norm;
      VecNorm(petsc_sol, NORM_2, &sol_norm);
      solution_norm = sol_norm;

      // Destroy vector
      VecDestroy(&petsc_sol);
#endif
    } else {
      solution_norm = solution_vector.norm();
    }

    // Convergence check
    if (solution_norm < this->tolerance_) convergence = true;

    if (mpi_rank == 0 && this->verbosity_ >= 2)
      console_->info("Solution norm: {}.", solution_norm);
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return convergence;
}
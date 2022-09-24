//! Compute milne estimator newmark
template <unsigned Tdim>
double mpm::Mesh<Tdim>::compute_error_estimate_displacement_newmark(
    unsigned phase) const {

  // Check mpi rank and size
  int mpi_rank = 0;
  int mpi_size = 1;

#ifdef USE_MPI
  // Get MPI rank
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  // Compute Milne estimator
  double kappa = 0;
#pragma omp parallel for schedule(runtime) reduction(+ : kappa)
  for (auto nitr = this->nodes_.cbegin(); nitr != this->nodes_.cend(); ++nitr) {
    const auto& displacement = (*nitr)->displacement(phase);
    const auto& pred_displacement = (*nitr)->predictor_displacement(phase);
    kappa += (displacement - pred_displacement).squaredNorm();
  }

  // Perform norm computation: using PETSC Vector for parallel case
  if (mpi_size > 1) {
#if USE_PETSC
    // Substract half squared norm from domain shared nodes
    double halo_nodes_norm = 0;
#pragma omp parallel for schedule(runtime) reduction(+ : halo_nodes_norm)
    for (auto nitr = domain_shared_nodes_.cbegin();
         nitr != domain_shared_nodes_.cend(); ++nitr) {
      const double nrank = double(domain_shared_nodes_[i]->mpi_ranks().size());
      const auto& displacement = (*nitr)->displacement(phase);
      const auto& pred_displacement = (*nitr)->predictor_displacement(phase);
      halo_nodes_norm += (nrank - 1.0) / nrank *
                         (displacement - pred_displacement).squaredNorm();
    }

    // Substract the multiple count of domain shared nodes
    kappa = kappa - halo_nodes_norm;

    // MPI All reduce kappa
    double global_kappa;
    MPI_Allreduce(&kappa, &global_kappa, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    kappa = global_kappa;
#endif
  }

  // Squared kappa
  kappa = std::sqrt(kappa);

  return kappa;
}
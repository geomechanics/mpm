//! Compute milne estimator newmark
template <unsigned Tdim>
double mpm::Mesh<Tdim>::compute_error_estimate_displacement_newmark(
    unsigned phase) const {

  // Check mpi rank and size
  int mpi_size = 1;

#ifdef USE_MPI
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

  if (mpi_size > 1) {
#if USE_PETSC
    // Substract half squared norm from domain shared nodes
    double halo_nodes_norm_squared = 0;
#pragma omp parallel for schedule(runtime) reduction(+ : halo_nodes_norm_squared)
    for (auto nitr = domain_shared_nodes_.cbegin();
         nitr != domain_shared_nodes_.cend(); ++nitr) {
      const double nrank = double((*nitr)->mpi_ranks().size());
      const auto& displacement = (*nitr)->displacement(phase);
      const auto& pred_displacement = (*nitr)->predictor_displacement(phase);
      halo_nodes_norm_squared +=
          (nrank - 1.0) / nrank *
          (displacement - pred_displacement).squaredNorm();
    }

    // Substract the multiple counts of domain shared nodes
    kappa = kappa - halo_nodes_norm_squared;

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

//! Compute newmark critical time step
template <unsigned Tdim>
double mpm::Mesh<Tdim>::critical_time_step_newmark(unsigned phase) const {

  // Check mpi rank and size
  int mpi_size = 1;

#ifdef USE_MPI
  // Get number of MPI ranks
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  // Compute average mesh size
  double h = this->compute_average_cell_size();

  // Compute largest velocity magnitude
  double v = 0.0;
#pragma omp parallel for schedule(runtime) reduction(max : v)
  for (auto nitr = this->nodes_.cbegin(); nitr != this->nodes_.cend(); ++nitr)
    v = std::max(v, (*nitr)->velocity(phase).norm());

  // Compute critical time step
  if (v < std::numeric_limits<double>::epsilon())
    v = std::numeric_limits<double>::epsilon();
  double time_crit = h / v;

  if (mpi_size > 1) {
#if USE_PETSC
    // MPI All reduce minimum time_crit
    double global_time_crit;
    MPI_Allreduce(&time_crit, &global_time_crit, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    time_crit = global_time_crit;
#endif
  }

  return time_crit;
}
//! Construct a implicit eigen matrix assembler
template <unsigned Tdim>
mpm::AssemblerEigenImplicit<Tdim>::AssemblerEigenImplicit(
    unsigned node_neighbourhood)
    : mpm::AssemblerBase<Tdim>(node_neighbourhood) {
  //! Logger
  std::string logger = "AssemblerEigenImplicit::";
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Assemble stiffness matrix
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::assemble_stiffness_matrix() {
  bool status = true;
  try {
    // Initialise stiffness matrix
    stiffness_matrix_.resize(active_dof_ * Tdim, active_dof_ * Tdim);
    stiffness_matrix_.setZero();

    // Triplets and reserve storage for sparse matrix
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(active_dof_ * Tdim * sparse_row_size_ * Tdim);

    // Cell pointer
    const auto& cells = mesh_->cells();

    // Active nodes
    const auto& active_nodes = mesh_->active_nodes();
    const unsigned nactive_node = active_nodes.size();

    // Iterate over cells
    mpm::Index cid = 0;
    for (auto cell_itr = cells.cbegin(); cell_itr != cells.cend(); ++cell_itr) {
      if ((*cell_itr)->status()) {
        // Node ids in each cell
        const auto nids = global_node_indices_.at(cid);

        // Element stiffness of cell
        const auto cell_stiffness = (*cell_itr)->stiffness_matrix();

        // Assemble global stiffness matrix
        for (unsigned i = 0; i < nids.size(); ++i) {
          for (unsigned j = 0; j < nids.size(); ++j) {
            for (unsigned k = 0; k < Tdim; ++k) {
              for (unsigned l = 0; l < Tdim; ++l) {
                if (std::abs(cell_stiffness(Tdim * i + k, Tdim * j + l)) >
                    std::numeric_limits<double>::epsilon())
                  tripletList.push_back(Eigen::Triplet<double>(
                      nactive_node * k + global_node_indices_.at(cid)(i),
                      nactive_node * l + global_node_indices_.at(cid)(j),
                      cell_stiffness(Tdim * i + k, Tdim * j + l)));
              }
            }
          }
        }
        ++cid;
      }
    }

    // Fast assembly from triplets
    stiffness_matrix_.setFromTriplets(tripletList.begin(), tripletList.end());

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Assemble residual force right vector
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::assemble_residual_force_right() {
  bool status = true;
  try {
    // Initialise residual force RHS vector
    residual_force_rhs_vector_.resize(active_dof_ * Tdim);
    residual_force_rhs_vector_.setZero();

    const unsigned solid = mpm::ParticlePhase::SinglePhase;

    // Active nodes pointer
    const auto& nodes = mesh_->active_nodes();
    // Iterate over nodes
    mpm::Index nid = 0;
    for (auto node_itr = nodes.cbegin(); node_itr != nodes.cend(); ++node_itr) {
      const Eigen::Matrix<double, Tdim, 1> residual_force =
          (*node_itr)->external_force(solid) +
          (*node_itr)->internal_force(solid);

      for (unsigned i = 0; i < Tdim; ++i) {
        // Nodal residual force
        residual_force_rhs_vector_(active_dof_ * i + nid) = residual_force[i];
      }
      nid++;
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign displacement constraints
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::assign_displacement_constraints(
    double current_time) {
  bool status = false;
  try {
    // Resize displacement constraints vector
    displacement_constraints_.resize(active_dof_ * Tdim);
    displacement_constraints_.reserve(int(0.5 * active_dof_ * Tdim));

    // Nodes container
    const auto& nodes = mesh_->active_nodes();
    // Iterate over nodes to get displacement constraints
    for (auto node = nodes.cbegin(); node != nodes.cend(); ++node) {
      for (unsigned i = 0; i < Tdim; ++i) {
        // Assign total displacement constraint
        const double displacement_constraint =
            (*node)->displacement_constraint(i, current_time);

        // Check if there is a displacement constraint
        if (displacement_constraint != std::numeric_limits<double>::max()) {
          // Insert the displacement constraints
          displacement_constraints_.insert(
              active_dof_ * i + (*node)->active_id()) = displacement_constraint;
        }
      }
    }
    status = true;
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return status;
}

//! Apply displacement constraints vector
template <unsigned Tdim>
void mpm::AssemblerEigenImplicit<Tdim>::apply_displacement_constraints() {
  try {
    // Modify residual_force_rhs_vector_
    residual_force_rhs_vector_ -= stiffness_matrix_ * displacement_constraints_;

    // Apply displacement constraints
    for (Eigen::SparseVector<double>::InnerIterator it(
             displacement_constraints_);
         it; ++it) {
      // Modify residual force_rhs_vector
      residual_force_rhs_vector_(it.index()) = it.value();
      // Modify stiffness_matrix
      stiffness_matrix_.row(it.index()) *= 0;
      stiffness_matrix_.col(it.index()) *= 0;
      stiffness_matrix_.coeffRef(it.index(), it.index()) = 1;
    }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
}

// Check residual convergence of Newton-Raphson iteration
// TODO: This function should be moved to a separate class
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::check_residual_convergence(
    bool initial, unsigned verbosity, double residual_tolerance,
    double relative_residual_tolerance) {
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

    // Perform norm computation: using PETSC Vector for parallel case
    if (mpi_size > 1) {
#ifdef USE_MPI
      // Prepare rank global mapper
      std::vector<int> vector_rgm;
      for (unsigned dir = 0; dir < Tdim; ++dir) {
        auto dir_rgm = this->rank_global_mapper_;
        std::for_each(dir_rgm.begin(), dir_rgm.end(),
                      [size = this->global_active_dof_, dir = dir](int& rgm) {
                        rgm += dir * size;
                      });
        vector_rgm.insert(vector_rgm.end(), dir_rgm.begin(), dir_rgm.end());
      }

      // Initiate PETSC residual vector across the ranks
      Vec petsc_res;
      VecCreateMPI(MPI_COMM_WORLD, PETSC_DECIDE,
                   Tdim * this->global_active_dof_, &petsc_res);

      // Copying local residual vector to petsc vector
      VecSetValues(petsc_res, vector_rgm.size(), vector_rgm.data(),
                   residual_force_rhs_vector_.data(), ADD_VALUES);
      VecAssemblyBegin(petsc_res);
      VecAssemblyEnd(petsc_res);

      // Compute PETSC Vector norm in all rank
      PetscScalar res_norm;
      VecNorm(petsc_res, NORM_2, &res_norm);
      residual_norm_ = res_norm;

      // Destroy vector
      VecDestroy(&petsc_res);
#endif
    } else {
      residual_norm_ = residual_force_rhs_vector_.norm();
    }

    // Save if this is the initial iteration
    if (initial) initial_residual_norm_ = residual_norm_;

    // Convergence check
    if (residual_norm_ < residual_tolerance) convergence = true;

    // Convergence check with relative residual norm
    relative_residual_norm_ = residual_norm_ / initial_residual_norm_;
    if (relative_residual_norm_ < relative_residual_tolerance)
      convergence = true;

    if (mpi_rank == 0 && verbosity == 2) {
      console_->info("Residual norm: {}.", residual_norm_);
      console_->info("Relative residual norm: {}.", relative_residual_norm_);
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return convergence;
}

// Check solution convergence of Newton-Raphson iteration
// TODO: This function should be moved to a separate class
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::check_solution_convergence(
    unsigned verbosity, double solution_tolerance) {
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

    // Perform norm computation: using PETSC Vector for parallel case
    if (mpi_size > 1) {
#ifdef USE_MPI
      // Prepare rank global mapper
      std::vector<int> vector_rgm;
      for (unsigned dir = 0; dir < Tdim; ++dir) {
        auto dir_rgm = this->rank_global_mapper_;
        std::for_each(dir_rgm.begin(), dir_rgm.end(),
                      [size = this->global_active_dof_, dir = dir](int& rgm) {
                        rgm += dir * size;
                      });
        vector_rgm.insert(vector_rgm.end(), dir_rgm.begin(), dir_rgm.end());
      }

      // Initiate PETSC solution vector across the ranks
      Vec petsc_sol;
      VecCreateMPI(MPI_COMM_WORLD, PETSC_DECIDE,
                   Tdim * this->global_active_dof_, &petsc_sol);

      // Copying local residual vector to petsc vector
      VecSetValues(petsc_sol, vector_rgm.size(), vector_rgm.data(),
                   displacement_increment_.data(), INSERT_VALUES);
      VecAssemblyBegin(petsc_sol);
      VecAssemblyEnd(petsc_sol);

      // Compute PETSC Vector norm in all rank
      PetscScalar sol_norm;
      VecNorm(petsc_sol, NORM_2, &sol_norm);
      disp_increment_norm_ = sol_norm;

      // Destroy vector
      VecDestroy(&petsc_sol);
#endif
    } else {
      disp_increment_norm_ = displacement_increment_.norm();
    }

    // Convergence check
    if (disp_increment_norm_ < solution_tolerance) convergence = true;

    if (mpi_rank == 0 && verbosity == 2)
      console_->info("Displacement increment norm: {}.", disp_increment_norm_);
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return convergence;
}
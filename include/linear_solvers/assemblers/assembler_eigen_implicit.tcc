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
                  tripletList.emplace_back(Eigen::Triplet<double>(
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

    // Apply null-space treatment
    this->apply_null_space_treatment(tripletList, Tdim);

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

//! Add 3D printing nozzle displacement constraints
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::assign_3dp_displacement_constraints(
    double dt, unsigned phase) {
  bool status = false;
  try {
    const auto& nodes = mesh_->active_nodes();
    for (auto node = nodes.cbegin(); node != nodes.cend(); ++node) {
      if (!(*node)->three_dp_nozzle()) continue;
      const auto velocity = (*node)->three_dp_velocity();
      const auto displacement = (*node)->displacement(phase);
      for (unsigned i = 0; i < Tdim; ++i)
        // Remaining correction to reach the prescribed increment
        displacement_constraints_.coeffRef(active_dof_ * i +
                                           (*node)->active_id()) =
            velocity(i) * dt - displacement(i);
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
    residual_force_rhs_vector_ +=
        -stiffness_matrix_ * displacement_constraints_;

    // Apply displacement constraints
    // (single pass over the nonzeros: zeroing rows/columns one constrained
    // dof at a time costs O(nnz) per dof on a column-major sparse matrix)
    std::vector<char> constrained(stiffness_matrix_.rows(), 0);
    for (Eigen::SparseVector<double>::InnerIterator it(
             displacement_constraints_);
         it; ++it) {
      // Modify residual force_rhs_vector
      residual_force_rhs_vector_(it.index()) = it.value();
      constrained[it.index()] = 1;
    }
    // Zero the rows and columns of constrained dofs
    for (int k = 0; k < stiffness_matrix_.outerSize(); ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(stiffness_matrix_, k);
           it; ++it)
        if (constrained[it.row()] || constrained[it.col()]) it.valueRef() = 0.;
    // Unit diagonal for constrained dofs
    for (Eigen::SparseVector<double>::InnerIterator it(
             displacement_constraints_);
         it; ++it)
      stiffness_matrix_.coeffRef(it.index(), it.index()) = 1;

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
}

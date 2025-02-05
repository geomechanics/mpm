//! Assemble thermal expansivity matrix
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::assemble_thermal_expansivity_matrix() {
  bool status = true;
  try {
    // Initialise thermal expansivity matrix
    thermal_expansivity_matrix_.resize(active_dof_ * Tdim, active_dof_);
    thermal_expansivity_matrix_.setZero();

    // Triplets and reserve storage for sparse matrix
    std::vector<Eigen::Triplet<double>> tripletList;
    // To be check
    tripletList.reserve(active_dof_ * Tdim * sparse_row_size_);

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

        // Element thermal expansivity of cell
        const auto cell_thermal_expansivity = 
                              (*cell_itr)->thermal_expansivity_matrix();

        // Assemble thermal expansivity matrix
        for (unsigned i = 0; i < nids.size(); ++i) {
          for (unsigned j = 0; j < nids.size(); ++j) {
            for (unsigned k = 0; k < Tdim; ++k) {
                if (std::abs(cell_thermal_expansivity(Tdim * i + k, j)) >
                    std::numeric_limits<double>::epsilon())
                  tripletList.emplace_back(Eigen::Triplet<double>(
                      nactive_node * k + global_node_indices_.at(cid)(i),
                      global_node_indices_.at(cid)(j),
                      cell_thermal_expansivity(Tdim * i + k, j)));
            }
          }
        }
        ++cid;
      }
    }

    // Apply null-space treatment
    this->apply_null_space_treatment(tripletList, 1);

    // Fast assembly from triplets
    thermal_expansivity_matrix_.setFromTriplets(tripletList.begin(), 
                                                tripletList.end());

    // Eigen::IOFormat matrix_fmt(4, 0, ", ", "\n");
    // std::cout << "\n" << "thermal_expansivity_matrix = " << "\n"
    //           << Eigen::MatrixXd(thermal_expansivity_matrix_).format(matrix_fmt) 
    //           << "\n";

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assemble thermal conductivity matrix
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::assemble_thermal_conductivity_matrix() {
  bool status = true;
  try {
    // Initialise global thermal conductivity matrix
    thermal_conductivity_matrix_.resize(active_dof_, active_dof_);
    thermal_conductivity_matrix_.setZero();

    // Triplets and reserve storage for sparse matrix
    std::vector<Eigen::Triplet<double>> tripletList;
    // To be check
    tripletList.reserve(active_dof_ * sparse_row_size_);

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

        // Element thermal conductivity of cell
        const auto cell_thermal_conductivity = 
                            (*cell_itr)->thermal_conductivity_matrix();

        // Assemble global thermal conductivity matrix
        for (unsigned i = 0; i < nids.size(); ++i) {
          for (unsigned j = 0; j < nids.size(); ++j) {
            if (std::abs(cell_thermal_conductivity(i, j)) >
                std::numeric_limits<double>::epsilon())
              tripletList.emplace_back(Eigen::Triplet<double>(
                  global_node_indices_.at(cid)(i),
                  global_node_indices_.at(cid)(j),
                  cell_thermal_conductivity(i, j)));
          }
        }
        ++cid;
      }
    }

    // Apply null-space treatment
    this->apply_null_space_treatment(tripletList, 1);

    // Fast assembly from triplets
    thermal_conductivity_matrix_.setFromTriplets(tripletList.begin(), 
                                              tripletList.end());

    // Eigen::IOFormat matrix_fmt(4, 0, ", ", "\n");
    // std::cout << "thermal_conductivity_matrix = " << "\n"
    //           << Eigen::MatrixXd(thermal_conductivity_matrix_).format(matrix_fmt) 
    //           << "\n";
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Assemble residual heat right vector
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::assemble_residual_heat_right() {
  bool status = true;
  try {
    // Initialise residual fheat RHS vector
    residual_heat_rhs_vector_.resize(active_dof_);
    residual_heat_rhs_vector_.setZero();

    const unsigned solid = mpm::ParticlePhase::SinglePhase;

    // Active nodes pointer
    const auto& nodes = mesh_->active_nodes();
    // Iterate over nodes
    mpm::Index nid = 0;
    for (auto node_itr = nodes.cbegin(); node_itr != nodes.cend(); ++node_itr) {
      const double residual_heat =
        (*node_itr)->internal_heat(solid) + (*node_itr)->external_heat(solid);
        
      // Nodal residual heat
      residual_heat_rhs_vector_(nid) = residual_heat;

      nid++;
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assemble global stiffness matrix
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::assemble_global_stiffness_matrix() {
  bool status = true;
  try {
    // Number of displacement DOFs
    unsigned n_u = active_dof_ * Tdim;
    // Number of temperature DOFs
    unsigned n_T = active_dof_;        
    // Initialise golbal stiffness matrix
    global_stiffness_matrix_.resize(n_u + n_T, n_u + n_T);
    global_stiffness_matrix_.setZero();

    Eigen::SparseMatrix<double> K_uu = stiffness_matrix_;
    Eigen::SparseMatrix<double> K_uT = thermal_expansivity_matrix_;
    Eigen::SparseMatrix<double> K_TT = thermal_conductivity_matrix_;

    // Create a list of triplets to assemble the global stiffness matrix
    std::vector<Eigen::Triplet<double>> tripletList;
    // Reserve memory for the triplets
    tripletList.reserve(K_uu.nonZeros() + K_uT.nonZeros() + K_TT.nonZeros());

    // Insert K_uu into the global stiffness matrix
    for (unsigned k = 0; k < K_uu.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(K_uu, k); it; ++it) {
          tripletList.emplace_back(it.row(), it.col(), it.value());
      }
    }

    // Insert K_uT into the global stiffness matrix
    for (unsigned k = 0; k < K_uT.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(K_uT, k); it; ++it) {
          tripletList.emplace_back(it.row(), it.col() + n_u, it.value());
      }
    }

    // Insert K_TT into the global stiffness matrix
    for (unsigned k = 0; k < K_TT.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(K_TT, k); it; ++it) {
          tripletList.emplace_back(it.row() + n_u, it.col() + n_u, it.value());
      }
    }

    // Assemble the coupling stiffness matrix from the triplets
    global_stiffness_matrix_.setFromTriplets(tripletList.begin(), tripletList.end());

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assemble global residual matrix
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::assemble_global_residual_right() {
  bool status = true;
  try {
    // Number of displacement DOFs
    unsigned n_u = active_dof_ * Tdim;
    // Number of temperature DOFs
    unsigned n_T = active_dof_;        
    // Initialise golbal coupling matrix
    global_residual_rhs_vector_.resize(n_u + n_T);
    global_residual_rhs_vector_.setZero();

    // Assign residual force to the top block
    global_residual_rhs_vector_.head(n_u) = residual_force_rhs_vector_; 
    // Assign residual heat to the bottom block
    global_residual_rhs_vector_.tail(n_T) = residual_heat_rhs_vector_; 

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign temperature constraints
template <unsigned Tdim>
bool mpm::AssemblerEigenImplicit<Tdim>::assign_temperature_constraints(
    double current_time) {
  bool status = false;
  try {
    // Resize temperature constraints vector
    temperature_increment_constraints_.resize(active_dof_);
    temperature_increment_constraints_.reserve(int(0.5 * active_dof_));

    // Nodes container
    const auto& nodes = mesh_->active_nodes();
    // Iterate over nodes to get temperature constraints
    for (auto node = nodes.cbegin(); node != nodes.cend(); ++node) {

        // Assign total temperature constraint
        const double temperature_increment_constraint =
            (*node)->temperature_increment_constraint(current_time);

        // Check if there is a temperature constraint
        if (temperature_increment_constraint != 
                      std::numeric_limits<double>::max()) {
          // Insert the temperature increment constraints
          temperature_increment_constraints_.insert(
              (*node)->active_id()) = temperature_increment_constraint;
        }

    }
    status = true;
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return status;
}

//! Apply temperature constraints vector
template <unsigned Tdim>
void mpm::AssemblerEigenImplicit<Tdim>::
                              apply_temperature_increment_constraints() {
  try {
    // Modify residual heat rhs vector 
    residual_heat_rhs_vector_ +=
        -thermal_conductivity_matrix_ * temperature_increment_constraints_;

    // Apply temperature constraints
    for (Eigen::SparseVector<double>::InnerIterator it(
              temperature_increment_constraints_);
          it; ++it) {
      // Modify residual heat_rhs_vector
      residual_heat_rhs_vector_(it.index()) = it.value();
      // Modify heat laplacian matrix
      thermal_conductivity_matrix_.row(it.index()) *= 0;
      thermal_conductivity_matrix_.col(it.index()) *= 0;
      // thermal_expansivity_matrix_.col(it.index()) *= 0;
      thermal_conductivity_matrix_.coeffRef(it.index(), it.index()) = 1;
    }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
}

//! Apply temperature and displacement constraints vector to coupling matrix
template <unsigned Tdim>
void mpm::AssemblerEigenImplicit<Tdim>::apply_coupling_constraints() {
  try {
    // Modify residual_force_rhs_vector_
    residual_force_rhs_vector_ +=
        -thermal_expansivity_matrix_ * temperature_increment_constraints_;

    // Apply displacement constraints
    for (Eigen::SparseVector<double>::InnerIterator it(
              displacement_constraints_);
          it; ++it) {
      // Modify thermal expansivity matrix      
      thermal_expansivity_matrix_.row(it.index()) *= 0;
    }

    // Apply temperature constraints
    for (Eigen::SparseVector<double>::InnerIterator it(
              temperature_increment_constraints_);
          it; ++it) {
      // Modify thermal expansion matrix      
      thermal_expansivity_matrix_.col(it.index()) *= 0;
    }

    // Eigen::IOFormat matrix_fmt(4, 0, ", ", "\n");
    // std::cout << "\n" << "thermal_expansivity_matrix_modified = " << "\n"
    //           << Eigen::MatrixXd(thermal_expansivity_matrix_).format(matrix_fmt) 
    //           << "\n";
    // std::cout << "thermal_conductivity_matrix_modified = " << "\n"
    //           << Eigen::MatrixXd(thermal_conductivity_matrix_).format(matrix_fmt)
    //           << "\n";
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
}

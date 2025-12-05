//! Construct a semi-implicit eigen matrix assembler
template <unsigned Tdim>
mpm::AssemblerEigenSemiImplicitTwoPhase<
    Tdim>::AssemblerEigenSemiImplicitTwoPhase(unsigned node_neighbourhood)
    : mpm::AssemblerEigenSemiImplicitNavierStokes<Tdim>(node_neighbourhood) {
  //! Logger
  std::string logger = "AssemblerEigenSemiImplicitTwoPhase::";
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Assemble coefficient matrix for two-phase predictor
template <unsigned Tdim>
bool mpm::AssemblerEigenSemiImplicitTwoPhase<Tdim>::assemble_predictor_left(
    double dt) {
  bool status = true;
  try {
    // Loop over three direction
    for (unsigned dir = 0; dir < Tdim; dir++) {
      // Initialise coefficient_matrix
      Eigen::SparseMatrix<double> coefficient_matrix;
      coefficient_matrix.setZero();

      // Resize coefficient matrix
      coefficient_matrix.resize(2 * active_dof_, 2 * active_dof_);

      // Reserve storage for sparse matrix
      coefficient_matrix.reserve(
          Eigen::VectorXi::Constant(2 * active_dof_, 2 * sparse_row_size_));

      // Cell pointer
      const auto& cells = mesh_->cells();

      // Iterate over cells for drag force coefficient
      mpm::Index cid = 0;
      for (auto cell_itr = cells.cbegin(); cell_itr != cells.cend();
           ++cell_itr) {
        if ((*cell_itr)->status()) {
          // Node ids in each cell
          const auto nids = global_node_indices_.at(cid);
          // Local drag matrix
          auto cell_drag_matrix = (*cell_itr)->drag_matrix(dir);
          // Assemble global coefficient matrix
          for (unsigned i = 0; i < nids.size(); ++i) {
            for (unsigned j = 0; j < nids.size(); ++j) {
              coefficient_matrix.coeffRef(nids(i) + active_dof_, nids(j)) +=
                  -cell_drag_matrix(i, j) * dt;
              coefficient_matrix.coeffRef(nids(i) + active_dof_,
                                          nids(j) + active_dof_) +=
                  cell_drag_matrix(i, j) * dt;
            }
          }
          ++cid;
        }
      }

      // Active nodes pointer
      const auto& nodes = mesh_->active_nodes();
      // Iterate over cells for mass coefficient
      for (auto node_itr = nodes.cbegin(); node_itr != nodes.cend();
           ++node_itr) {
        // Id for active node
        auto active_id = (*node_itr)->active_id();
        // Assemble global coefficient matrix for solid mass
        coefficient_matrix.coeffRef(active_id, active_id) +=
            (*node_itr)->mass(mpm::NodePhase::NSolid);
        // Assemble global coefficient matrix for liquid mass
        coefficient_matrix.coeffRef(active_id + active_dof_,
                                    active_id + active_dof_) +=
            (*node_itr)->mass(mpm::NodePhase::NLiquid);
        coefficient_matrix.coeffRef(active_id, active_id + active_dof_) +=
            (*node_itr)->mass(mpm::NodePhase::NLiquid);
      }

      // Apply null-space treatment
      this->apply_null_space_treatment(coefficient_matrix, 2);

      // Add coefficient matrix to map
      if (predictor_lhs_matrix_.find(dir) != predictor_lhs_matrix_.end())
        predictor_lhs_matrix_.erase(dir);

      predictor_lhs_matrix_.insert(
          std::make_pair<unsigned, Eigen::SparseMatrix<double>>(
              static_cast<unsigned>(dir),
              static_cast<Eigen::SparseMatrix<double>>(coefficient_matrix)));
    }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assemble predictor RHS force vector
template <unsigned Tdim>
bool mpm::AssemblerEigenSemiImplicitTwoPhase<Tdim>::assemble_predictor_right(
    double dt) {
  bool status = true;
  try {
    // Resize intermediate velocity vector
    intermediate_acceleration_.resize(active_dof_ * 2, Tdim);
    intermediate_acceleration_.setZero();

    // Resize force vector
    predictor_rhs_vector_.resize(active_dof_ * 2, Tdim);
    predictor_rhs_vector_.setZero();

    // Active nodes pointer
    const auto& nodes = mesh_->active_nodes();
    // Iterate over nodes
    mpm::Index nid = 0;
    for (auto node_itr = nodes.cbegin(); node_itr != nodes.cend(); ++node_itr) {
      // Compute nodal intermediate force
      const Eigen::Matrix<double, Tdim, 1> mixture_force =
          (*node_itr)->external_force(mpm::NodePhase::NMixture) +
          (*node_itr)->internal_force(mpm::NodePhase::NMixture);

      const Eigen::Matrix<double, Tdim, 1> drag_force =
          (*node_itr)->drag_force_coefficient().cwiseProduct(
              (*node_itr)->velocity(mpm::NodePhase::NLiquid) -
              (*node_itr)->velocity(mpm::NodePhase::NSolid));

      const Eigen::Matrix<double, Tdim, 1> fluid_force =
          (*node_itr)->external_force(mpm::NodePhase::NLiquid) +
          (*node_itr)->internal_force(mpm::NodePhase::NLiquid) - drag_force;

      // Assemble intermediate force vector
      predictor_rhs_vector_.row(nid) = mixture_force.transpose();
      predictor_rhs_vector_.row(nid + active_dof_) = fluid_force.transpose();
      ++nid;
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Assemble poisson right vector
template <unsigned Tdim>
bool mpm::AssemblerEigenSemiImplicitTwoPhase<Tdim>::assemble_poisson_right(
    double dt) {
  bool status = true;
  try {
    // Initialise Poisson RHS matrix
    Eigen::SparseMatrix<double> solid_poisson_right_matrix,
        liquid_poisson_right_matrix;

    // Resize poisson right matrix for solid
    solid_poisson_right_matrix.resize(active_dof_, active_dof_ * Tdim);
    solid_poisson_right_matrix.setZero();
    // Resize poisson right matrix for liquid
    liquid_poisson_right_matrix.resize(active_dof_, active_dof_ * Tdim);
    liquid_poisson_right_matrix.setZero();

    // Reserve storage for sparse matrix
    solid_poisson_right_matrix.reserve(
        Eigen::VectorXi::Constant(active_dof_ * Tdim, sparse_row_size_));
    liquid_poisson_right_matrix.reserve(
        Eigen::VectorXi::Constant(active_dof_ * Tdim, sparse_row_size_));

    // Cell pointer
    const auto& cells = mesh_->cells();

    // Iterate over cells
    mpm::Index cid = 0;
    for (auto cell_itr = cells.cbegin(); cell_itr != cells.cend(); ++cell_itr) {
      if ((*cell_itr)->status()) {
        // Node ids in each cell
        const auto nids = global_node_indices_.at(cid);

        // Local Poisson RHS matrix for solid
        auto cell_poisson_right_solid =
            (*cell_itr)->poisson_right_matrix(mpm::NodePhase::NSolid);
        // Local Poisson RHS matrix for liquid
        auto cell_poisson_right_liquid =
            (*cell_itr)->poisson_right_matrix(mpm::NodePhase::NLiquid);

        // Assemble global poisson RHS matrix
        for (unsigned i = 0; i < nids.size(); ++i) {
          for (unsigned j = 0; j < nids.size(); ++j) {
            for (unsigned k = 0; k < Tdim; ++k) {
              solid_poisson_right_matrix.coeffRef(
                  global_node_indices_.at(cid)(i),
                  global_node_indices_.at(cid)(j) + k * active_dof_) +=
                  cell_poisson_right_solid(i, j + k * nids.size());
              liquid_poisson_right_matrix.coeffRef(
                  global_node_indices_.at(cid)(i),
                  global_node_indices_.at(cid)(j) + k * active_dof_) +=
                  cell_poisson_right_liquid(i, j + k * nids.size());
            }
          }
        }
        cid++;
      }
    }

    // Resize poisson right vector
    poisson_rhs_vector_.resize(active_dof_);
    poisson_rhs_vector_.setZero();

    // Compute intermediate solid and liquid velocity
    Eigen::MatrixXd solid_velocity, liquid_velocity;
    solid_velocity.resize(active_dof_, Tdim);
    solid_velocity.setZero();
    liquid_velocity.resize(active_dof_, Tdim);
    liquid_velocity.setZero();

    // Active nodes
    const auto& active_nodes = mesh_->active_nodes();
    unsigned node_index = 0;

    for (auto node_itr = active_nodes.cbegin(); node_itr != active_nodes.cend();
         ++node_itr) {
      // Compute nodal intermediate force
      solid_velocity.row(node_index) =
          (*node_itr)->velocity(mpm::NodePhase::NSolid).transpose();
      liquid_velocity.row(node_index) =
          (*node_itr)->velocity(mpm::NodePhase::NLiquid).transpose();
      node_index++;
    }

    // Resize velocity vectors
    solid_velocity.resize(active_dof_ * Tdim, 1);
    liquid_velocity.resize(active_dof_ * Tdim, 1);

    // Compute poisson RHS vector
    poisson_rhs_vector_ = -(solid_poisson_right_matrix * solid_velocity) -
                          (liquid_poisson_right_matrix * liquid_velocity);

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assemble corrector right matrix
template <unsigned Tdim>
bool mpm::AssemblerEigenSemiImplicitTwoPhase<Tdim>::assemble_corrector_right(
    double dt) {
  bool status = true;
  try {
    // 1. トリプレット（行, 列, 値 の組）を格納する配列を用意
    std::vector<Eigen::Triplet<double>> triplets;
    
    // 必要なメモリを概算して予約（再確保のコストを防ぐ）
    // (セル数 * ノード数^2 * 次元 * 2相分) 程度
    const auto& cells = mesh_->cells();
    unsigned estimated_entries = cells.size() * 64 * Tdim * 2; // 適当な係数
    triplets.reserve(estimated_entries);

    // 2. セルごとのループ（OpenMPで並列化可能！）
    // Tripletへの追加はスレッドセーフではないので、
    // 並列化する場合はスレッドごとにvectorを作るなどの工夫が必要ですが、
    // 単純化のためここではシリアルで書きます（それでもcoeffRefより圧倒的に速い）

    unsigned cid = 0;
    for (auto cell_itr = cells.cbegin(); cell_itr != cells.cend(); ++cell_itr) {
      if ((*cell_itr)->status()) {
        const auto& cell_nodes = global_node_indices_.at(cid);
        unsigned nnodes_per_cell = cell_nodes.size();

        // Local correction matrices
        auto correction_matrix_solid =
            (*cell_itr)->correction_matrix(mpm::NodePhase::NSolid);
        auto coefficient_matrix_liquid =
            (*cell_itr)->correction_matrix(mpm::NodePhase::NLiquid);

        for (unsigned k = 0; k < Tdim; k++) {
          for (unsigned i = 0; i < nnodes_per_cell; i++) {
            unsigned row_solid = cell_nodes(i);
            unsigned row_liquid = cell_nodes(i) + active_dof_;
            
            for (unsigned j = 0; j < nnodes_per_cell; j++) {
              unsigned col = k * active_dof_ + cell_nodes(j);

              // Solid phase value
              double val_solid = correction_matrix_solid(i, j + k * nnodes_per_cell);
              if (std::abs(val_solid) > 1e-16) { // ゼロに近い値は入れないのも手
                 triplets.emplace_back(row_solid, col, val_solid);
              }

              // Liquid phase value
              double val_liquid = coefficient_matrix_liquid(i, j + k * nnodes_per_cell);
              if (std::abs(val_liquid) > 1e-16) {
                 triplets.emplace_back(row_liquid, col, val_liquid);
              }
            }
          }
        }
        cid++;
      }
    }

    // 3. 行列のリサイズと構築を一括で行う
    correction_matrix_.resize(2 * active_dof_, active_dof_ * Tdim);
    // 重複したインデックスの値は自動的に加算される
    correction_matrix_.setFromTriplets(triplets.begin(), triplets.end());

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign pressure constraints
template <unsigned Tdim>
bool mpm::AssemblerEigenSemiImplicitTwoPhase<Tdim>::assign_pressure_constraints(
    double beta, double current_time) {
  bool status = false;
  try {
    // Resize pressure constraints vector
    pressure_constraints_.setZero();
    pressure_constraints_.data().squeeze();
    pressure_constraints_.resize(active_dof_);
    pressure_constraints_.reserve(int(0.5 * active_dof_));

    // Nodes container
    const auto& nodes = mesh_->active_nodes();
    // Iterate over nodes to get pressure constraints
    for (auto node = nodes.cbegin(); node != nodes.cend(); ++node) {
      // Assign total pressure constraint
      const double pressure_constraint =
          (*node)->pressure_constraint(mpm::NodePhase::NLiquid, current_time);

      // Check if there is a pressure constraint
      if (pressure_constraint != std::numeric_limits<double>::max()) {
        // Insert the pressure constraints
        pressure_constraints_.insert((*node)->active_id()) =
            (1 - beta) * pressure_constraint;
      }
    }
    status = true;
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return status;
}

//! Assign velocity constraints
template <unsigned Tdim>
bool mpm::AssemblerEigenSemiImplicitTwoPhase<
    Tdim>::assign_velocity_constraints() {
  bool status = false;
  try {
    // Initialise constraints matrix from triplet
    std::vector<Eigen::Triplet<double>> triplet_list;
    // Nodes container
    const auto& nodes = mesh_->active_nodes();
    // Iterate over nodes
    for (auto node = nodes.cbegin(); node != nodes.cend(); ++node) {
      // Get velocity constraints
      const auto& velocity_constraints = (*node)->velocity_constraints();
      // Assign constraints matrix
      for (const auto constraint : velocity_constraints) {
        // Insert constraint to the matrix
        triplet_list.push_back(Eigen::Triplet<double>(
            (constraint).first / Tdim * active_dof_ + (*node)->active_id(),
            (constraint).first % Tdim, (constraint).second));
      }
    }
    // Reserve the storage for the velocity constraints matrix
    velocity_constraints_.setZero();
    velocity_constraints_.data().squeeze();
    velocity_constraints_.resize(active_dof_ * 2, Tdim);
    velocity_constraints_.reserve(Eigen::VectorXi::Constant(
        Tdim, triplet_list.size() + sparse_row_size_));
    // Assemble the velocity constraints matrix
    velocity_constraints_.setFromTriplets(triplet_list.begin(),
                                          triplet_list.end());

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return status;
}

//! Apply velocity constraints to predictor RHS vector (use prune like #1)
template <unsigned Tdim>
bool mpm::AssemblerEigenSemiImplicitTwoPhase<Tdim>::apply_velocity_constraints() {
  bool status = false;
  try {
    // dir ごとに独立な線形系:  b_dir = b_dir - A_dir * bc_dir
    for (unsigned dir = 0; dir < Tdim; ++dir) {
      predictor_rhs_vector_.col(dir) +=
          -predictor_lhs_matrix_.at(dir) * velocity_constraints_.col(dir);

      // この方向で拘束されている DOF を収集
      std::unordered_set<int> constrained_dofs;
      // 速度拘束行列は [ndof x Tdim] を想定。列 dir の非ゼロが拘束DOF
      for (Eigen::SparseMatrix<double>::InnerIterator it(velocity_constraints_, static_cast<int>(dir));
           it; ++it) {
        const int dof = it.row();
        constrained_dofs.insert(dof);
        // RHS は拘束値に置き換える（A*bc 減算後なので、最終的に b(dof)=bc）
        predictor_rhs_vector_(dof, static_cast<int>(dir)) = 0.0;
      }

      // 行列を prune で間引く：拘束行 or 列に触れる非対角成分は落とす
      auto& A = predictor_lhs_matrix_.at(dir);
      A.prune([&constrained_dofs](int r, int c, double /*v*/) {
        if (constrained_dofs.count(r) || constrained_dofs.count(c)) {
          return r == c;            // 対角のみ残す
        }
        return true;                // 非拘束は残す
      });

      // 対角に 1 を再挿入（ゼロの場合も coeffRef で生やす）
      for (int dof : constrained_dofs) {
        A.coeffRef(dof, dof) = 1.0;
      }

      // ゼロ項目を落として圧縮
      A.prune(0.0);
      A.makeCompressed();
    }

    status = true;
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return status;
}


//! Apply velocity constraints to preditor RHS vector
// template <unsigned Tdim>
// bool mpm::AssemblerEigenSemiImplicitTwoPhase<
//     Tdim>::apply_velocity_constraints() {
//   bool status = false;
//   try {
//     // Modify the force vector(b = b - A * bc)
//     for (unsigned dir = 0; dir < Tdim; dir++) {
//       predictor_rhs_vector_.col(dir) +=
//           -predictor_lhs_matrix_.at(dir) * velocity_constraints_.col(dir);

//       // Iterate over velocity constraints (non-zero elements)
//       for (unsigned j = 0; j < velocity_constraints_.outerSize(); ++j) {
//         for (Eigen::SparseMatrix<double>::InnerIterator itr(
//                  velocity_constraints_, j);
//              itr; ++itr) {
//           // Check direction
//           if (itr.col() == dir) {
//             // Assign 0 to specified column
//             predictor_lhs_matrix_.at(dir).col(itr.row()) *= 0;
//             // Assign 0 to specified row
//             predictor_lhs_matrix_.at(dir).row(itr.row()) *= 0;
//             // Assign 1  to diagnal element
//             predictor_lhs_matrix_.at(dir).coeffRef(itr.row(), itr.row()) = 1.0;

//             predictor_rhs_vector_(itr.row(), itr.col()) = 0.;
//           }
//         }
//       }
//     }

//   } catch (std::exception& exception) {
//     console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
//   }
//   return status;
// }
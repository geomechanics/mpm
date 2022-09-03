//! Assign global node indices
template <unsigned Tdim>
bool mpm::AssemblerBase<Tdim>::assign_global_node_indices(
    unsigned nactive_node, unsigned nglobal_active_node) {
  bool status = true;
  try {
    // Total number of active node (in a rank) and (rank) node indices
    active_dof_ = nactive_node;
    global_node_indices_ = mesh_->global_node_indices();

#ifdef USE_MPI
    // Total number of active node (in all rank)
    global_active_dof_ = nglobal_active_node;

    // Initialise mapping vector
    rank_global_mapper_.resize(active_dof_);

    // Nodes container
    const auto& nodes = mesh_->active_nodes();
    for (int counter = 0; counter < nodes.size(); counter++) {
      // Assign get nodal global index
      rank_global_mapper_[counter] = nodes[counter]->global_active_id();
    }
#endif

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Null-space treatment of a sparse matrix given a coefficient matrix
template <unsigned Tdim>
void mpm::AssemblerBase<Tdim>::apply_null_space_treatment(
    Eigen::SparseMatrix<double>& coefficient_matrix, unsigned nblock) {
  // Nodes container
  const auto& nodes = mesh_->active_nodes();
  std::vector<mpm::Index> null_space_node;
  // Iterate over nodes to check if any nodal mass is zero
#pragma omp parallel
  {
    std::vector<mpm::Index> ns_node;
#pragma omp for nowait
    for (auto node = nodes.cbegin(); node != nodes.cend(); ++node) {
      if ((*node)->mass(mpm::NodePhase::NSinglePhase) <
          std::numeric_limits<double>::epsilon()) {
        const auto n_id = (*node)->active_id();
        for (unsigned nb = 0; nb < nblock; nb++)
          ns_node.push_back(nb * active_dof_ + n_id);
      }
    }

#pragma omp critical
    null_space_node.insert(null_space_node.end(),
                           std::make_move_iterator(ns_node.begin()),
                           std::make_move_iterator(ns_node.end()));
  }

  // Modify coefficient matrix diagonal element
  for (const auto index : null_space_node) {
    // Assign 1 to diagonal element
    coefficient_matrix.coeffRef(index, index) = 1.0;
  }
}

//! Null-space treatment of a sparse matrix given a triplet list
template <unsigned Tdim>
void mpm::AssemblerBase<Tdim>::apply_null_space_treatment(
    std::vector<Eigen::Triplet<double>>& coefficient_tripletList,
    unsigned nblock) {
  // Nodes container
  const auto& nodes = mesh_->active_nodes();
  std::vector<mpm::Index> null_space_node;
  // Iterate over nodes to check if any nodal mass is zero
#pragma omp parallel
  {
    std::vector<mpm::Index> ns_node;
#pragma omp for nowait
    for (auto node = nodes.cbegin(); node != nodes.cend(); ++node) {
      if ((*node)->mass(mpm::NodePhase::NSinglePhase) <
          std::numeric_limits<double>::epsilon()) {
        const auto n_id = (*node)->active_id();
        for (unsigned nb = 0; nb < nblock; nb++)
          ns_node.push_back(nb * active_dof_ + n_id);
      }
    }

#pragma omp critical
    null_space_node.insert(null_space_node.end(),
                           std::make_move_iterator(ns_node.begin()),
                           std::make_move_iterator(ns_node.end()));
  }

  // Modify coefficient matrix diagonal element
  for (const auto index : null_space_node) {
    // Assign 1 to diagonal element
    coefficient_tripletList.emplace_back(
        Eigen::Triplet<double>(index, index, 1.0));
  }
}
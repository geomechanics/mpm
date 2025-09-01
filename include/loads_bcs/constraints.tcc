//! Assign nodal acceleration constraints
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_acceleration_constraint(
    int set_id,
    const std::shared_ptr<mpm::AccelerationConstraint>& constraint) {
  bool status = true;
  try {
    int set_id = constraint->setid();
    auto nset = mesh_->nodes(set_id);
    if (nset.size() == 0)
      throw std::runtime_error(
          "Node set is empty for assignment of acceleration constraints");

    unsigned dir = constraint->dir();
    double acceleration = constraint->acceleration(0);
    for (auto nitr = nset.cbegin(); nitr != nset.cend(); ++nitr) {
      if (!(*nitr)->assign_acceleration_constraint(dir, acceleration))
        throw std::runtime_error(
            "Failed to initialise acceleration constraint at node");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign acceleration constraints to nodes
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_acceleration_constraints(
    const std::vector<std::tuple<mpm::Index, unsigned, double>>&
        acceleration_constraints) {
  bool status = true;
  try {
    for (const auto& acceleration_constraint : acceleration_constraints) {
      // Node id
      mpm::Index nid = std::get<0>(acceleration_constraint);
      // Direction
      unsigned dir = std::get<1>(acceleration_constraint);
      // Acceleration
      double acceleration = std::get<2>(acceleration_constraint);

      // Apply constraint
      if (!mesh_->node(nid)->assign_acceleration_constraint(dir, acceleration))
        throw std::runtime_error(
            "Nodal acceleration constraints assignment failed");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign nodal velocity constraints
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_velocity_constraint(
    int set_id, const std::shared_ptr<mpm::VelocityConstraint>& vconstraint) {
  bool status = true;
  try {
    int set_id = vconstraint->setid();
    auto nset = mesh_->nodes(set_id);
    if (nset.size() == 0)
      throw std::runtime_error(
          "Node set is empty for assignment of velocity constraints");

    unsigned dir = vconstraint->dir();
    double velocity = vconstraint->velocity();
    for (auto nitr = nset.cbegin(); nitr != nset.cend(); ++nitr) {
      if (!(*nitr)->assign_velocity_constraint(dir, velocity))
        throw std::runtime_error(
            "Failed to initialise velocity constraint at node");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign velocity constraints to nodes
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_velocity_constraints(
    const std::vector<std::tuple<mpm::Index, unsigned, double>>&
        velocity_constraints) {
  bool status = true;
  try {
    for (const auto& velocity_constraint : velocity_constraints) {
      // Node id
      mpm::Index nid = std::get<0>(velocity_constraint);
      // Direction
      unsigned dir = std::get<1>(velocity_constraint);
      // Velocity
      double velocity = std::get<2>(velocity_constraint);

      // Apply constraint
      if (!mesh_->node(nid)->assign_velocity_constraint(dir, velocity))
        throw std::runtime_error(
            "Nodal velocity constraints assignment failed");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign nodal displacement constraints
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_displacement_constraint(
    const std::shared_ptr<FunctionBase>& dfunction, int set_id,
    const std::shared_ptr<mpm::DisplacementConstraint>& dconstraint) {
  bool status = true;
  try {
    int set_id = dconstraint->setid();
    auto nset = mesh_->nodes(set_id);
    if (nset.size() == 0)
      throw std::runtime_error(
          "Node set is empty for assignment of displacement constraints");

    unsigned dir = dconstraint->dir();
    double displacement = dconstraint->displacement();
    for (auto nitr = nset.cbegin(); nitr != nset.cend(); ++nitr) {
      if (!(*nitr)->assign_displacement_constraint(dir, displacement,
                                                   dfunction))
        throw std::runtime_error(
            "Failed to initialise displacement constraint at node");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign displacement constraints to nodes
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_displacement_constraints(
    const std::vector<std::tuple<mpm::Index, unsigned, double>>&
        displacement_constraints) {
  bool status = true;
  try {
    for (const auto& displacement_constraint : displacement_constraints) {
      // Node id
      mpm::Index nid = std::get<0>(displacement_constraint);
      // Direction
      unsigned dir = std::get<1>(displacement_constraint);
      // Displacement
      double displacement = std::get<2>(displacement_constraint);

      // Apply constraint
      if (!mesh_->node(nid)->assign_displacement_constraint(dir, displacement,
                                                            nullptr))
        throw std::runtime_error(
            "Nodal displacement constraints assignment failed");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign friction constraints to nodes
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_frictional_constraint(
    int nset_id, const std::shared_ptr<mpm::FrictionConstraint>& fconstraint) {
  bool status = true;
  try {
    int set_id = fconstraint->setid();
    auto nset = mesh_->nodes(set_id);
    if (nset.size() == 0)
      throw std::runtime_error(
          "Node set is empty for assignment of friction constraints");
    unsigned dir = fconstraint->dir();
    int sign_n = fconstraint->sign_n();
    double friction = fconstraint->friction();
    for (auto nitr = nset.cbegin(); nitr != nset.cend(); ++nitr) {
      if (!(*nitr)->assign_friction_constraint(dir, sign_n, friction))
        throw std::runtime_error(
            "Failed to initialise friction constraint at node");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign friction constraints to nodes
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_friction_constraints(
    const std::vector<std::tuple<mpm::Index, unsigned, int, double>>&
        friction_constraints) {
  bool status = true;
  try {
    for (const auto& friction_constraint : friction_constraints) {
      // Node id
      mpm::Index nid = std::get<0>(friction_constraint);
      // Direction (normal)
      unsigned dir = std::get<1>(friction_constraint);
      // Sign of normal direction
      int sign_n = std::get<2>(friction_constraint);
      // Friction
      double friction = std::get<3>(friction_constraint);

      // Apply constraint
      if (!mesh_->node(nid)->assign_friction_constraint(dir, sign_n, friction))
        throw std::runtime_error(
            "Nodal friction constraints assignment failed");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign adhesion constraints to nodes
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_adhesional_constraint(
    int nset_id, const std::shared_ptr<mpm::AdhesionConstraint>& aconstraint) {
  bool status = true;
  try {
    int set_id = aconstraint->setid();
    auto nset = mesh_->nodes(set_id);
    if (nset.size() == 0)
      throw std::runtime_error(
          "Node set is empty for assignment of adhesion constraints");
    unsigned dir = aconstraint->dir();
    int sign_n = aconstraint->sign_n();
    double adhesion = aconstraint->adhesion();
    double h_min = aconstraint->h_min();
    int nposition = aconstraint->nposition();
    for (auto nitr = nset.cbegin(); nitr != nset.cend(); ++nitr) {
      if (!(*nitr)->assign_adhesion_constraint(dir, sign_n, adhesion, h_min,
                                               nposition))
        throw std::runtime_error(
            "Failed to initialise adhesion constraint at node");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign adhesion constraints to nodes
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_adhesion_constraints(
    const std::vector<std::tuple<mpm::Index, unsigned, int, double, double,
                                 int>>& adhesion_constraints) {
  bool status = true;
  try {
    for (const auto& adhesion_constraint : adhesion_constraints) {
      // Node id
      mpm::Index nid = std::get<0>(adhesion_constraint);
      // Direction (normal)
      unsigned dir = std::get<1>(adhesion_constraint);
      // Sign of normal direction
      int sign_n = std::get<2>(adhesion_constraint);
      // Adhesion
      double adhesion = std::get<3>(adhesion_constraint);
      // Cell height for area computation
      double h_min = std::get<4>(adhesion_constraint);
      // Location of node for area computation
      int nposition = std::get<5>(adhesion_constraint);

      // Apply constraint
      if (!mesh_->node(nid)->assign_adhesion_constraint(dir, sign_n, adhesion,
                                                        h_min, nposition))
        throw std::runtime_error(
            "Nodal adhesion constraints assignment failed");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign nodal pressure constraints
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_pressure_constraint(
    const std::shared_ptr<FunctionBase>& mfunction, int set_id, unsigned phase,
    double pconstraint) {
  bool status = true;
  try {
    auto nset = mesh_->nodes(set_id);
    if (nset.size() == 0)
      throw std::runtime_error(
          "Node set is empty for assignment of pressure constraints");

#pragma omp parallel for schedule(runtime)
    for (auto nitr = nset.cbegin(); nitr != nset.cend(); ++nitr) {
      if (!(*nitr)->assign_pressure_constraint(phase, pconstraint, mfunction))
        throw std::runtime_error("Setting pressure constraint failed");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign nodal pressure constraints to nodes
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_pressure_constraints(
    const unsigned phase,
    const std::vector<std::tuple<mpm::Index, double>>& pressure_constraints) {
  bool status = true;
  try {
    for (const auto& pressure_constraint : pressure_constraints) {
      // Node id
      mpm::Index nid = std::get<0>(pressure_constraint);
      // Pressure
      double pressure = std::get<1>(pressure_constraint);

      // Apply constraint
      if (!mesh_->node(nid)->assign_pressure_constraint(phase, pressure,
                                                        nullptr))
        throw std::runtime_error(
            "Nodal pressure constraints assignment failed");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Apply absorbing constraints to nodes
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_absorbing_constraint(
    int nset_id,
    const std::shared_ptr<mpm::AbsorbingConstraint>& absorbing_constraint) {
  bool status = true;
  try {
    int set_id = nset_id;
    auto nset = mesh_->nodes(set_id);
    if (nset.size() == 0)
      throw std::runtime_error(
          "Node set is empty for application of absorbing constraints");
    unsigned dir = absorbing_constraint->dir();
    double delta = absorbing_constraint->delta();
    double h_min = absorbing_constraint->h_min();
    double a = absorbing_constraint->a();
    double b = absorbing_constraint->b();
    mpm::Position position = absorbing_constraint->position();
    if (delta >= h_min / (2 * a) and delta >= h_min / (2 * b))
      for (auto nitr = nset.cbegin(); nitr != nset.cend(); ++nitr) {
        if (!(*nitr)->apply_absorbing_constraint(dir, delta, h_min, a, b,
                                                 position))
          throw std::runtime_error(
              "Failed to apply absorbing constraint at node");
      }
    else
      throw std::runtime_error("Invalid value for delta");
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign absorbing constraints to nodes
template <unsigned Tdim>
bool mpm::Constraints<Tdim>::assign_nodal_absorbing_constraints(
    const std::vector<std::tuple<mpm::Index, unsigned, double, double, double,
                                 double, mpm::Position>>&
        absorbing_constraints) {
  bool status = true;
  try {
    for (const auto& absorbing_constraint : absorbing_constraints) {
      // Node id
      mpm::Index nid = std::get<0>(absorbing_constraint);
      // Direction
      unsigned dir = std::get<1>(absorbing_constraint);
      // Delta
      double delta = std::get<2>(absorbing_constraint);
      // h_min
      double h_min = std::get<3>(absorbing_constraint);
      // a
      double a = std::get<4>(absorbing_constraint);
      // b
      double b = std::get<5>(absorbing_constraint);
      // Position
      mpm::Position position = std::get<6>(absorbing_constraint);
      // delta check
      if (delta >= h_min / (2 * a) and delta >= h_min / (2 * b)) {
        if (position == mpm::Position::Corner or
            position == mpm::Position::Edge or
            position == mpm::Position::Face) {
          // Apply constraint
          if (!mesh_->node(nid)->apply_absorbing_constraint(dir, delta, h_min,
                                                            a, b, position))
            throw std::runtime_error(
                "Nodal absorbing constraints assignment failed");
        } else
          throw std::runtime_error("Invalid position");
      } else
        throw std::runtime_error("Invalid value for delta");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Assign absorbing constraints pointers and ids
template <unsigned Tdim>
void mpm::Constraints<Tdim>::assign_absorbing_id_ptr(
    unsigned nset_id,
    std::shared_ptr<mpm::AbsorbingConstraint>& absorbing_constraint) {
  this->absorbing_constraint_.emplace_back(absorbing_constraint);
  this->absorbing_nset_id_.emplace_back(nset_id);
}
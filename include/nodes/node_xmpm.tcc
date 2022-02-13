//! Initialise xmpm nodal variables
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
mpm::NodeXMPM<Tdim, Tdof, Tnphases>::NodeXMPM(Index id, const VectorDim& coord)
    : Node<Tdim, Tdof, Tnphases>(id, coord) {
  this->initialise();
  // Specific variables for xmpm
}

//! Initialise nodal properties
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::initialise() noexcept {

  mpm::Node<Tdim, Tdof, Tnphases>::initialise();
  enrich_type_ = mpm::NodeEnrichType::regular;

  mass_enrich_.setZero();

  momentum_enrich_.setZero();
  internal_force_enrich_.setZero();
  external_force_enrich_.setZero();

  for (unsigned i = 0; i < 2; i++) {
    friction_coef_[i] = 0;
    cohesion_[i] = 0;
    cohesion_area_[i] = 0;
  }
  for (unsigned i = 0; i < levelset_phi_.size(); i++) levelset_phi_[i] = 0;
  for (unsigned i = 0; i < normal_.size(); i++) normal_[i].setZero();
  for (unsigned i = 0; i < contact_detection_.size(); i++)
    contact_detection_[i] = false;
}

//! Compute momentum for discontinuity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
bool mpm::NodeXMPM<Tdim, Tdof, Tnphases>::compute_momentum_discontinuity(
    unsigned phase, double dt) {
  momentum_.col(phase) =
      momentum_.col(phase) + (this->internal_force(phase).col(phase) +
                              this->external_force(phase).col(phase)) *
                                 dt;
  if (enrich_type_ != mpm::NodeEnrichType::regular) {
    for (unsigned int i = 0; i < 3; i++)
      momentum_enrich_.col(i) +=
          (internal_force_enrich_.col(i) + external_force_enrich_.col(i)) * dt;
  }

  // Apply velocity constraints, which also sets acceleration to 0,
  // when velocity is set.
  this->apply_velocity_constraints();

  this->self_contact_discontinuity(dt);

  this->apply_velocity_constraints();

  return true;
}

//! Compute momentum for discontinuity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
bool mpm::NodeXMPM<Tdim, Tdof, Tnphases>::
    compute_momentum_discontinuity_cundall(unsigned phase, double dt,
                                           double damping_factor) {
  const double tolerance = 1.0E-15;

  if (enrich_type_ == mpm::NodeEnrichType::regular) {

    if (mass_.col(phase)(0, 0) > tolerance) {

      auto unbalanced_force =
          this->external_force_.col(phase) + this->internal_force_.col(phase);
      this->external_force_.col(phase) -=
          damping_factor * unbalanced_force.norm() *
          this->momentum_.col(phase).normalized();
    }

  } else {

    // // obtain the enriched values of enriched nodes
    // double mass_enrich = property_handle_->property(
    //     "mass_enrich", discontinuity_prop_id_, 0, 1)(0, 0);

    // // mass for different sides
    // double mass_p = mass_(phase) + mass_enrich;
    // double mass_n = mass_(phase) - mass_enrich;

    // Eigen::Matrix<double, Tdim, 1> momenta_enrich =
    // property_handle_->property(
    //     "momenta_enrich", discontinuity_prop_id_, 0, Tdim);
    // Eigen::Matrix<double, Tdim, 1> unbalanced_force =
    //     this->external_force_.col(phase) + this->internal_force_.col(phase);
    // Eigen::Matrix<double, Tdim, 1> unbalanced_force_enrich =
    //     property_handle_->property("internal_force_enrich",
    //                                discontinuity_prop_id_, 0, Tdim) +
    //     property_handle_->property("external_force_enrich",
    //                                discontinuity_prop_id_, 0, Tdim);
    // // neet to be fixed
    // Eigen::Matrix<double, Tdim, 1> damp_force_p =
    //     -damping_factor * (unbalanced_force + unbalanced_force_enrich).norm()
    //     * (this->momentum_.col(phase) + momenta_enrich).normalized();

    // if (mass_p < tolerance) damp_force_p.setZero();

    // Eigen::Matrix<double, Tdim, 1> damp_force_n =
    //     -damping_factor * (unbalanced_force - unbalanced_force_enrich).norm()
    //     * (this->momentum_.col(phase) - momenta_enrich).normalized();

    // if (mass_n < tolerance) damp_force_n.setZero();
    // this->external_force_.col(phase) +=
    //     0.5 * (damp_force_p + damp_force_n);

    // property_handle_->update_property(
    //     "external_force_enrich", discontinuity_prop_id_, 0,
    //     0.5 * (damp_force_p - damp_force_n), Tdim);
  }

  compute_momentum_discontinuity(phase, dt);

  return true;
}

//! Determine node type
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::determine_node_type(int dis_id) {
  const double tolerance = 1.0E-15;
  unsigned phase = 0;

  // unenriched
  if (mass_(phase) + mass_h_ < tolerance || mass_(phase) - mass_h_ < tolerance)
    return;

  assign_discontinuity_enrich(true, dis_id);
}

//! Apply velocity constraints for discontinuity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::apply_velocity_constraints() {

  // Set velocity constraint
  for (const auto& constraint : this->velocity_constraints_) {
    // Direction value in the constraint (0, Dim * Nphases)
    const unsigned dir = constraint.first;
    // Direction: dir % Tdim (modulus)
    const auto direction = static_cast<unsigned>(dir % Tdim);
    // Phase: Integer value of division (dir / Tdim)
    const auto phase = static_cast<unsigned>(dir / Tdim);

    // Velocity constraints are applied on Cartesian boundaries
    this->momentum_(direction, phase) = this->mass(phase) * constraint.second;

    // Set acceleration to 0 in direction of velocity constraint
    this->internal_force_(direction, phase) = 0;
    this->external_force_(direction, phase) = 0;

    if (enrich_type_ == mpm::NodeEnrichType::regular) continue;

    for (int i = 0; i < 3; i++) {
      momentum_enrich_.col(i)[direction] = mass_enrich_[i] * constraint.second;
      internal_force_enrich_.col(i)[direction] = 0;
      external_force_enrich_.col(i)[direction] = 0;
    }
  }
}

//! Apply velocity filter
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::apply_velocity_filter() {

  const double tolerance = 1.E-16;

  for (unsigned i = 0; i < 3; i++) {
    if (std::abs(mass_enrich_[i]) > tolerance) continue;
    mass_enrich_[i] = 0;
    // momentum_enrich_.col(i) = Eigen::Matrix<double, Tdim, 1>::Zero();
  }
  for (unsigned phase = 0; phase < Tnphases; ++phase) {
    if (mass_(phase) > tolerance) continue;
    mass_(phase) = 0;
    momentum_.col(phase) = Eigen::Matrix<double, Tdim, 1>::Zero();
  }
}

//! Initialise shared pointer to nodal properties pool for discontinuity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::
    initialise_discontinuity_property_handle(
        unsigned prop_id,
        std::shared_ptr<mpm::NodalProperties> property_handle) {
  // the property handle and the property id is set in the node
  this->property_handle_ = property_handle;
  this->discontinuity_prop_id_ = prop_id;
}

//! Update nodal property at the nodes from particle for discontinuity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::update_discontinuity_property(
    bool update, const std::string& property,
    const Eigen::MatrixXd& property_value, unsigned discontinuity_id,
    unsigned nprops) {
  // Update property
  node_mutex_.lock();
  property_handle_->update_property(property, discontinuity_prop_id_,
                                    discontinuity_id, property_value, nprops);
  node_mutex_.unlock();
}

//! Assign nodal property at the nodes from particle for discontinuity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::assign_discontinuity_property(
    bool update, const std::string& property,
    const Eigen::MatrixXd& property_value, unsigned discontinuity_id,
    unsigned nprops) {
  // assign property
  node_mutex_.lock();
  property_handle_->assign_property(property, discontinuity_prop_id_,
                                    discontinuity_id, property_value, nprops);
  node_mutex_.unlock();
}

// Return data in the nodal properties map at a specific index
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
Eigen::MatrixXd mpm::NodeXMPM<Tdim, Tdof, Tnphases>::discontinuity_property(
    const std::string& property, unsigned nprops) {
  // Const pointer to location of property: node_id * nprops x mat_id
  auto property_value =
      property_handle_->property(property, discontinuity_prop_id_, 0, nprops);

  return property_value;
}

//! Apply self-contact of the discontinuity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::self_contact_discontinuity(
    double dt) {

  if (enrich_type_ == mpm::NodeEnrichType::regular) return;

  //  single phase for solid
  unsigned phase = 0;
  const double tolerance = 1.0E-16;

  if (enrich_type_ == mpm::NodeEnrichType::single_enriched) {

    Eigen::Matrix<double, Tdim, 1> normal_vector =
        normal_[discontinuity_id_[0]];

    if (!contact_detection_[0]) return;

    // mass for different sides, _p = _positive, _n = _negative
    double mass_p = mass_(phase) + mass_enrich_[0];
    double mass_n = mass_(phase) - mass_enrich_[0];

    if (mass_p < tolerance || mass_n < tolerance) return;

    // velocity for different sides
    auto velocity_p = (momentum_.col(phase) + momentum_enrich_.col(0)) / mass_p;
    auto velocity_n = (momentum_.col(phase) - momentum_enrich_.col(0)) / mass_n;

    // relative normal velocity
    if ((velocity_p - velocity_n).col(phase).dot(normal_vector) >= 0) return;

    // the contact momentum, force vector for sticking contact
    auto momentum_contact = (mass_enrich_[0] * momentum_.col(phase) -
                             mass_(phase) * momentum_enrich_.col(0)) /
                            mass_(phase);
    auto force_contact = momentum_contact / dt;

    if (friction_coef_[0] < 0) {
      momentum_enrich_.col(0) += momentum_contact.col(phase);
      external_force_enrich_.col(0) += force_contact.col(phase);
    } else {
      // the contact momentum, force value for sticking contact at normal
      // direction
      double momentum_contact_norm =
          momentum_contact.col(phase).dot(normal_vector);
      double force_contact_norm = momentum_contact_norm / dt;

      // the cohesion at nodes
      double cohesion = cohesion_[0];

      double max_friction_force = friction_coef_[0] * abs(force_contact_norm) +
                                  2 * cohesion * cohesion_area_[0];

      // the contact momentum, force vector for sticking contact at tangential
      // direction
      auto momentum_tangential =
          momentum_contact.col(phase) - momentum_contact_norm * normal_vector;
      auto force_tangential = momentum_tangential / dt;

      // the friction force magnitude
      double force_tangential_value = force_tangential.norm();

      double force_friction = force_tangential_value < max_friction_force
                                  ? force_tangential_value
                                  : max_friction_force;

      // adjust the momentum and force
      momentum_enrich_.col(0) +=
          momentum_contact_norm * normal_vector +
          force_friction * force_tangential.col(phase).normalized() * dt;
      external_force_enrich_.col(0) +=
          force_contact_norm * normal_vector +
          force_friction * force_tangential.col(phase).normalized();
    }
  } else if (enrich_type_ == mpm::NodeEnrichType::double_enriched) {

    int itr_max = 100;
    double itr_tol = 1e-6;
    double itr_error;
    Eigen::Matrix<int, 4, 2> flag;

    flag << -1, -1, 1, -1, -1, 1, 1, 1;

    Eigen::Matrix<double, 4, 1> mass;
    // the mass of 4 different parts
    for (int i = 0; i < 4; i++) {
      mass[i] = mass_(phase) + flag(i, 0) * mass_enrich_[0] +
                flag(i, 1) * mass_enrich_[1] +
                flag(i, 0) * flag(i, 1) * mass_enrich_[2];
    }

    for (int itr = 0; itr < itr_max; ++itr) {

      double max_update_normal_p = 0;
      int update_normal = 0;
      Eigen::Matrix<double, Tdim, 1> update_p;
      // normal vector of two discontinuities
      Eigen::Matrix<double, Tdim, 2> normal_vector;
      normal_vector.col(0) = normal_[discontinuity_id_[0]];
      normal_vector.col(1) = normal_[discontinuity_id_[1]];

      // momentum of 4 different parts
      Eigen::Matrix<double, Tdim, 4> momentum;
      for (int i = 0; i < 4; i++) {
        momentum.col(i) = momentum_.col(phase) +
                          flag(i, 0) * momentum_enrich_.col(0) +
                          flag(i, 1) * momentum_enrich_.col(1) +
                          flag(i, 0) * flag(i, 1) * momentum_enrich_.col(2);
      }
      Eigen::Matrix<double, Tdim, 4> velocitynew;
      for (int i = 0; i < 4; i++) {
        velocitynew.col(i) = momentum.col(i) / mass(i);
      }
      int k = -1;
      itr_error = 0;
      double coef_couple = 0;
      for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 4; j++) {
          k++;
          if (!contact_detection_[k]) continue;
          // loop for 2 normal directions
          for (int n = 0; n < 2; n++) {
            if (flag(i, n) * flag(j, n) > 0) continue;

            if (mass[i] < tolerance || mass[j] < tolerance) continue;
            Eigen::Matrix<double, Tdim, 2> velocity;

            // velocity for different sides
            velocity.col(0) = momentum.col(i) / mass[i];
            velocity.col(1) = momentum.col(j) / mass[j];
            // relative normal velocity
            if ((velocity.col(0) - velocity.col(1))
                        .col(phase)
                        .dot(normal_vector.col(n)) *
                    flag(i, n) >=
                0)
              continue;

            Eigen::Matrix<double, 1, 4> a;
            a << 1, flag(i, 0), flag(i, 1), flag(i, 0) * flag(i, 1);
            Eigen::Matrix<double, 1, 4> b;
            b << 1, flag(j, 0), flag(j, 1), flag(j, 0) * flag(j, 1);

            Eigen::Matrix<double, 4, 1> m;
            m << mass_(phase), mass_enrich_[0], mass_enrich_[1],
                mass_enrich_[2];

            Eigen::Matrix<double, Tdim, 4> p;
            p << momentum_.col(phase), momentum_enrich_.col(0),
                momentum_enrich_.col(1), momentum_enrich_.col(2);

            Eigen::Matrix<double, 4, 4> bt_a = b.transpose() * a;
            Eigen::Matrix<double, 4, 4> at_b = a.transpose() * b;

            Eigen::Matrix<double, 1, 4> coef =
                m.transpose() * (bt_a - at_b).transpose();
            Eigen::Matrix<double, Tdim, 1> deltap;
            Eigen::Matrix<double, Tdim, 1> fvalue;
            for (unsigned dir = 0; dir < Tdim; dir++) {
              Eigen::Matrix<double, 4, 1> p;
              p << momentum_.col(phase)[dir], momentum_enrich_.col(0)[dir],
                  momentum_enrich_.col(1)[dir], momentum_enrich_.col(2)[dir];
              fvalue[dir] = m.transpose() * (at_b - bt_a).transpose() * p;
            }
            if (n == 0)
              deltap =
                  fvalue /
                  (coef[1] + coef[3] * 0.5 * (flag(i, 1 - n) + flag(j, 1 - n)));
            else
              deltap =
                  fvalue /
                  (coef[2] + coef[3] * 0.5 * (flag(i, 1 - n) + flag(j, 1 - n)));
            double updatep = deltap.transpose() * normal_vector.col(n);
            if (std::abs(updatep) > max_update_normal_p) {
              max_update_normal_p = std::abs(updatep);
              update_normal = n;
              update_p = deltap;
              // to do
              itr_error = std::abs(updatep);
              coef_couple = 0.5 * (flag(i, 1 - n) + flag(j, 1 - n));
            }
          }
        }

      if (itr_error < itr_tol) break;

      if (friction_coef_[update_normal] < 0) {
        momentum_enrich_.col(update_normal) += update_p;
        momentum_enrich_.col(2) += coef_couple * update_p;
        external_force_enrich_.col(update_normal) += update_p / dt;
        external_force_enrich_.col(2) += coef_couple * update_p / dt;
      } else {
        // the contact momentum, force value for sticking contact at normal
        // direction
        Eigen::Matrix<double, Tdim, 1> momentum_contact = update_p;
        double momentum_contact_norm =
            momentum_contact.col(phase).dot(normal_vector.col(update_normal));
        double force_contact_norm = momentum_contact_norm / dt;

        // the cohesion at nodes
        // double cohesion = cohesion_[0];

        double max_friction_force =
            friction_coef_[update_normal] * abs(force_contact_norm);
        //  + 2 * cohesion * cohesion_area;

        // // the contact momentum, force vector for sticking contact at
        // tangential
        // // direction
        auto momentum_tangential =
            momentum_contact.col(phase) -
            momentum_contact_norm * normal_vector.col(update_normal);
        auto force_tangential = momentum_tangential / dt;

        // // the friction force magnitude
        double force_tangential_value = force_tangential.norm();

        double force_friction = force_tangential_value < max_friction_force
                                    ? force_tangential_value
                                    : max_friction_force;

        momentum_enrich_.col(update_normal) +=
            momentum_contact_norm * normal_vector.col(update_normal) +
            force_friction * force_tangential.col(phase).normalized() * dt;
        momentum_enrich_.col(2) +=
            coef_couple * momentum_contact_norm *
                normal_vector.col(update_normal) +
            force_friction * force_tangential.col(phase).normalized() * dt;
        external_force_enrich_.col(update_normal) +=
            force_contact_norm * normal_vector.col(update_normal) +
            force_friction * force_tangential.col(phase).normalized();
        external_force_enrich_.col(2) +=
            coef_couple * force_contact_norm *
                normal_vector.col(update_normal) +
            force_friction * force_tangential.col(phase).normalized();

        // // adjust the momentum and force
        // momentum_enrich_.col(0) +=
        //     momentum_contact_norm * normal_vector +
        //     force_friction * force_tangential.col(phase).normalized() * dt;
        // external_force_enrich_.col(0) +=
        //     force_contact_norm * normal_vector +
        //     force_friction * force_tangential.col(phase).normalized();
      }
    }
  }
}

//! Add a cell id
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::add_cell_id(Index id) {
  cells_.emplace_back(id);
}

//! Assign whether the node is enriched
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::assign_discontinuity_enrich(
    bool enrich, unsigned dis_id) {
  if (enrich) {
    if (enrich_type_ == mpm::NodeEnrichType::regular) {
      enrich_type_ = mpm::NodeEnrichType::single_enriched;
      discontinuity_id_[0] = dis_id;
    } else if (enrich_type_ == mpm::NodeEnrichType::single_enriched) {
      enrich_type_ = mpm::NodeEnrichType::double_enriched;
      discontinuity_id_[1] = dis_id;
    } else {
      console_->error("Multiple discontinuities are detected at the node");
    }
  } else {
    if (enrich_type_ == mpm::NodeEnrichType::single_enriched) {
      enrich_type_ = mpm::NodeEnrichType::regular;

    } else if (enrich_type_ == mpm::NodeEnrichType::double_enriched) {
      enrich_type_ = mpm::NodeEnrichType::single_enriched;
    }
  }
}
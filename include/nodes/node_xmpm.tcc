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

  // Apply contact treatment
  this->self_contact_discontinuity(dt);

  // Re-apply velocity constraints after contact is considered to properly
  // enforced boundary conditions
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
      const auto unbalanced_force =
          this->external_force_.col(phase) + this->internal_force_.col(phase);
      this->external_force_.col(phase) -=
          damping_factor * unbalanced_force.norm() *
          this->momentum_.col(phase).normalized();
    }
  } else {
    // TODO: yliang to do list 1
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

//! Determine node type from mass and enriched mass
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::determine_node_type(int dis_id) {
  const double tolerance = std::numeric_limits<double>::epsilon();
  unsigned phase = 0;

  // For unenriched nodes
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

    // Continue if it is regular nodes
    if (enrich_type_ == mpm::NodeEnrichType::regular) continue;

    // Specific for enrich nodes
    for (int i = 0; i < 3; i++) {
      momentum_enrich_.col(i)[direction] = mass_enrich_[i] * constraint.second;
      internal_force_enrich_.col(i)[direction] = 0;
      external_force_enrich_.col(i)[direction] = 0;
    }
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

  // Regular nodes
  if (enrich_type_ == mpm::NodeEnrichType::regular) return;

  // Single phase
  const unsigned phase = mpm::NodePhase::NSinglePhase;
  const double tolerance = 1.0E-16;

  // Contact treatment of single discontinuity
  if (enrich_type_ == mpm::NodeEnrichType::single_enriched) {
    // Exit if not in contact
    if (!contact_detection_[0]) return;

    // Get normal vector
    const Eigen::Matrix<double, Tdim, 1>& normal_vector =
        normal_[discontinuity_id_[0]];

    // Mass for different sides, _p = _positive, _n = _negative
    const double mass_p = mass_(phase) + mass_enrich_[0];
    const double mass_n = mass_(phase) - mass_enrich_[0];

    // Exit if nodal masses are zero
    if (mass_p < tolerance || mass_n < tolerance) return;

    // Velocity for different sides
    const auto velocity_p =
        (momentum_.col(phase) + momentum_enrich_.col(0)) / mass_p;
    const auto velocity_n =
        (momentum_.col(phase) - momentum_enrich_.col(0)) / mass_n;

    // Relative normal velocity
    if ((velocity_p - velocity_n).col(phase).dot(normal_vector) >= 0) return;

    // Contact momentum, force vector for sticking contact
    const auto momentum_contact = (mass_enrich_[0] * momentum_.col(phase) -
                                   mass_(phase) * momentum_enrich_.col(0)) /
                                  mass_(phase);
    const auto force_contact = momentum_contact / dt;

    // For sticking contact
    if (friction_coef_[0] < 0) {
      momentum_enrich_.col(0).noalias() += momentum_contact.col(phase);
      external_force_enrich_.col(0).noalias() += force_contact.col(phase);
    }
    // For frictional contact
    else {
      // Contact momentum, force values for sticking contact at normal
      // direction
      const double momentum_contact_norm =
          momentum_contact.col(phase).dot(normal_vector);
      const double force_contact_norm = momentum_contact_norm / dt;

      // Cohesion at nodes
      const double cohesion = cohesion_[0];

      // Maximum friction force (assuming Coulomb's law)
      const double max_friction_force =
          friction_coef_[0] * abs(force_contact_norm) +
          2 * cohesion * cohesion_area_[0];

      // Contact momentum, force vector for sticking contact at tangential
      // direction
      const auto momentum_tangential =
          momentum_contact.col(phase) - momentum_contact_norm * normal_vector;
      const auto force_tangential = momentum_tangential / dt;
      const double force_tangential_value = force_tangential.norm();

      // Compare tangential contact with maximum threshold
      const double force_friction =
          std::min(force_tangential_value, max_friction_force);

      // Compute contact force and momentum
      momentum_enrich_.col(0).noalias() +=
          momentum_contact_norm * normal_vector +
          force_friction * force_tangential.normalized() * dt;
      external_force_enrich_.col(0).noalias() +=
          force_contact_norm * normal_vector +
          force_friction * force_tangential.normalized();
    }
  }
  // Contact treatment of two discontinuity
  else if (enrich_type_ == mpm::NodeEnrichType::double_enriched) {
    const int itr_max = 100;
    const double itr_tol = 1e-6;
    double itr_error;

    // Flag to denote the sign of four partitioned regions
    Eigen::Matrix<int, 4, 2> flag;
    flag << -1, -1, 1, -1, -1, 1, 1, 1;

    // Mass of 4 different regions
    Eigen::Matrix<double, 4, 1> mass;
    for (int i = 0; i < 4; i++) {
      mass[i] = mass_(phase) + flag(i, 0) * mass_enrich_[0] +
                flag(i, 1) * mass_enrich_[1] +
                flag(i, 0) * flag(i, 1) * mass_enrich_[2];
    }

    // Perform iteration to minimize contact moment
    for (int itr = 0; itr < itr_max; ++itr) {
      double max_update_normal_p = 0;
      int update_normal = 0;
      Eigen::Matrix<double, Tdim, 1> update_p;

      // Normal vectors of two discontinuities
      Eigen::Matrix<double, Tdim, 2> normal_vector;
      normal_vector.col(0) = normal_[discontinuity_id_[0]];
      normal_vector.col(1) = normal_[discontinuity_id_[1]];

      // Momenta of 4 different parts
      Eigen::Matrix<double, Tdim, 4> momentum;
      for (int i = 0; i < 4; i++) {
        momentum.col(i) = momentum_.col(phase) +
                          flag(i, 0) * momentum_enrich_.col(0) +
                          flag(i, 1) * momentum_enrich_.col(1) +
                          flag(i, 0) * flag(i, 1) * momentum_enrich_.col(2);
      }

      // Velocity at different parts
      Eigen::Matrix<double, Tdim, 4> velocitynew;
      for (int i = 0; i < 4; i++) {
        velocitynew.col(i) = momentum.col(i) / mass(i);
      }

      // Permutation index k
      int k = -1;
      itr_error = 0;
      double coef_couple = 0;

      // Loop over different parts: i (current), j(neighbour)
      for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 4; j++) {
          k++;

          // Exit if it is not contacting
          if (!contact_detection_[k]) continue;

          // Loop for 2 normal directions
          for (int n = 0; n < 2; n++) {
            // Check if regions are in the same side of discontinuity n
            if (flag(i, n) * flag(j, n) > 0) continue;

            // Exit if nodal masses of different parts are zero
            if (mass[i] < tolerance || mass[j] < tolerance) continue;

            // Velocity for different sides
            Eigen::Matrix<double, Tdim, 2> velocity;
            velocity.col(0) = momentum.col(i) / mass[i];
            velocity.col(1) = momentum.col(j) / mass[j];

            // Check relative normal velocity
            if ((velocity.col(0) - velocity.col(1)).dot(normal_vector.col(n)) *
                    flag(i, n) >=
                0)
              continue;

            // Variables associated with flags
            Eigen::Matrix<double, 1, 4> a;
            a << 1, flag(i, 0), flag(i, 1), flag(i, 0) * flag(i, 1);
            Eigen::Matrix<double, 1, 4> b;
            b << 1, flag(j, 0), flag(j, 1), flag(j, 0) * flag(j, 1);

            // Mass as a vector
            Eigen::Matrix<double, 4, 1> m;
            m << mass_(phase), mass_enrich_[0], mass_enrich_[1],
                mass_enrich_[2];

            // Momentum as a vector
            Eigen::Matrix<double, Tdim, 4> p;
            p << momentum_.col(phase), momentum_enrich_.col(0),
                momentum_enrich_.col(1), momentum_enrich_.col(2);

            // Matrix of flags combination
            const Eigen::Matrix<double, 4, 4> bt_a = b.transpose() * a;
            const Eigen::Matrix<double, 4, 4> at_b = a.transpose() * b;

            const Eigen::Matrix<double, 1, 4> coef =
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

            // Check maximum momentum
            const double updatep = deltap.transpose() * normal_vector.col(n);
            if (std::abs(updatep) > max_update_normal_p) {
              max_update_normal_p = std::abs(updatep);
              update_normal = n;
              update_p = deltap;
              itr_error = std::abs(updatep);
              coef_couple = 0.5 * (flag(i, 1 - n) + flag(j, 1 - n));
            }
          }
        }
      }

      // Exit loop if the error is small enough
      if (itr_error < itr_tol) break;

      // Update momentum and force by the largest momentum increment
      // For sticking contact
      if (friction_coef_[update_normal] < 0) {
        momentum_enrich_.col(update_normal) += update_p;
        momentum_enrich_.col(2) += coef_couple * update_p;
        external_force_enrich_.col(update_normal) += update_p / dt;
        external_force_enrich_.col(2) += coef_couple * update_p / dt;
      }
      // For cohesive-frictional contact
      else {
        // Contact momentum, force value for sticking contact at normal
        // direction
        const Eigen::Matrix<double, Tdim, 1> momentum_contact = update_p;
        const double momentum_contact_norm =
            momentum_contact.dot(normal_vector.col(update_normal));
        const double force_contact_norm = momentum_contact_norm / dt;

        // Cohesion and max friction force
        const double cohesion = cohesion_[update_normal];
        const double cohesion_area = cohesion_area_[update_normal];
        const double max_friction_force =
            friction_coef_[update_normal] * abs(force_contact_norm) +
            2 * cohesion * cohesion_area;

        // Tangential contact momentum and force
        const auto momentum_tangential =
            momentum_contact -
            momentum_contact_norm * normal_vector.col(update_normal);
        const auto force_tangential = momentum_tangential / dt;
        const double force_tangential_value = force_tangential.norm();

        // Frictional force
        const double force_friction =
            force_tangential_value < max_friction_force ? force_tangential_value
                                                        : max_friction_force;

        // Update total momentum and force
        momentum_enrich_.col(update_normal) +=
            momentum_contact_norm * normal_vector.col(update_normal) +
            force_friction * force_tangential.normalized() * dt;
        momentum_enrich_.col(2) +=
            coef_couple * momentum_contact_norm *
                normal_vector.col(update_normal) +
            force_friction * force_tangential.normalized() * dt;
        external_force_enrich_.col(update_normal) +=
            force_contact_norm * normal_vector.col(update_normal) +
            force_friction * force_tangential.normalized();
        external_force_enrich_.col(2) +=
            coef_couple * force_contact_norm *
                normal_vector.col(update_normal) +
            force_friction * force_tangential.normalized();
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
  node_mutex_.lock();
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
  node_mutex_.unlock();
}
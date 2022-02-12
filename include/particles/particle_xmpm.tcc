//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleXMPM<Tdim>::ParticleXMPM(Index id, const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {
  this->initialise();
  // Logger
  std::string logger =
      "particlexmpm" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);

  // to do:
  // if (coordinates_[1] < 0.5) {
  //   velocity_[1] = 1;
  // } else {
  //   velocity_[1] = -1;
  // }

  // if (coordinates_[0] > 0.5) {
  //   velocity_[0] = -1;
  // } else
  //   velocity_[0] = 1;

  // if (coordinates_[0] < 0.5)
  //   levelset_phi_[0] = -0.25;
  // else
  //   levelset_phi_[0] = 0.25;
  // if (coordinates_[1] < 0.5)
  //   levelset_phi_[1] = -0.25;
  // else
  //   levelset_phi_[1] = 0.25;
}

//! Initialise particle data from pod
template <unsigned Tdim>
bool mpm::ParticleXMPM<Tdim>::initialise_particle(PODParticle& particle) {
  bool status = mpm::Particle<Tdim>::initialise_particle(particle);
  // auto xmpm_particle = reinterpret_cast<PODParticleXMPM*>(&particle);
  // to do: levelset_phi
  // this->levelset_phi_[0] = xmpm_particle->levelset_phi;

  return status;
}

//! Return particle data as POD
template <unsigned Tdim>
// cppcheck-suppress *
std::shared_ptr<void> mpm::ParticleXMPM<Tdim>::pod() const {

  auto particle_data = std::make_shared<mpm::PODParticleXMPM>();

  Eigen::Vector3d coordinates;
  coordinates.setZero();
  for (unsigned j = 0; j < Tdim; ++j) coordinates[j] = this->coordinates_[j];

  Eigen::Vector3d displacement;
  displacement.setZero();
  for (unsigned j = 0; j < Tdim; ++j) displacement[j] = this->displacement_[j];

  Eigen::Vector3d velocity;
  velocity.setZero();
  for (unsigned j = 0; j < Tdim; ++j) velocity[j] = this->velocity_[j];

  Eigen::Vector3d acceleration;
  acceleration.setZero();
  for (unsigned j = 0; j < Tdim; ++j) acceleration[j] = this->acceleration_[j];

  // Particle local size
  Eigen::Vector3d nsize;
  nsize.setZero();
  Eigen::VectorXd size = this->natural_size();
  for (unsigned j = 0; j < Tdim; ++j) nsize[j] = size[j];

  Eigen::Matrix<double, 6, 1> stress = this->stress_;

  Eigen::Matrix<double, 6, 1> strain = this->strain_;

  particle_data->id = this->id();
  particle_data->mass = this->mass();
  particle_data->volume = this->volume();
  particle_data->pressure =
      (state_variables_[mpm::ParticlePhase::Solid].find("pressure") !=
       state_variables_[mpm::ParticlePhase::Solid].end())
          ? state_variables_[mpm::ParticlePhase::Solid].at("pressure")
          : 0.;

  particle_data->coord_x = coordinates[0];
  particle_data->coord_y = coordinates[1];
  particle_data->coord_z = coordinates[2];

  particle_data->displacement_x = displacement[0];
  particle_data->displacement_y = displacement[1];
  particle_data->displacement_z = displacement[2];

  particle_data->nsize_x = nsize[0];
  particle_data->nsize_y = nsize[1];
  particle_data->nsize_z = nsize[2];

  particle_data->velocity_x = velocity[0];
  particle_data->velocity_y = velocity[1];
  particle_data->velocity_z = velocity[2];

  particle_data->acceleration_x = acceleration[0];
  particle_data->acceleration_y = acceleration[1];
  particle_data->acceleration_z = acceleration[2];

  particle_data->stress_xx = stress[0];
  particle_data->stress_yy = stress[1];
  particle_data->stress_zz = stress[2];
  particle_data->tau_xy = stress[3];
  particle_data->tau_yz = stress[4];
  particle_data->tau_xz = stress[5];

  particle_data->strain_xx = strain[0];
  particle_data->strain_yy = strain[1];
  particle_data->strain_zz = strain[2];
  particle_data->gamma_xy = strain[3];
  particle_data->gamma_yz = strain[4];
  particle_data->gamma_xz = strain[5];

  particle_data->epsilon_v = this->volumetric_strain_centroid_;

  particle_data->status = this->status();

  particle_data->cell_id = this->cell_id();

  particle_data->material_id = this->material_id();

  // Write state variables
  if (this->material() != nullptr) {
    particle_data->nstate_vars =
        state_variables_[mpm::ParticlePhase::Solid].size();
    if (state_variables_[mpm::ParticlePhase::Solid].size() > 20)
      throw std::runtime_error("# of state variables cannot be more than 20");
    unsigned i = 0;
    auto state_variables = (this->material())->state_variables();
    for (const auto& state_var : state_variables) {
      particle_data->svars[i] =
          state_variables_[mpm::ParticlePhase::Solid].at(state_var);
      ++i;
    }
  }

  // XMPM
  // particle_data->levelset_phi = levelset_phi_[0];
  return particle_data;
}

// Initialise particle properties
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::initialise() {

  du_dx_.setZero();

  this->scalar_properties_["levelsetf"] = [&]() {
    if (levelset_phi_.size() < 1) return 0.;
    return levelset_phi_[0];
  };
  this->scalar_properties_["levelsets"] = [&]() {
    if (levelset_phi_.size() < 2) return 0.;
    return levelset_phi_[1];
  };

  // this->scalar_properties_["levelsett"] = [&]() { return levelset_phi_[2]; };
  this->scalar_properties_["minimum_acoustic_eigenvalue"] = [&]() {
    return minimum_acoustic_eigenvalue_;
  };
  this->scalar_properties_["discontinuity_angle"] = [&]() {
    return discontinuity_angle_;
  };
}

//! Map particle mass and momentum to nodes
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::map_mass_momentum_to_nodes() noexcept {
  // Check if particle mass is set
  assert(mass_ != std::numeric_limits<double>::max());

  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_mass(true, mpm::ParticlePhase::Solid,
                           mass_ * shapefn_[i]);
    nodes_[i]->update_momentum(true, mpm::ParticlePhase::Solid,
                               mass_ * shapefn_[i] * velocity_);
    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::regular) continue;
    auto discontinuity_id = nodes_[i]->discontinuity_id();
    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::single_enriched) {

      Eigen::Matrix<double, 3, 1> mass_enrich;
      mass_enrich.setZero();
      mass_enrich[0] =
          sgn(levelset_phi_[discontinuity_id[0]]) * mass_ * shapefn_[i];

      Eigen::Matrix<double, Tdim, 3> momentum_enrich;
      momentum_enrich.setZero();
      momentum_enrich.col(0) = velocity_ * mass_enrich[0];

      nodes_[i]->update_mass_enrich(mass_enrich);
      nodes_[i]->update_momentum_enrich(momentum_enrich);
    } else if (nodes_[i]->enrich_type() ==
               mpm::NodeEnrichType::double_enriched) {

      Eigen::Matrix<double, 3, 1> mass_enrich;
      mass_enrich.setZero();
      mass_enrich[0] =
          sgn(levelset_phi_[discontinuity_id[0]]) * mass_ * shapefn_[i];
      mass_enrich[1] =
          sgn(levelset_phi_[discontinuity_id[1]]) * mass_ * shapefn_[i];
      mass_enrich[2] = sgn(levelset_phi_[discontinuity_id[0]]) *
                       sgn(levelset_phi_[discontinuity_id[1]]) * mass_ *
                       shapefn_[i];

      Eigen::Matrix<double, Tdim, 3> momentum_enrich;
      momentum_enrich.setZero();
      for (unsigned j = 0; j < 3; j++) {
        momentum_enrich.col(j) = velocity_ * mass_enrich[j];
      }

      nodes_[i]->update_mass_enrich(mass_enrich);
      nodes_[i]->update_momentum_enrich(momentum_enrich);
    }
  }
}

//! Map particle mass to nodes
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::map_mass_to_nodes() noexcept {
  // Check if particle mass is set
  assert(mass_ != std::numeric_limits<double>::max());

  // Map mass and momentum to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_mass(true, mpm::ParticlePhase::Solid,
                           mass_ * shapefn_[i]);
  }
}

//! Map particle mass*h to nodes
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::map_mass_h_to_nodes(unsigned dis_id) noexcept {
  // Check if particle mass is set
  assert(mass_ != std::numeric_limits<double>::max());

  // Map mass and momentum to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_mass_h(mass_ * shapefn_[i] * sgn(levelset_phi_[dis_id]));
  }
}

//! Map particle levelset to nodes
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::map_levelset_to_nodes(unsigned dis_id) noexcept {
  double volume_node;
  // Map levelset to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    volume_node = nodes_[i]->volume(mpm::ParticlePhase::Solid);
    nodes_[i]->update_levelset_phi(
        levelset_phi_[dis_id] * volume_ * shapefn_[i] / volume_node, dis_id);
  }
}

//! Map particle friction_coef_to_nodes
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::map_friction_coef_to_nodes(
    double discontinuity_friction_coef) noexcept {
  double volume_node;

  double friction_coef_scalar = 0;
  Eigen::Matrix<double, 1, 1> friction_coef;

  double friction_angle = this->state_variable("phi");

  if (std::isnan(friction_angle))
    friction_angle = std::atan(discontinuity_friction_coef);
  friction_coef_scalar = std::tan(friction_angle);
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    volume_node = nodes_[i]->volume(mpm::ParticlePhase::Solid);

    // Unit 1x1 Eigen matrix to be used with scalar quantities
    friction_coef(0, 0) =
        friction_coef_scalar * volume_ * shapefn_[i] / volume_node;

    nodes_[i]->update_discontinuity_property(true, "friction_coef",
                                             friction_coef, 0, 1);
  }
}

// Compute strain rate of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::ParticleXMPM<3>::compute_strain_rate(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();
  const double tolerance = 1.E-16;
  Eigen::Vector3d vel;
  vel.setZero();

  // Compute corresponding nodal velocity
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {

    // branch
    double nodal_mass = nodes_[i]->mass(phase);
    auto nodal_mass_enrich = nodes_[i]->mass_enrich();

    auto nodal_momentum = nodes_[i]->momentum(phase);
    auto nodal_momentum_enrich = nodes_[i]->momentum_enrich();

    auto discontinuity_id = nodes_[i]->discontinuity_id();
    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::single_enriched) {

      nodal_mass +=
          nodal_mass_enrich[0] * sgn(levelset_phi_[discontinuity_id[0]]);
      nodal_momentum.col(0) += nodal_momentum_enrich.col(0) *
                               sgn(levelset_phi_[discontinuity_id[0]]);

    } else if (nodes_[i]->enrich_type() ==
               mpm::NodeEnrichType::double_enriched) {
      nodal_mass +=
          nodal_mass_enrich[0] * sgn(levelset_phi_[discontinuity_id[0]]) +
          nodal_mass_enrich[1] * sgn(levelset_phi_[discontinuity_id[1]]) +
          nodal_mass_enrich[2] * sgn(levelset_phi_[discontinuity_id[0]]) *
              sgn(levelset_phi_[discontinuity_id[1]]);

      nodal_momentum.col(0) += nodal_momentum_enrich.col(0) *
                                   sgn(levelset_phi_[discontinuity_id[0]]) +
                               nodal_momentum_enrich.col(1) *
                                   sgn(levelset_phi_[discontinuity_id[1]]) +
                               nodal_momentum_enrich.col(2) *
                                   sgn(levelset_phi_[discontinuity_id[0]]) *
                                   sgn(levelset_phi_[discontinuity_id[1]]);
    }
    if (nodal_mass < tolerance) continue;

    vel = nodal_momentum / nodal_mass;

    strain_rate[0] += dn_dx(i, 0) * vel[0];
    strain_rate[1] += dn_dx(i, 1) * vel[1];
    strain_rate[2] += dn_dx(i, 2) * vel[2];
    strain_rate[3] += dn_dx(i, 1) * vel[0] + dn_dx(i, 0) * vel[1];
    strain_rate[4] += dn_dx(i, 2) * vel[1] + dn_dx(i, 1) * vel[2];
    strain_rate[5] += dn_dx(i, 2) * vel[0] + dn_dx(i, 0) * vel[2];
  }

  for (unsigned i = 0; i < strain_rate.size(); ++i)
    if (std::fabs(strain_rate[i]) < 1.E-15) strain_rate[i] = 0.;

  // return strain_rate;
  return strain_rate;
}

//! Map body force
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::map_body_force(
    const VectorDim& pgravity) noexcept {
  // Compute nodal body forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_external_force(true, mpm::ParticlePhase::Solid,
                                     (pgravity * mass_ * shapefn_(i)));

    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::regular) continue;
    auto discontinuity_id = nodes_[i]->discontinuity_id();
    Eigen::Matrix<double, Tdim, 3> external_force;
    external_force.setZero();
    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::single_enriched) {
      external_force.col(0) = sgn(levelset_phi_[discontinuity_id[0]]) *
                              pgravity * mass_ * shapefn_(i);
    } else if (nodes_[i]->enrich_type() ==
               mpm::NodeEnrichType::double_enriched) {
      external_force.col(0) = sgn(levelset_phi_[discontinuity_id[0]]) *
                              pgravity * mass_ * shapefn_(i);
      external_force.col(1) = sgn(levelset_phi_[discontinuity_id[1]]) *
                              pgravity * mass_ * shapefn_(i);
      external_force.col(2) = sgn(levelset_phi_[discontinuity_id[0]]) *
                              sgn(levelset_phi_[discontinuity_id[1]]) *
                              pgravity * mass_ * shapefn_(i);
    }
    nodes_[i]->update_external_force_enrich(external_force);
  }
}

//! Map traction force
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::map_traction_force() noexcept {
  if (this->set_traction_) {
    // Map particle traction forces to nodes
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      nodes_[i]->update_external_force(true, mpm::ParticlePhase::Solid,
                                       (shapefn_[i] * traction_));

      if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::regular) continue;
      auto discontinuity_id = nodes_[i]->discontinuity_id();
      Eigen::Matrix<double, Tdim, 3> external_force;
      external_force.setZero();
      if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::single_enriched) {
        external_force.col(0) =
            sgn(levelset_phi_[discontinuity_id[0]]) * traction_ * shapefn_(i);
      } else if (nodes_[i]->enrich_type() ==
                 mpm::NodeEnrichType::double_enriched) {
        external_force.col(0) =
            sgn(levelset_phi_[discontinuity_id[0]]) * traction_ * shapefn_(i);
        external_force.col(1) =
            sgn(levelset_phi_[discontinuity_id[1]]) * traction_ * shapefn_(i);
        external_force.col(2) = sgn(levelset_phi_[discontinuity_id[0]]) *
                                sgn(levelset_phi_[discontinuity_id[1]]) *
                                traction_ * shapefn_(i);
      }
      nodes_[i]->update_external_force_enrich(external_force);
    }
  }
}

//! Map internal force
template <>
inline void mpm::ParticleXMPM<1>::map_internal_force() noexcept {
  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 1, 1> force;
    force[0] = -1. * dn_dx_(i, 0) * volume_ * stress_[0];

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Solid, force);

    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::regular) continue;
    auto discontinuity_id = nodes_[i]->discontinuity_id();
    Eigen::Matrix<double, 1, 3> internal_force;
    internal_force.setZero();
    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::single_enriched) {
      internal_force.col(0) = sgn(levelset_phi_[discontinuity_id[0]]) * force;
    } else if (nodes_[i]->enrich_type() ==
               mpm::NodeEnrichType::double_enriched) {
      internal_force.col(0) = sgn(levelset_phi_[discontinuity_id[0]]) * force;
      internal_force.col(1) = sgn(levelset_phi_[discontinuity_id[1]]) * force;
      internal_force.col(2) = sgn(levelset_phi_[discontinuity_id[0]]) *
                              sgn(levelset_phi_[discontinuity_id[1]]) * force;
    }
    nodes_[i]->update_internal_force_enrich(internal_force);
  }
}

//! Map internal force
template <>
inline void mpm::ParticleXMPM<2>::map_internal_force() noexcept {
  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 2, 1> force;
    force[0] = dn_dx_(i, 0) * stress_[0] + dn_dx_(i, 1) * stress_[3];
    force[1] = dn_dx_(i, 1) * stress_[1] + dn_dx_(i, 0) * stress_[3];

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Solid, force);

    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::regular) continue;
    auto discontinuity_id = nodes_[i]->discontinuity_id();
    Eigen::Matrix<double, 2, 3> internal_force;
    internal_force.setZero();
    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::single_enriched) {
      internal_force.col(0) = sgn(levelset_phi_[discontinuity_id[0]]) * force;
    } else if (nodes_[i]->enrich_type() ==
               mpm::NodeEnrichType::double_enriched) {
      internal_force.col(0) = sgn(levelset_phi_[discontinuity_id[0]]) * force;
      internal_force.col(1) = sgn(levelset_phi_[discontinuity_id[1]]) * force;
      internal_force.col(2) = sgn(levelset_phi_[discontinuity_id[0]]) *
                              sgn(levelset_phi_[discontinuity_id[1]]) * force;
    }
    nodes_[i]->update_internal_force_enrich(internal_force);
  }
}

//! Map internal force
template <>
inline void mpm::ParticleXMPM<3>::map_internal_force() noexcept {
  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 3, 1> force;
    force[0] = dn_dx_(i, 0) * stress_[0] + dn_dx_(i, 1) * stress_[3] +
               dn_dx_(i, 2) * stress_[5];

    force[1] = dn_dx_(i, 1) * stress_[1] + dn_dx_(i, 0) * stress_[3] +
               dn_dx_(i, 2) * stress_[4];

    force[2] = dn_dx_(i, 2) * stress_[2] + dn_dx_(i, 1) * stress_[4] +
               dn_dx_(i, 0) * stress_[5];

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Solid, force);

    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::regular) continue;
    auto discontinuity_id = nodes_[i]->discontinuity_id();
    Eigen::Matrix<double, 3, 3> internal_force;
    internal_force.setZero();
    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::single_enriched) {
      internal_force.col(0) = sgn(levelset_phi_[discontinuity_id[0]]) * force;
    } else if (nodes_[i]->enrich_type() ==
               mpm::NodeEnrichType::double_enriched) {
      internal_force.col(0) = sgn(levelset_phi_[discontinuity_id[0]]) * force;
      internal_force.col(1) = sgn(levelset_phi_[discontinuity_id[1]]) * force;
      internal_force.col(2) = sgn(levelset_phi_[discontinuity_id[0]]) *
                              sgn(levelset_phi_[discontinuity_id[1]]) * force;
    }

    nodes_[i]->update_internal_force_enrich(internal_force);
  }
}

// Compute updated position of the particle
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::compute_updated_position(
    double dt, bool velocity_update) noexcept {

  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);
  // Get interpolated nodal velocity
  Eigen::Matrix<double, Tdim, 1> nodal_velocity =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  const double tolerance = 1.E-16;
  unsigned int phase = mpm::ParticlePhase::Solid;
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // branch
    double nodal_mass = nodes_[i]->mass(phase);

    auto nodal_momentum = nodes_[i]->momentum(phase);

    // if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::regular) continue;

    auto nodal_mass_enrich = nodes_[i]->mass_enrich();

    auto nodal_momentum_enrich = nodes_[i]->momentum_enrich();

    auto discontinuity_id = nodes_[i]->discontinuity_id();

    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::single_enriched) {

      nodal_mass +=
          nodal_mass_enrich[0] * sgn(levelset_phi_[discontinuity_id[0]]);
      nodal_momentum.col(0) += nodal_momentum_enrich.col(0) *
                               sgn(levelset_phi_[discontinuity_id[0]]);

    } else if (nodes_[i]->enrich_type() ==
               mpm::NodeEnrichType::double_enriched) {
      nodal_mass +=
          nodal_mass_enrich[0] * sgn(levelset_phi_[discontinuity_id[0]]) +
          nodal_mass_enrich[1] * sgn(levelset_phi_[discontinuity_id[1]]) +
          nodal_mass_enrich[2] * sgn(levelset_phi_[discontinuity_id[0]]) *
              sgn(levelset_phi_[discontinuity_id[1]]);

      nodal_momentum.col(0) += nodal_momentum_enrich.col(0) *
                                   sgn(levelset_phi_[discontinuity_id[0]]) +
                               nodal_momentum_enrich.col(1) *
                                   sgn(levelset_phi_[discontinuity_id[1]]) +
                               nodal_momentum_enrich.col(2) *
                                   sgn(levelset_phi_[discontinuity_id[0]]) *
                                   sgn(levelset_phi_[discontinuity_id[1]]);
    }
    if (nodal_mass < tolerance) continue;

    nodal_velocity += shapefn_[i] * nodal_momentum / nodal_mass;
  }
  Eigen::Matrix<double, Tdim, 1> nodal_velocity_enrich =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  // Acceleration update
  if (!velocity_update) {
    // to do
    double shapefn_enrich = 0;
    // Get interpolated nodal acceleration
    Eigen::Matrix<double, Tdim, 1> nodal_acceleration =
        Eigen::Matrix<double, Tdim, 1>::Zero();
    for (unsigned i = 0; i < nodes_.size(); ++i) {

      if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::regular) {
        double nodal_mass = nodes_[i]->mass(phase);
        if (nodal_mass < tolerance) continue;

        auto force =
            nodes_[i]->internal_force(phase) + nodes_[i]->external_force(phase);

        nodal_acceleration += shapefn_[i] * force / nodal_mass;

      } else if (nodes_[i]->enrich_type() ==
                 mpm::NodeEnrichType::single_enriched) {

        auto discontinuity_id = nodes_[i]->discontinuity_id();

        double nodal_mass = nodes_[i]->mass(phase);
        auto nodal_momentum = nodes_[i]->momentum(phase);
        auto nodal_mass_enrich = nodes_[i]->mass_enrich();
        auto nodal_momentum_enrich = nodes_[i]->momentum_enrich();

        nodal_mass +=
            nodal_mass_enrich[0] * sgn(levelset_phi_[discontinuity_id[0]]);

        if (nodal_mass < tolerance) continue;
        nodal_momentum.col(0) += nodal_momentum_enrich.col(0) *
                                 sgn(levelset_phi_[discontinuity_id[0]]);

        nodal_velocity_enrich += shapefn_[i] * nodal_momentum / nodal_mass;
        shapefn_enrich += shapefn_[i];
      } else if (nodes_[i]->enrich_type() ==
                 mpm::NodeEnrichType::double_enriched) {

        double nodal_mass = nodes_[i]->mass(phase);
        auto nodal_momentum = nodes_[i]->momentum(phase);
        auto nodal_mass_enrich = nodes_[i]->mass_enrich();
        auto nodal_momentum_enrich = nodes_[i]->momentum_enrich();

        auto discontinuity_id = nodes_[i]->discontinuity_id();
        nodal_mass +=
            nodal_mass_enrich[0] * sgn(levelset_phi_[discontinuity_id[0]]) +
            nodal_mass_enrich[1] * sgn(levelset_phi_[discontinuity_id[1]]) +
            nodal_mass_enrich[2] * sgn(levelset_phi_[discontinuity_id[0]]) *
                sgn(levelset_phi_[discontinuity_id[1]]);

        if (nodal_mass < tolerance) continue;

        nodal_momentum.col(0) += nodal_momentum_enrich.col(0) *
                                     sgn(levelset_phi_[discontinuity_id[0]]) +
                                 nodal_momentum_enrich.col(1) *
                                     sgn(levelset_phi_[discontinuity_id[1]]) +
                                 nodal_momentum_enrich.col(2) *
                                     sgn(levelset_phi_[discontinuity_id[0]]) *
                                     sgn(levelset_phi_[discontinuity_id[1]]);
        nodal_velocity_enrich += shapefn_[i] * nodal_momentum / nodal_mass;
        shapefn_enrich += shapefn_[i];
      }
    }

    // Update particle velocity from interpolated nodal acceleration
    this->velocity_ = nodal_velocity_enrich +
                      (1 - shapefn_enrich) * this->velocity_ +
                      nodal_acceleration * dt;
  }
  // Update particle velocity using interpolated nodal velocity
  else
    this->velocity_ = nodal_velocity;

  // New position  current position + velocity * dt
  this->coordinates_ += nodal_velocity * dt;
  // Update displacement (displacement is initialized from zero)
  this->displacement_ += nodal_velocity * dt;
}

//! Map levelset from nodes to particles
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::map_levelset_to_particle(unsigned dis_id) {
  double levelset_phi = 0;
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    levelset_phi += shapefn_[i] * nodes_[i]->levelset_phi(dis_id);
  }
  levelset_phi_[dis_id] = levelset_phi;
}

//! compute the mininum eigenvalue of the acoustic tensor
//! Map levelset from nodes to particles
template <unsigned Tdim>
bool mpm::ParticleXMPM<Tdim>::minimum_acoustic_tensor(VectorDim& normal_cell,
                                                      bool initiation,
                                                      unsigned dis_id) {

  minimum_acoustic_eigenvalue_ = std::numeric_limits<double>::max();
  // compute the acoustic tensor
  bool yield_status = true;
  Eigen::Matrix<double, 6, 6> dp =
      (this->material())
          ->compute_elasto_plastic_tensor(
              stress_, dstrain_, this,
              &state_variables_[mpm::ParticlePhase::Solid], false);

  if (!yield_status) return false;

  Eigen::Matrix<int, 3, 3> index;
  index << 0, 3, 5, 3, 1, 4, 5, 4, 2;
  Eigen::Matrix<double, 3, 1> normal_cellcenter;
  normal_cellcenter.setZero();
  // compute the initial direction
  if (!initiation) {

    for (int i = 0; i < nodes_.size(); i++) {
      double levelset = nodes_[i]->levelset_phi(dis_id);
      for (int j = 0; j < 3; j++)
        normal_cellcenter[j] += dn_dx_centroid_(i, j) * levelset;
    }
    normal_cellcenter = cell_->normal_discontinuity(dis_id);
  }

  if (initiation) {
    //! compute the gradient of displacement dot direction
    compute_initiation_normal(normal_cellcenter);
    initiation = false;
  }

  normal_cellcenter.normalize();
  // Iteration
  Eigen::Matrix<double, 3, 1> nk = normal_cellcenter;
  Eigen::Matrix<double, 3, 1> nk1;

  double uk = 0;
  double uk1;

  int itr_max = 1000;
  double itr_tol = 1e-9;
  double itr_error = 1;
  double eigenvalue_j[3]{0};

  Eigen::Matrix<double, 3, 3> A;
  Eigen::Matrix<double, 3, 3> J;
  for (int itr = 0; itr < itr_max; ++itr) {
    if (itr_error < itr_tol) break;

    A.setZero();
    J.setZero();
    for (int m = 0; m < 3; m++)
      for (int n = 0; n < 3; n++) {

        for (int r = 0; r < 3; r++)
          for (int s = 0; s < 3; s++)
            A(m, n) += nk(r) * nk(s) * dp(index(m, r), index(n, s));
      }
    double det_A = A.determinant();
    Eigen::Matrix<double, 3, 3> inv_A = A.inverse();

    for (int m = 0; m < 3; m++)
      for (int n = 0; n < 3; n++) {

        for (int r = 0; r < 3; r++)
          for (int s = 0; s < 3; s++)
            J(m, n) += inv_A(r, s) * (dp(index(m, s), index(r, n)) +
                                      dp(index(n, s), index(r, m)));
      }
    J = J * 0.5 * det_A;

    Eigen::EigenSolver<Eigen::Matrix3d> eigen_J(J);
    auto eigenvalues = eigen_J.pseudoEigenvalueMatrix();
    auto eigenvectors = eigen_J.pseudoEigenvectors();

    double project = -1e6;
    for (int i = 0; i < 3; ++i) {
      Eigen::Matrix<double, 3, 1> eigen = eigenvectors.col(i);
      eigenvalue_j[i] = eigenvalues(i, i);

      if (eigen.dot(nk) < 0) eigen = -eigen;
      if (std::abs(eigen.dot(nk)) < project) continue;

      project = std::abs(eigen.dot(nk));
      nk1 = eigen;
      uk1 = eigenvalues(i, i);
    }

    itr_error = (nk1 - nk).norm() + std::abs(uk1 / uk - 1);
    console_->info("\n itr:{},{}\nJ eigen:{},{},{}", itr, itr_error,
                   eigenvalues(0, 0), eigenvalues(1, 1), eigenvalues(2, 2));
    nk = nk1;
    uk = uk1;
    if (itr >= 999) {
      console_->error(
          "wrong iteration of acoustic tensor, {},{},{},{},{},{},{},{},{},{}",
          itr, itr_error, nk[0], nk[1], nk[2], nk1[0], nk1[1], nk1[2], uk1, uk);
      if (itr_error < 1e-6) continue;
      nk = normal_cellcenter;
    }
  }
  if (normal_cellcenter.dot(nk) < 0) nk = -nk;
  normal_cell = nk;
  // to do
  // A.setZero();
  // for (int m = 0; m < 3; m++)
  // for (int n = 0; n < 3; n++) {
  //     for (int r = 0; r < 3; r++)
  //     for (int s = 0; s < 3; s++)
  //         A(m, n) +=
  //             nk(r) * nk(s) * dp(index(m, r), index(n, s));
  // }
  // double det_A = A.determinant();
  // Eigen::EigenSolver<Eigen::Matrix3d> eigen_A(A);
  //   auto eigenvalues = eigen_A.pseudoEigenvalueMatrix();
  //   auto eigenvectors = eigen_A.pseudoEigenvectors();

  // console_->info("\n A eigen:{},{},{},{}\n",
  // det_A,eigenvalues(0,0),eigenvalues(1,1),eigenvalues(2,2));

  //   console_->info("\n normal:{},{},{},\nA eigen:{},{},{},{}\n", nk[0],
  //   nk[1],
  //                  nk[2], det_A, eigenvalues(0, 0), eigenvalues(1, 1),
  //                  eigenvalues(2, 2));
  return true;
}

template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::compute_initiation_normal(
    VectorDim& normal_initiation) {

  double dtheta = 1;
  const double PI = M_PI / 180;
  VectorDim normal_m;
  VectorDim normal_m1;
  VectorDim normal_m2;
  VectorDim dudx_n;
  VectorDim normal;
  double a1, a2;

  Eigen::Matrix<double, 3, 3> dp_n;
  Eigen::Matrix<double, 3, 3> de_n;

  double det_de_n;
  double det_dp_n;

  double max_dudxmn = 0;

  Eigen::Matrix<double, 6, 6> dp =
      (this->material())
          ->compute_elasto_plastic_tensor(
              stress_, dstrain_, this,
              &state_variables_[mpm::ParticlePhase::Solid], false);

  Eigen::Matrix<double, 6, 6> de =
      (this->material())
          ->compute_elastic_tensor(
              &state_variables_[mpm::ParticlePhase::Solid]);
  // clang-format off
  Eigen::Matrix<int, 3, 3> index;
  index << 0, 3, 5,
           3, 1, 4,
           5, 4, 2;

  for (int i = 0; i < std::floor(360 / dtheta); i++) {
    for (int j = 0; j < 180; j++) {
      double theta = i * dtheta * PI;
      double phi = j * dtheta * PI;
      normal << std::cos(phi) * std::cos(theta),
          std::cos(phi) * std::sin(theta), std::sin(phi);

      dp_n.setZero();
      de_n.setZero();
      for (int m = 0; m < 3; m++)
        for (int n = 0; n < 3; n++) {

          for (int r = 0; r < 3; r++)
            for (int s = 0; s < 3; s++)
              dp_n(m, n) +=
                  normal(r) * normal(s) * dp(index(m, r), index(n, s));
        }

      det_dp_n = dp_n.determinant();
      for (int m = 0; m < 3; m++)
        for (int n = 0; n < 3; n++) {

          for (int r = 0; r < 3; r++)
            for (int s = 0; s < 3; s++)
              de_n(m, n) +=
                  normal(r) * normal(s) * de(index(m, r), index(n, s));
        }

      det_de_n = de_n.determinant();
      double ratio = det_dp_n / det_de_n;
      if(ratio > 0.01)
        continue;

     dudx_n.setZero();
      for(unsigned m = 0; m < 3; m++)
        for(unsigned n = 0; n < 3; n++)
            dudx_n[m] += du_dx_(m,n)*normal[n];

      normal_m1 << std::sin(theta),-std::cos(theta),0;
      normal_m2 = normal.cross(normal_m1);
     
     a1 = dudx_n.dot(normal_m1);
     a2 = dudx_n.dot(normal_m2);
    
    double angle = 0;

    angle = std::atan(a2/a1);
    if(a1 == 0)
        angle = M_PI_2;

    double dudx_mn = std::abs(a1*std::cos(angle) + a2*std::sin(angle));
    if(dudx_mn > max_dudxmn)
    {
        max_dudxmn = dudx_mn;
        normal_initiation = normal;
    }
    }
  }
}

// Compute du_dx_ of the particle
template <>
void inline mpm::ParticleXMPM<3>::compute_displacement_gradient(double dt) {
  // Define strain rate
  Eigen::Matrix<double, 3, 3> dudx_rate = Eigen::Matrix<double, 3, 3>::Zero();
  const double tolerance = 1.E-16;
  unsigned phase = mpm::ParticlePhase::Solid;
  Eigen::Vector3d vel;
  vel.setZero();
  // Compute corresponding nodal velocity
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
// branch
    double nodal_mass = nodes_[i]->mass(phase);

    auto nodal_momentum = nodes_[i]->momentum(phase);

    // if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::regular) continue;

    auto nodal_mass_enrich = nodes_[i]->mass_enrich();

    auto nodal_momentum_enrich = nodes_[i]->momentum_enrich();

    auto discontinuity_id = nodes_[i]->discontinuity_id();

    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::single_enriched) {

      nodal_mass +=
          nodal_mass_enrich[0] * sgn(levelset_phi_[discontinuity_id[0]]);
      nodal_momentum.col(0) += nodal_momentum_enrich.col(0) *
                               sgn(levelset_phi_[discontinuity_id[0]]);

    } else if (nodes_[i]->enrich_type() ==
               mpm::NodeEnrichType::double_enriched) {
      nodal_mass +=
          nodal_mass_enrich[0] * sgn(levelset_phi_[discontinuity_id[0]]) +
          nodal_mass_enrich[1] * sgn(levelset_phi_[discontinuity_id[1]]) +
          nodal_mass_enrich[2] * sgn(levelset_phi_[discontinuity_id[0]]) *
              sgn(levelset_phi_[discontinuity_id[1]]);

      nodal_momentum.col(0) += nodal_momentum_enrich.col(0) *
                                   sgn(levelset_phi_[discontinuity_id[0]]) +
                               nodal_momentum_enrich.col(1) *
                                   sgn(levelset_phi_[discontinuity_id[1]]) +
                               nodal_momentum_enrich.col(2) *
                                   sgn(levelset_phi_[discontinuity_id[0]]) *
                                   sgn(levelset_phi_[discontinuity_id[1]]);
    }
    if (nodal_mass < tolerance) continue;

    vel += shapefn_[i] * nodal_momentum / nodal_mass;
    for(unsigned j = 0; j < 3; j++) 
        for(unsigned k = 0; k < 3; k++) 
            dudx_rate(j,k) +=  vel[j] * dn_dx_(i, k) ;
  }

    for(unsigned j = 0; j < 3; j++) 
        for(unsigned k = 0; k < 3; k++) 
            du_dx_(j,k) += dudx_rate(j,k)*dt;
}

//to do
//! Map particle friction_coef_to_nodes
template <unsigned Tdim>
void mpm::ParticleXMPM<Tdim>::check_levelset(unsigned dis_id) noexcept {
    if(levelset_phi_[dis_id] != 0)
        return;
    assert(cell_ != nullptr);
   
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    if (!nodes_[i]->discontinuity_enrich(dis_id)) 
        continue;
      
    auto normal = cell_->normal_discontinuity(dis_id);
    double d = cell_->d_discontinuity(dis_id);
    levelset_phi_[dis_id] = coordinates_.dot(normal) + d;
    return;
    }
}
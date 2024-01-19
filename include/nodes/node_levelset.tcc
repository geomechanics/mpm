// Assign levelset values to the nodes
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
bool mpm::NodeLevelset<Tdim>::assign_levelset(double levelset,
                                              double levelset_mu,
                                              double barrier_stiffness,
                                              double slip_threshold,
                                              double levelset_mp_radius) {
  bool status = true;
  try {
    if ((levelset_mu <= 0.) || (barrier_stiffness < +0.) ||
        (slip_threshold <= 0.) || (levelset_mp_radius <= 0.))
      throw std::runtime_error(
          "Levelset variables (apart from levelset value itself) cannot be "
          "negative");

    // Set variables
    this->levelset_ = levelset;
    this->levelset_mu_ = levelset_mu;
    this->barrier_stiffness_ = barrier_stiffness;
    this->slip_threshold_ = slip_threshold;
    this->levelset_mp_radius_ = levelset_mp_radius;

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}
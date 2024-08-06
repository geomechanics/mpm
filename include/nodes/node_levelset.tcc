// Assign levelset values to the nodes
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
bool mpm::NodeLevelset<Tdim, Tdof, Tnphases>::assign_levelset(
    double levelset, double levelset_mu, double levelset_alpha,
    double barrier_stiffness, double slip_threshold) {
  bool status = true;
  try {
    if ((levelset_mu < 0.) || (levelset_alpha < 0.) || (slip_threshold < 0.))
      throw std::runtime_error(
          "Levelset mu, alpha, and slip threshold cannot be negative");
    if ((barrier_stiffness <= 0.))
      throw std::runtime_error("Barrier stiffness must be greater than zero");

    // Set variables
    this->levelset_ = levelset;
    this->levelset_mu_ = levelset_mu;
    this->levelset_alpha_ = levelset_alpha;
    this->barrier_stiffness_ = barrier_stiffness;
    this->slip_threshold_ = slip_threshold;

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}
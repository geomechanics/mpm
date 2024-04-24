// Assign levelset values to the nodes
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
bool mpm::NodeLevelset<Tdim, Tdof, Tnphases>::assign_levelset(
    double levelset, double levelset_mu, double levelset_alpha,
    double barrier_stiffness, double slip_threshold) {
  bool status = true;
  try {
    if ((levelset_mu < 0.) || (levelset_alpha < 0.))
      throw std::runtime_error("Levelset mu and alpha cannot be negative");
    if (slip_threshold == 0.)
      slip_threshold = std::numeric_limits<double>::epsilon();  // LEDT check
    if ((barrier_stiffness <= 0.) || (slip_threshold <= 0.))
      throw std::runtime_error(
          "Barrier stiffness and slip threshold must be greater than zero");

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
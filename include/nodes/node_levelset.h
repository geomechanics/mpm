#ifndef MPM_NODE_LEVELSET_H_
#define MPM_NODE_LEVELSET_H_

#include "node.h"
#include "node_base.h"

namespace mpm {

//! Levelset subclass
//! \brief subclass that stores the information about levelset node
//! \tparam Tdim Dimension
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
class NodeLevelset : public Node<Tdim, Tdof, Tnphases> {

 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Constructor with id, coordinates and dof
  //! \param[in] id Node id
  //! \param[in] coord coordinates of the node
  NodeLevelset(Index id, const VectorDim& coord)
      : mpm::Node<Tdim, Tdof, Tnphases>(id, coord) {
    console_ =
        std::make_unique<spdlog::logger>("NodeLevelset", mpm::stdout_sink);
  };

  //! Virtual destructor
  ~NodeLevelset() override{};

  //! Delete copy constructor
  NodeLevelset(const NodeLevelset<Tdim, Tdof, Tnphases>&) = delete;

  //! Delete assignement operator
  NodeLevelset& operator=(const NodeLevelset<Tdim, Tdof, Tnphases>&) = delete;

  // Assign levelset values to nodes
  //! \param[in] levelset Levelset value at the particle
  //! \param[in] levelset_mu Levelset friction
  //! \param[in] barrier_stiffness Barrier stiffness
  //! \param[in] slip_threshold Slip threshold
  bool assign_levelset(double levelset, double levelset_mu,
                       double barrier_stiffness,
                       double slip_threshold) override;

  //! Return levelset value
  inline double levelset() const override { return levelset_; }

  //! Return levelset friction
  inline double levelset_mu() const override { return levelset_mu_; }

  //! Return barrier stiffness
  inline double barrier_stiffness() const override {
    return barrier_stiffness_;
  }

  //! Return slip threshold
  inline double slip_threshold() const override { return slip_threshold_; }

 private:
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Levelset value
  double levelset_;
  //! Levelset friction
  double levelset_mu_;
  //! Barrier stiffness
  double barrier_stiffness_;
  //! Slip threshold
  double slip_threshold_;

};  // NodeLevelset class

}  // namespace mpm

#include "node_levelset.tcc"

#endif  // MPM_NODE_LEVELSET_H_
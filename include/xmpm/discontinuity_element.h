#ifndef MPM_DISCONTINUITY_ELEMENT_H_
#define MPM_DISCONTINUITY_ELEMENT_H_

#include <algorithm>
#include <limits>
#include <vector>

#include <Eigen/Dense>

//! MPM namespace
namespace mpm {

// Element enrich type
enum EnrichType {
  Regular = 1,
  Crossed = 2,
  Tip = 3,
  NeighbourTip_1 = 4,
  NeighbourTip_2 = 5,
  NeighbourTip_3 = 6,
  PotentialTip = 7,
  NextTip = 8,
  NeighbourNextTip_1 = 9,
  NeighbourNextTip_2 = 10,
  NeighbourNextTip_3 = 11,
  InitialTip = 12
};

// Discontinuity element class
//! \brief Base class for  Discontinuity element
//! \tparam Tdim Dimension
template <unsigned Tdim>
class DiscontinuityElement {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;
  //! Default constructor
  DiscontinuityElement(mpm::EnrichType type) { enrich_type_ = type; };

  //! Destructor
  virtual ~DiscontinuityElement(){};

  void initialize() {
    enrich_type_ = mpm::EnrichType::Regular;
    normal_.setZero();
    d_ = 0;
    area_ = 0;
    cohesion_cor_ = VectorDim::Ones() * std::numeric_limits<double>::max();
  }

  //! Assign discontinuity element type
  void assign_element_type(mpm::EnrichType type) { enrich_type_ = type; }

  //! assign the normal direction of the discontinuity in the cell
  //! \param[in] the normal direction
  void assign_normal_discontinuity(VectorDim normal) { normal_ = normal; };

  //! assign constant value of the discontinuity plane equation
  //! \param[in] the constant value
  void assign_d(double d) { d_ = d; };

  //! assign the area of the discontinuity in the cell
  //! \param[in] the area
  void assign_area(double area) { area_ = area; };

  //! assign the center of the discontinuity in the cell
  //! \param[in] the area
  void assign_cohesion_cor(VectorDim cor) { cohesion_cor_ = cor; };

  //! return enriched type
  unsigned element_type() { return enrich_type_; }

  //! return normal_discontinuity
  VectorDim normal_discontinuity() { return normal_; }

  //! return d
  double d_discontinuity() { return d_; }

  //! return area
  double area() { return area_; }

  //! return the centroid
  VectorDim cohesion_cor() { return cohesion_cor_; }

 private:
  // default enrich type
  mpm::EnrichType enrich_type_;

  // normal direction
  VectorDim normal_;

  // constant value of the discontinuity plane equation a*x + b*y + c*z + d = 0
  double d_;

  // the section area of the crossed cell
  double area_{0};

  // the center of the discontinuity
  VectorDim cohesion_cor_ =
      VectorDim::Ones() * std::numeric_limits<double>::max();
};

}  // namespace mpm

#endif  // MPM_QUADRATURE_BASE_H_

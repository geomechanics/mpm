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
  InitialTip = 12,
  Irregular = 13
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

  void initialise() {
    enrich_type_ = mpm::EnrichType::Regular;
    normal_.setZero();
    d_ = 0;
    area_ = 0;
    cohesion_cor_ = VectorDim::Ones() * std::numeric_limits<double>::max();
  }

  //! Assign discontinuity element type
  //! \param[in] the element type
  void assign_element_type(mpm::EnrichType type) { enrich_type_ = type; }

  //! Assign the normal direction of the discontinuity in the cell
  //! \param[in] the normal direction
  void assign_normal_discontinuity(VectorDim normal) { normal_ = normal; };

  //! Assign constant value of the discontinuity plane equation
  //! \param[in] the constant value
  void assign_d(double d) { d_ = d; };

  //! Assign the area of the discontinuity in the cell
  //! \param[in] the area
  void assign_area(double area) { area_ = area; };

  //! Assign the center of the discontinuity in the cell
  //! \param[in] the center of the discontinuity
  void assign_cohesion_cor(VectorDim cor) { cohesion_cor_ = cor; };

  //! Return enriched type
  //! \retval the element type
  unsigned element_type() { return enrich_type_; }

  //! Return normal_discontinuity
  //! \retval the normal of the discontinuity at the cell center
  VectorDim normal_discontinuity() { return normal_; }

  //! Return d
  //! \retval the constant of the plane equation
  double d_discontinuity() { return d_; }

  //! Return area
  //! \retval the area of the discontinuity for the cohesion
  double area() { return area_; }

  //! Return the centroid
  //! \retval the center of the discontinuity
  VectorDim cohesion_cor() { return cohesion_cor_; }

 private:
  //! Default enrich type
  mpm::EnrichType enrich_type_;
  //! Normal direction
  VectorDim normal_;
  //! Constant value of the discontinuity plane equation a*x + b*y + c*z + d = 0
  double d_;
  //! The section area of the crossed cell
  double area_{0};
  // TODO: Check if it needs a default value
  //! The center of the discontinuity
  VectorDim cohesion_cor_ =
      VectorDim::Ones() * std::numeric_limits<double>::max();
};

}  // namespace mpm

#endif  // MPM_QUADRATURE_BASE_H_

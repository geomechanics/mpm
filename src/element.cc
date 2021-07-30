#include "element.h"
#include "factory.h"
#include "hexahedron_bspline_element.h"
#include "hexahedron_element.h"
#include "hexahedron_gimp_element.h"
#include "quadrilateral_bspline_element.h"
#include "quadrilateral_element.h"
#include "quadrilateral_gimp_element.h"
#include "triangle_element.h"

// Triangle 3-noded element
static Register<mpm::Element<2>, mpm::TriangleElement<2, 3>> tri3("ED2T3");

// Triangle 6-noded element
static Register<mpm::Element<2>, mpm::TriangleElement<2, 6>> tri6("ED2T6");

// Quadrilateral 4-noded element
static Register<mpm::Element<2>, mpm::QuadrilateralElement<2, 4>> quad4(
    "ED2Q4");

// Quadrilateral 8-noded element
static Register<mpm::Element<2>, mpm::QuadrilateralElement<2, 8>> quad8(
    "ED2Q8");

// Quadrilateral 9-noded element
static Register<mpm::Element<2>, mpm::QuadrilateralElement<2, 9>> quad9(
    "ED2Q9");

// Quadrilateral 4-node-base GIMP element
static Register<mpm::Element<2>, mpm::QuadrilateralGIMPElement<2, 16>>
    quad_gimp16("ED2Q16G");

// Quadrilateral BSpline element of second order polynomial
static Register<mpm::Element<2>, mpm::QuadrilateralBSplineElement<2, 2>>
    quad_bspline4_p2("ED2Q4P2B");

// Hexahedron 8-noded element
static Register<mpm::Element<3>, mpm::HexahedronElement<3, 8>> hex8("ED3H8");

// Hexahedron 20-noded element
static Register<mpm::Element<3>, mpm::HexahedronElement<3, 20>> hex20("ED3H20");

// Hexahedron 8-node-base GIMP element
static Register<mpm::Element<3>, mpm::HexahedronGIMPElement<3, 64>> hex_gimp64(
    "ED3H64G");

// Hexahedron BSpline element of second order polynomial
static Register<mpm::Element<3>, mpm::HexahedronBSplineElement<3, 2>>
    hex_bspline8_p2("ED3H8P2B");

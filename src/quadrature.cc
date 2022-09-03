#include "quadrature.h"
#include "factory.h"
#include "hexahedron_quadrature.h"
#include "quadrilateral_quadrature.h"
#include "tetrahedron_quadrature.h"
#include "triangle_quadrature.h"

// Triangle 1
static Register<mpm::Quadrature<2>, mpm::TriangleQuadrature<2, 1>> triangle1(
    "QTRI1");

// Triangle 3
static Register<mpm::Quadrature<2>, mpm::TriangleQuadrature<2, 3>> triangle3(
    "QTRI2");

// Quadrilateral 1
static Register<mpm::Quadrature<2>, mpm::QuadrilateralQuadrature<2, 1>>
    quadrilateral1("QQUAD1");

// Quadrilateral 4
static Register<mpm::Quadrature<2>, mpm::QuadrilateralQuadrature<2, 4>>
    quadrilateral4("QQUAD2");

// Quadrilateral 9
static Register<mpm::Quadrature<2>, mpm::QuadrilateralQuadrature<2, 9>>
    quadrilateral9("QQUAD3");

// Quadrilateral 16
static Register<mpm::Quadrature<2>, mpm::QuadrilateralQuadrature<2, 16>>
    quadrilateral16("QQUAD4");

// Tetrahedron 1
static Register<mpm::Quadrature<3>, mpm::TetrahedronQuadrature<3, 1>>
    tetrahedron1("QTET1");

// Tetrahedron 4
static Register<mpm::Quadrature<3>, mpm::TetrahedronQuadrature<3, 4>>
    tetrahedron4("QTET2");

// Hexahedron 1
static Register<mpm::Quadrature<3>, mpm::HexahedronQuadrature<3, 1>>
    hexahedron1("QHEX1");

// Hexahedron 8
static Register<mpm::Quadrature<3>, mpm::HexahedronQuadrature<3, 8>>
    hexahedron8("QHEX2");

// Hexahedron 27
static Register<mpm::Quadrature<3>, mpm::HexahedronQuadrature<3, 27>>
    hexahedron27("QHEX3");

// Hexahedron 64
static Register<mpm::Quadrature<3>, mpm::HexahedronQuadrature<3, 64>>
    hexahedron64("QHEX4");

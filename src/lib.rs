// Copyright 2020 Jesper de Jong
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::array;
use std::convert::TryFrom;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, Sub, SubAssign};

use num_traits::{Float, NumAssign};

/// Trait for types that have `min()` and `max()` methods.
pub trait MinMax {
    /// Compares and returns the minimum of two values.
    fn min(self, other: Self) -> Self;

    /// Compares and returns the maximum of two values.
    fn max(self, other: Self) -> Self;
}

/// Compares and returns the minimum of two values.
#[inline]
pub fn min<T: MinMax>(a: T, b: T) -> T {
    a.min(b)
}

/// Compares and returns the maximum of two values.
#[inline]
pub fn max<T: MinMax>(a: T, b: T) -> T {
    a.max(b)
}

/// Trait for types for which a distance between values can be computed.
pub trait Distance: RelativeDistance {
    /// The output type which expresses the distance between values.
    type Output;

    /// Computes and returns the distance between two values.
    fn distance(self, other: Self) -> Self::Output;
}

/// Computes and returns the distance between two values.
#[inline]
pub fn distance<T: Distance>(a: T, b: T) -> T::Output {
    a.distance(b)
}

/// Trait for types for which distances between values can be compared.
pub trait RelativeDistance {
    /// Checks which of the values `a` and `b` is closest to this value and returns the closest one.
    fn closest(self, a: Self, b: Self) -> Self;

    /// Checks which of the values `a` and `b` is farthest from this value and returns the farthest one.
    fn farthest(self, a: Self, b: Self) -> Self;
}

/// Trait for types for which a length can be computed.
pub trait Length: RelativeLength {
    /// The output type which expresses the length of a value.
    type Output;

    /// Computes and returns the length of a value.
    fn length(self) -> Self::Output;
}

/// Trait for types for which lengths can be compared.
pub trait RelativeLength {
    /// Returns the shortest of two values.
    fn shortest(self, other: Self) -> Self;

    /// Returns the longest of two values.
    fn longest(self, other: Self) -> Self;
}

/// Returns the shortest of two values.
#[inline]
pub fn shortest<T: RelativeLength>(a: T, b: T) -> T {
    a.shortest(b)
}

/// Returns the longest of two values.
#[inline]
pub fn longest<T: RelativeLength>(a: T, b: T) -> T {
    a.longest(b)
}

/// Trait for types for which a dot product between values can be computed.
pub trait DotProduct<U> {
    /// The output type which expresses the dot product between values.
    type Output;

    /// Computes and returns the dot product between two values.
    fn dot(self, other: U) -> Self::Output;
}

/// Computes and returns the dot product between two values.
#[inline]
pub fn dot<T: DotProduct<U>, U>(a: T, b: U) -> T::Output {
    a.dot(b)
}

/// Trait for types for which a cross product between values can be computed.
pub trait CrossProduct<U> {
    /// The output type which expresses the cross product between values.
    type Output;

    /// Computes and returns the cross product between two values.
    fn cross(self, other: U) -> Self::Output;
}

/// Computes and returns the cross product between two values.
#[inline]
pub fn cross<T: CrossProduct<U>, U>(a: T, b: U) -> T::Output {
    a.cross(b)
}

/// Trait for types for which a union with a value can be computed.
pub trait Union<U> {
    /// The output type which represents the union between values.
    type Output;

    /// Computes and returns the union between two values.
    fn union(self, other: U) -> Self::Output;
}

/// Computes and returns the union between two values.
#[inline]
pub fn union<T: Union<U>, U>(a: T, b: U) -> T::Output {
    a.union(b)
}

/// Trait for types for which an intersection with a value can be computed.
pub trait Intersection<U> {
    /// The output type which represents the intersection between values.
    type Output;

    /// Computes and returns the intersection between two values.
    ///
    /// Returns `Some` when the intersection between the values is not empty; `None` if the intersection is empty.
    fn intersection(self, other: U) -> Option<Self::Output>;
}

/// Computes and returns the intersection between two values.
///
/// Returns `Some` when the intersection between the values is not empty; `None` if the intersection is empty.
#[inline]
pub fn intersection<T: Intersection<U>, U>(a: T, b: U) -> Option<T::Output> {
    a.intersection(b)
}

/// Trait to be implemented for `Transform2` or `Transform3` for types which can be transformed.
pub trait Transform<T> {
    /// The output type that results from transforming a value of type `T`.
    type Output;

    /// Transforms a value of type `T`.
    fn transform(&self, value: T) -> Self::Output;
}

/// Error returned when computing the inverse of a singular matrix is attempted.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct NonInvertibleMatrixError;

/// Extra constants that are not defined in `num_traits::float::FloatConst`.
pub trait ScalarConst {
    /// Return `0.5`.
    fn half() -> Self;

    /// Return `2.0`.
    fn two() -> Self;
}

/// Scalar value.
pub trait Scalar : Float + NumAssign + ScalarConst {
}

/// Dimension in 2D space.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Dimension2 { X, Y }

/// Point in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point2<S: Scalar> {
    pub x: S,
    pub y: S,
}

/// Vector in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector2<S: Scalar> {
    pub x: S,
    pub y: S,
}

/// Ray in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Ray2<S: Scalar> {
    pub origin: Point2<S>,
    pub direction: Vector2<S>,
}

/// Axis-aligned bounding box in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct BoundingBox2<S: Scalar> {
    pub min: Point2<S>,
    pub max: Point2<S>,
}

/// Matrix with 3 rows and 3 columns for transformations in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix3x3<S: Scalar> {
    m: [S; 9]
}

/// Transform for transformations in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Transform2<S: Scalar> {
    pub forward: Matrix3x3<S>,
    pub inverse: Matrix3x3<S>,
}

/// Dimension in 3D space.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Dimension3 { X, Y, Z }

/// Point in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point3<S: Scalar> {
    pub x: S,
    pub y: S,
    pub z: S,
}

/// Vector in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector3<S: Scalar> {
    pub x: S,
    pub y: S,
    pub z: S,
}

/// Surface normal in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Normal3<S: Scalar> {
    pub x: S,
    pub y: S,
    pub z: S,
}

/// Ray in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Ray3<S: Scalar> {
    pub origin: Point3<S>,
    pub direction: Vector3<S>,
}

/// Axis-aligned bounding box in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct BoundingBox3<S: Scalar> {
    pub min: Point3<S>,
    pub max: Point3<S>,
}

/// Matrix with 4 rows and 4 columns for transformations in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix4x4<S: Scalar> {
    m: [S; 16]
}

/// Transform for transformations in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Transform3<S: Scalar> {
    pub forward: Matrix4x4<S>,
    pub inverse: Matrix4x4<S>,
}

pub type Point2f = Point2<f32>;
pub type Vector2f = Vector2<f32>;
pub type Ray2f = Ray2<f32>;
pub type BoundingBox2f = BoundingBox2<f32>;
pub type Matrix3x3f = Matrix3x3<f32>;
pub type Transform2f = Transform2<f32>;
pub type Point3f = Point3<f32>;
pub type Vector3f = Vector3<f32>;
pub type Normal3f = Normal3<f32>;
pub type Ray3f = Ray3<f32>;
pub type BoundingBox3f = BoundingBox3<f32>;
pub type Matrix4x4f = Matrix4x4<f32>;
pub type Transform3f = Transform3<f32>;

pub type Point2d = Point2<f64>;
pub type Vector2d = Vector2<f64>;
pub type Ray2d = Ray2<f64>;
pub type BoundingBox2d = BoundingBox2<f64>;
pub type Matrix3x3d = Matrix3x3<f64>;
pub type Transform2d = Transform2<f64>;
pub type Point3d = Point3<f64>;
pub type Vector3d = Vector3<f64>;
pub type Normal3d = Normal3<f64>;
pub type Ray3d = Ray3<f64>;
pub type BoundingBox3d = BoundingBox3<f64>;
pub type Matrix4x4d = Matrix4x4<f64>;
pub type Transform3d = Transform3<f64>;

// ===== Scalar ================================================================================================================================================

impl<S: Scalar> MinMax for S {
    /// Compares and returns the minimum of two scalars.
    #[inline]
    fn min(self, s: S) -> S {
        S::min(self, s)
    }

    /// Compares and returns the maximum of two scalars.
    #[inline]
    fn max(self, s: S) -> S {
        S::max(self, s)
    }
}

impl Scalar for f32 {}
impl Scalar for f64 {}

// ===== ScalarConst ===========================================================================================================================================

impl ScalarConst for f32 {
    #[inline]
    fn half() -> f32 {
        0.5f32
    }

    #[inline]
    fn two() -> f32 {
        2f32
    }
}

impl ScalarConst for f64 {
    #[inline]
    fn half() -> f64 {
        0.5f64
    }

    #[inline]
    fn two() -> f64 {
        2f64
    }
}

// ===== Point2 ================================================================================================================================================

impl<S: Scalar> Point2<S> {
    /// Creates and returns a new `Point2` with x and y coordinates.
    #[inline]
    pub fn new(x: S, y: S) -> Point2<S> {
        Point2 { x, y }
    }

    /// Returns a `Point2` which represents the origin (x = 0 and y = 0).
    #[inline]
    pub fn origin() -> Point2<S> {
        Point2::new(S::zero(), S::zero())
    }

    /// Returns the dimension with the smallest extent of this point.
    #[inline]
    pub fn min_dimension(self) -> Dimension2 {
        let Point2 { x, y } = self.abs();
        if x <= y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this point.
    #[inline]
    pub fn max_dimension(self) -> Dimension2 {
        let Point2 { x, y } = self.abs();
        if x > y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the element-wise floor of this point.
    #[inline]
    pub fn floor(self) -> Point2<S> {
        Point2::new(self.x.floor(), self.y.floor())
    }

    /// Returns the element-wise ceiling of this point.
    #[inline]
    pub fn ceil(self) -> Point2<S> {
        Point2::new(self.x.ceil(), self.y.ceil())
    }

    /// Returns the element-wise rounded value of this point.
    #[inline]
    pub fn round(self) -> Point2<S> {
        Point2::new(self.x.round(), self.y.round())
    }

    /// Returns the element-wise truncated value of this point.
    #[inline]
    pub fn trunc(self) -> Point2<S> {
        Point2::new(self.x.trunc(), self.y.trunc())
    }

    /// Returns the element-wise fractional value of this point.
    #[inline]
    pub fn fract(self) -> Point2<S> {
        Point2::new(self.x.fract(), self.y.fract())
    }

    /// Returns the element-wise absolute value of this point.
    #[inline]
    pub fn abs(self) -> Point2<S> {
        Point2::new(self.x.abs(), self.y.abs())
    }

    /// Returns a point with a permutation of the elements of this point.
    #[inline]
    pub fn permute(self, dim_x: Dimension2, dim_y: Dimension2) -> Point2<S> {
        Point2::new(self[dim_x], self[dim_y])
    }
}

impl<S: Scalar> MinMax for Point2<S> {
    /// Returns the element-wise minimum of two points.
    #[inline]
    fn min(self, p: Point2<S>) -> Point2<S> {
        Point2::new(min(self.x, p.x), min(self.y, p.y))
    }

    /// Returns the element-wise maximum of two points.
    #[inline]
    fn max(self, p: Point2<S>) -> Point2<S> {
        Point2::new(max(self.x, p.x), max(self.y, p.y))
    }
}

impl<S: Scalar> Distance for Point2<S> {
    type Output = S;

    /// Computes and returns the distance between two points.
    #[inline]
    fn distance(self, p: Point2<S>) -> S {
        (p - self).length()
    }
}

impl<S: Scalar> RelativeDistance for Point2<S> {
    /// Checks which of the points `p1` and `p2` is closest to this point and returns the closest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn closest(self, p1: Point2<S>, p2: Point2<S>) -> Point2<S> {
        let (dp1, dp2) = (p1 - self, p2 - self);
        if dot(dp1, dp1) <= dot(dp2, dp2) { p1 } else { p2 }
    }

    /// Checks which of the points `p1` and `p2` is farthest from this point and returns the farthest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn farthest(self, p1: Point2<S>, p2: Point2<S>) -> Point2<S> {
        let (dp1, dp2) = (p1 - self, p2 - self);
        if dot(dp1, dp1) > dot(dp2, dp2) { p1 } else { p2 }
    }
}

impl<S: Scalar> Index<Dimension2> for Point2<S> {
    type Output = S;

    #[inline]
    fn index(&self, dim: Dimension2) -> &S {
        match dim {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension2> for Point2<S> {
    #[inline]
    fn index_mut(&mut self, dim: Dimension2) -> &mut S {
        match dim {
            Dimension2::X => &mut self.x,
            Dimension2::Y => &mut self.y,
        }
    }
}

impl<S: Scalar> Add<Vector2<S>> for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn add(self, v: Vector2<S>) -> Point2<S> {
        Point2::new(self.x + v.x, self.y + v.y)
    }
}

impl<S: Scalar> AddAssign<Vector2<S>> for Point2<S> {
    #[inline]
    fn add_assign(&mut self, v: Vector2<S>) {
        self.x += v.x;
        self.y += v.y;
    }
}

impl<S: Scalar> Sub<Vector2<S>> for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn sub(self, v: Vector2<S>) -> Point2<S> {
        Point2::new(self.x - v.x, self.y - v.y)
    }
}

impl<S: Scalar> SubAssign<Vector2<S>> for Point2<S> {
    #[inline]
    fn sub_assign(&mut self, v: Vector2<S>) {
        self.x -= v.x;
        self.y -= v.y;
    }
}

impl<S: Scalar> Sub<Point2<S>> for Point2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, p: Point2<S>) -> Vector2<S> {
        Vector2::new(self.x - p.x, self.y - p.y)
    }
}

impl<S: Scalar> Neg for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn neg(self) -> Point2<S> {
        Point2::new(-self.x, -self.y)
    }
}

impl<S: Scalar> Mul<S> for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, s: S) -> Point2<S> {
        Point2::new(self.x * s, self.y * s)
    }
}

impl Mul<Point2f> for f32 {
    type Output = Point2f;

    #[inline]
    fn mul(self, p: Point2f) -> Point2f {
        p * self
    }
}

impl Mul<Point2d> for f64 {
    type Output = Point2d;

    #[inline]
    fn mul(self, p: Point2d) -> Point2d {
        p * self
    }
}

impl<S: Scalar> MulAssign<S> for Point2<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        self.x *= s;
        self.y *= s;
    }
}

impl<S: Scalar> Div<S> for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn div(self, s: S) -> Point2<S> {
        Point2::new(self.x / s, self.y / s)
    }
}

impl<S: Scalar> DivAssign<S> for Point2<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        self.x /= s;
        self.y /= s;
    }
}

impl<S: Scalar> Transform<Point2<S>> for Transform2<S> {
    type Output = Point2<S>;

    /// Transforms a point.
    #[inline]
    fn transform(&self, p: Point2<S>) -> Point2<S> {
        &self.forward * p
    }
}

impl<S: Scalar> From<Vector2<S>> for Point2<S> {
    #[inline]
    fn from(v: Vector2<S>) -> Point2<S> {
        Point2::new(v.x, v.y)
    }
}

// ===== Vector2 ===============================================================================================================================================

impl<S: Scalar> Vector2<S> {
    /// Creates and returns a new `Vector2` with x and y coordinates.
    #[inline]
    pub fn new(x: S, y: S) -> Vector2<S> {
        Vector2 { x, y }
    }

    /// Returns a `Vector2` which represents the zero vector (x = 0 and y = 0).
    #[inline]
    pub fn zero() -> Vector2<S> {
        Vector2::new(S::zero(), S::zero())
    }

    /// Returns a `Vector2` of length 1 which represents the X axis (x = 1 and y = 0).
    #[inline]
    pub fn x_axis() -> Vector2<S> {
        Vector2::new(S::one(), S::zero())
    }

    /// Returns a `Vector2` of length 1 which represents the Y axis (x = 0 and y = 1).
    #[inline]
    pub fn y_axis() -> Vector2<S> {
        Vector2::new(S::zero(), S::one())
    }

    /// Returns a `Vector2` of length 1 which represents the axis specified by a dimension.
    #[inline]
    pub fn axis(dim: Dimension2) -> Vector2<S> {
        match dim {
            Dimension2::X => Vector2::x_axis(),
            Dimension2::Y => Vector2::y_axis(),
        }
    }

    /// Creates and returns a new `Vector2` which points in the same direction as this vector, but with length 1.
    #[inline]
    pub fn normalize(self) -> Vector2<S> {
        self / self.length()
    }

    /// Returns the dimension with the smallest extent of this vector.
    #[inline]
    pub fn min_dimension(self) -> Dimension2 {
        let Vector2 { x, y } = self.abs();
        if x <= y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this vector.
    #[inline]
    pub fn max_dimension(self) -> Dimension2 {
        let Vector2 { x, y } = self.abs();
        if x > y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the element-wise floor of this vector.
    #[inline]
    pub fn floor(self) -> Vector2<S> {
        Vector2::new(self.x.floor(), self.y.floor())
    }

    /// Returns the element-wise ceiling of this vector.
    #[inline]
    pub fn ceil(self) -> Vector2<S> {
        Vector2::new(self.x.ceil(), self.y.ceil())
    }

    /// Returns the element-wise rounded value of this vector.
    #[inline]
    pub fn round(self) -> Vector2<S> {
        Vector2::new(self.x.round(), self.y.round())
    }

    /// Returns the element-wise truncated value of this vector.
    #[inline]
    pub fn trunc(self) -> Vector2<S> {
        Vector2::new(self.x.trunc(), self.y.trunc())
    }

    /// Returns the element-wise fractional value of this vector.
    #[inline]
    pub fn fract(self) -> Vector2<S> {
        Vector2::new(self.x.fract(), self.y.fract())
    }

    /// Returns the element-wise absolute value of this vector.
    #[inline]
    pub fn abs(self) -> Vector2<S> {
        Vector2::new(self.x.abs(), self.y.abs())
    }

    /// Returns a point with a permutation of the elements of this vector.
    #[inline]
    pub fn permute(self, dim_x: Dimension2, dim_y: Dimension2) -> Vector2<S> {
        Vector2::new(self[dim_x], self[dim_y])
    }
}

impl<S: Scalar> MinMax for Vector2<S> {
    /// Returns the element-wise minimum of two vectors.
    #[inline]
    fn min(self, v: Vector2<S>) -> Vector2<S> {
        Vector2::new(min(self.x, v.x), min(self.y, v.y))
    }

    /// Returns the element-wise maximum of two vectors.
    #[inline]
    fn max(self, v: Vector2<S>) -> Vector2<S> {
        Vector2::new(max(self.x, v.x), max(self.y, v.y))
    }
}

impl<S: Scalar> Length for Vector2<S> {
    type Output = S;

    /// Computes and returns the length of a vector.
    #[inline]
    fn length(self) -> S {
        S::sqrt(dot(self, self))
    }
}

impl<S: Scalar> RelativeLength for Vector2<S> {
    /// Returns the shortest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn shortest(self, v: Vector2<S>) -> Vector2<S> {
        if dot(self, self) <= dot(v, v) { self } else { v }
    }

    /// Returns the longest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn longest(self, v: Vector2<S>) -> Vector2<S> {
        if dot(self, self) > dot(v, v) { self } else { v }
    }
}

impl<S: Scalar> DotProduct<Vector2<S>> for Vector2<S> {
    type Output = S;

    /// Computes and returns the dot product between two vectors.
    #[inline]
    fn dot(self, v: Vector2<S>) -> S {
        self.x * v.x + self.y * v.y
    }
}

impl<S: Scalar> Index<Dimension2> for Vector2<S> {
    type Output = S;

    #[inline]
    fn index(&self, dim: Dimension2) -> &S {
        match dim {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension2> for Vector2<S> {
    #[inline]
    fn index_mut(&mut self, dim: Dimension2) -> &mut S {
        match dim {
            Dimension2::X => &mut self.x,
            Dimension2::Y => &mut self.y,
        }
    }
}

impl<S: Scalar> Add<Vector2<S>> for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn add(self, v: Vector2<S>) -> Vector2<S> {
        Vector2::new(self.x + v.x, self.y + v.y)
    }
}

impl<S: Scalar> AddAssign<Vector2<S>> for Vector2<S> {
    #[inline]
    fn add_assign(&mut self, v: Vector2<S>) {
        self.x += v.x;
        self.y += v.y;
    }
}

impl<S: Scalar> Sub<Vector2<S>> for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn sub(self, v: Vector2<S>) -> Vector2<S> {
        Vector2::new(self.x - v.x, self.y - v.y)
    }
}

impl<S: Scalar> SubAssign<Vector2<S>> for Vector2<S> {
    #[inline]
    fn sub_assign(&mut self, v: Vector2<S>) {
        self.x -= v.x;
        self.y -= v.y;
    }
}

impl<S: Scalar> Neg for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn neg(self) -> Vector2<S> {
        Vector2::new(-self.x, -self.y)
    }
}

impl<S: Scalar> Mul<S> for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, s: S) -> Vector2<S> {
        Vector2::new(self.x * s, self.y * s)
    }
}

impl Mul<Vector2f> for f32 {
    type Output = Vector2f;

    #[inline]
    fn mul(self, v: Vector2f) -> Vector2f {
        v * self
    }
}

impl Mul<Vector2d> for f64 {
    type Output = Vector2d;

    #[inline]
    fn mul(self, v: Vector2d) -> Vector2d {
        v * self
    }
}

impl<S: Scalar> MulAssign<S> for Vector2<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        self.x *= s;
        self.y *= s;
    }
}

impl<S: Scalar> Div<S> for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn div(self, s: S) -> Vector2<S> {
        Vector2::new(self.x / s, self.y / s)
    }
}

impl<S: Scalar> DivAssign<S> for Vector2<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        self.x /= s;
        self.y /= s;
    }
}

impl<S: Scalar> Transform<Vector2<S>> for Transform2<S> {
    type Output = Vector2<S>;

    /// Transforms a vector.
    #[inline]
    fn transform(&self, v: Vector2<S>) -> Vector2<S> {
        &self.forward * v
    }
}

impl<S: Scalar> From<Point2<S>> for Vector2<S> {
    #[inline]
    fn from(p: Point2<S>) -> Vector2<S> {
        Vector2::new(p.x, p.y)
    }
}

// ===== Ray2 ==================================================================================================================================================

impl<S: Scalar> Ray2<S> {
    /// Creates and returns a new `Ray2` with an origin point and direction vector.
    #[inline]
    pub fn new(origin: Point2<S>, direction: Vector2<S>) -> Ray2<S> {
        Ray2 { origin, direction }
    }

    /// Computes and returns a point at a distance along this ray.
    #[inline]
    pub fn at(&self, distance: S) -> Point2<S> {
        self.origin + self.direction * distance
    }
}

impl<S: Scalar> Transform<&Ray2<S>> for Transform2<S> {
    type Output = Ray2<S>;

    /// Transforms a ray.
    #[inline]
    fn transform(&self, ray: &Ray2<S>) -> Ray2<S> {
        Ray2::new(self.transform(ray.origin), self.transform(ray.direction))
    }
}

// ===== BoundingBox2 ==========================================================================================================================================

impl<S: Scalar> BoundingBox2<S> {
    /// Creates and returns a new `BoundingBox2` with minimum and maximum corner points.
    #[inline]
    pub fn new(p1: Point2<S>, p2: Point2<S>) -> BoundingBox2<S> {
        let min = min(p1, p2);
        let max = max(p1, p2);
        BoundingBox2 { min, max }
    }

    /// Returns an empty `BoundingBox2`.
    #[inline]
    pub fn empty() -> BoundingBox2<S> {
        BoundingBox2 {
            min: Point2::new(S::infinity(), S::infinity()),
            max: Point2::new(S::neg_infinity(), S::neg_infinity()),
        }
    }

    /// Returns an infinite `BoundingBox2` which contains all of 2D space.
    #[inline]
    pub fn infinite() -> BoundingBox2<S> {
        BoundingBox2 {
            min: Point2::new(S::neg_infinity(), S::neg_infinity()),
            max: Point2::new(S::infinity(), S::infinity()),
        }
    }

    /// Returns the width (extent in the X dimension) of this bounding box.
    #[inline]
    pub fn width(&self) -> S {
        self.max.x - self.min.x
    }

    /// Returns the height (extent in the Y dimension) of this bounding box.
    #[inline]
    pub fn height(&self) -> S {
        self.max.y - self.min.y
    }

    /// Returns the extent of this bounding box in a dimension.
    #[inline]
    pub fn extent(&self, dim: Dimension2) -> S {
        match dim {
            Dimension2::X => self.width(),
            Dimension2::Y => self.height(),
        }
    }

    /// Returns the dimension with the smallest extent of this bounding box.
    #[inline]
    pub fn min_dimension(&self) -> Dimension2 {
        let d = self.diagonal();
        if d.x <= d.y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this bounding box.
    #[inline]
    pub fn max_dimension(&self) -> Dimension2 {
        let d = self.diagonal();
        if d.x > d.y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the area (width times height) of this bounding box.
    #[inline]
    pub fn area(&self) -> S {
        let d = self.diagonal();
        d.x * d.y
    }

    /// Returns the center point of this bounding box.
    #[inline]
    pub fn center(&self) -> Point2<S> {
        self.min + self.diagonal() * S::half()
    }

    /// Returns a corner point of this bounding box, indicated by an index (which must be between 0 and 3 inclusive).
    #[inline]
    pub fn corner(&self, index: usize) -> Point2<S> {
        debug_assert!(index < 4, "Invalid corner index: {}", index);
        let x = if index & 0b01 == 0 { self.min.x } else { self.max.x };
        let y = if index & 0b10 == 0 { self.min.y } else { self.max.y };
        Point2::new(x, y)
    }

    /// Returns the diagonal of this bounding box as a vector.
    #[inline]
    pub fn diagonal(&self) -> Vector2<S> {
        self.max - self.min
    }

    /// Checks if two bounding boxes overlap.
    #[inline]
    pub fn overlaps(&self, bb: &BoundingBox2<S>) -> bool {
        //@formatter:off
        self.max.x >= bb.min.x && self.min.x <= bb.max.x &&
        self.max.y >= bb.min.y && self.min.y <= bb.max.y
        //@formatter:on
    }

    /// Checks if a point is inside this bounding box.
    #[inline]
    pub fn is_inside(&self, p: Point2<S>) -> bool {
        //@formatter:off
        p.x >= self.min.x && p.x <= self.max.x &&
        p.y >= self.min.y && p.y <= self.max.y
        //@formatter:on
    }

    /// Computes the closest intersection of this bounding box with a ray within a range.
    ///
    /// Returns a `Some` containing the closest intersection, or `None` if the ray does not intersect the bounding box within the range.
    pub fn intersect_ray(&self, ray: &Ray2<S>, range: &Range<S>) -> Option<S> {
        let (start, end) = (range.start, range.end);

        let d1 = (self.min.x - ray.origin.x) / ray.direction.x;
        let d2 = (self.max.x - ray.origin.x) / ray.direction.x;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start > end {
            return None;
        }

        let d1 = (self.min.y - ray.origin.y) / ray.direction.y;
        let d2 = (self.max.y - ray.origin.y) / ray.direction.y;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start <= end {
            Some(start)
        } else {
            None
        }
    }
}

impl<S: Scalar> Union<&BoundingBox2<S>> for BoundingBox2<S> {
    type Output = BoundingBox2<S>;

    /// Computes and returns the union between two bounding boxes.
    ///
    /// The union is the smallest bounding box that contains both bounding boxes.
    #[inline]
    fn union(self, bb: &BoundingBox2<S>) -> BoundingBox2<S> {
        BoundingBox2::new(min(self.min, bb.min), max(self.max, bb.max))
    }
}

impl<S: Scalar> Union<Point2<S>> for BoundingBox2<S> {
    type Output = BoundingBox2<S>;

    /// Computes and returns the union between this bounding box and a point.
    ///
    /// The union is the smallest bounding box that contains both the bounding box and the point.
    #[inline]
    fn union(self, p: Point2<S>) -> BoundingBox2<S> {
        BoundingBox2::new(min(self.min, p), max(self.max, p))
    }
}

impl<S: Scalar> Intersection<&BoundingBox2<S>> for BoundingBox2<S> {
    type Output = BoundingBox2<S>;

    /// Computes and returns the intersection between two bounding boxes.
    ///
    /// The intersection is the largest bounding box that contains the region where the two bounding boxes overlap.
    ///
    /// Returns `Some` when the bounding boxes overlap; `None` if the bounding boxes do not overlap.
    #[inline]
    fn intersection(self, bb: &BoundingBox2<S>) -> Option<BoundingBox2<S>> {
        if self.overlaps(bb) {
            Some(BoundingBox2::new(max(self.min, bb.min), min(self.max, bb.max)))
        } else {
            None
        }
    }
}

impl<S: Scalar> Transform<&BoundingBox2<S>> for Transform2<S> {
    type Output = BoundingBox2<S>;

    /// Transforms a bounding box.
    fn transform(&self, bb: &BoundingBox2<S>) -> BoundingBox2<S> {
        let o = self.transform(bb.min);
        let d = self.transform(bb.diagonal());

        let (mut min_corner, mut max_corner) = (o, o);
        for i in 1..4 {
            let mut corner = o;
            if i & 0b01 != 0 { corner.x += d.x; }
            if i & 0b10 != 0 { corner.y += d.y; }

            min_corner = min(min_corner, corner);
            max_corner = max(max_corner, corner);
        }

        BoundingBox2::new(min_corner, max_corner)
    }
}

// ===== Matrix3x3 =============================================================================================================================================

impl<S: Scalar> Matrix3x3<S> {
    /// Creates and returns a new `Matrix3x3` with the specified elements.
    #[inline]
    pub fn new(m: [S; 9]) -> Matrix3x3<S> {
        Matrix3x3 { m }
    }

    /// Returns a `Matrix3x3` which represents the identity matrix.
    #[inline]
    pub fn identity() -> Matrix3x3<S> {
        let (o, i) = (S::zero(), S::one());

        Matrix3x3::new([
            i, o, o,
            o, i, o,
            o, o, i,
        ])
    }

    /// Returns a translation matrix which translates over a vector.
    #[inline]
    pub fn translate(v: Vector2<S>) -> Matrix3x3<S> {
        let (o, i) = (S::zero(), S::one());

        Matrix3x3::new([
            i, o, v.x,
            o, i, v.y,
            o, o, i,
        ])
    }

    /// Returns a rotation matrix which rotates around the origin.
    #[inline]
    pub fn rotate(angle: S) -> Matrix3x3<S> {
        let (o, i) = (S::zero(), S::one());
        let (sin, cos) = angle.sin_cos();

        Matrix3x3::new([
            cos, -sin, o,
            sin, cos, o,
            o, o, i,
        ])
    }

    /// Returns a matrix which scales by factors in the X and Y dimensions.
    #[inline]
    pub fn scale(sx: S, sy: S) -> Matrix3x3<S> {
        let (o, i) = (S::zero(), S::one());

        Matrix3x3::new([
            sx, o, o,
            o, sy, o,
            o, o, i,
        ])
    }

    /// Returns a matrix which scales uniformly in all dimensions by a factor.
    #[inline]
    pub fn scale_uniform(s: S) -> Matrix3x3<S> {
        let (o, i) = (S::zero(), S::one());

        Matrix3x3::new([
            s, o, o,
            o, s, o,
            o, o, i,
        ])
    }

    /// Returns an element at a row and column of the matrix.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> S {
        debug_assert!(row < 3, "Invalid row index: {}", row);
        debug_assert!(col < 3, "Invalid column index: {}", row);
        self.m[row * 3 + col]
    }

    /// Returns a mutable reference to an element at a row and column of the matrix.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut S {
        debug_assert!(row < 3, "Invalid row index: {}", row);
        debug_assert!(col < 3, "Invalid column index: {}", row);
        &mut self.m[row * 3 + col]
    }

    /// Sets the value of an element at a row and column of the matrix.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: S) {
        debug_assert!(row < 3, "Invalid row index: {}", row);
        debug_assert!(col < 3, "Invalid column index: {}", row);
        self.m[row * 3 + col] = value;
    }

    /// Returns the transpose of this matrix.
    #[inline]
    pub fn transpose(&self) -> Matrix3x3<S> {
        Matrix3x3::new([
            self.m[0], self.m[3], self.m[6],
            self.m[1], self.m[4], self.m[7],
            self.m[2], self.m[5], self.m[8],
        ])
    }

    /// Computes and returns the inverse of this matrix.
    ///
    /// If this matrix is singular, a `NonInvertibleMatrixError` is returned.
    pub fn inverse(&self) -> Result<Matrix3x3<S>, NonInvertibleMatrixError> {
        let det = self.m[0] * self.m[4] * self.m[8] + self.m[1] * self.m[5] * self.m[6] + self.m[2] * self.m[3] * self.m[7]
            - self.m[2] * self.m[4] * self.m[6] - self.m[1] * self.m[3] * self.m[8] - self.m[0] * self.m[5] * self.m[7];

        if det != S::zero() {
            let inv_det = det.recip();
            Ok(Matrix3x3::new([
                (self.m[4] * self.m[8] - self.m[5] * self.m[7]) * inv_det,
                (self.m[2] * self.m[7] - self.m[1] * self.m[8]) * inv_det,
                (self.m[1] * self.m[5] - self.m[2] * self.m[4]) * inv_det,
                (self.m[5] * self.m[6] - self.m[3] * self.m[8]) * inv_det,
                (self.m[0] * self.m[8] - self.m[2] * self.m[6]) * inv_det,
                (self.m[2] * self.m[3] - self.m[0] * self.m[5]) * inv_det,
                (self.m[3] * self.m[7] - self.m[4] * self.m[6]) * inv_det,
                (self.m[1] * self.m[6] - self.m[0] * self.m[7]) * inv_det,
                (self.m[0] * self.m[4] - self.m[1] * self.m[3]) * inv_det,
            ]))
        } else {
            Err(NonInvertibleMatrixError)
        }
    }
}

impl<S: Scalar> Mul<S> for &Matrix3x3<S> {
    type Output = Matrix3x3<S>;

    #[inline]
    fn mul(self, s: S) -> Matrix3x3<S> {
        Matrix3x3::new(array::from_fn(|i| self.m[i] * s))
    }
}

impl Mul<&Matrix3x3f> for f32 {
    type Output = Matrix3x3f;

    #[inline]
    fn mul(self, m: &Matrix3x3f) -> Matrix3x3f {
        m * self
    }
}

impl Mul<&Matrix3x3d> for f64 {
    type Output = Matrix3x3d;

    #[inline]
    fn mul(self, m: &Matrix3x3d) -> Matrix3x3d {
        m * self
    }
}

impl<S: Scalar> MulAssign<S> for &mut Matrix3x3<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        for m in &mut self.m { *m *= s; }
    }
}

impl<S: Scalar> Div<S> for &Matrix3x3<S> {
    type Output = Matrix3x3<S>;

    #[inline]
    fn div(self, s: S) -> Matrix3x3<S> {
        Matrix3x3::new(array::from_fn(|i| self.m[i] / s))
    }
}

impl<S: Scalar> DivAssign<S> for &mut Matrix3x3<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        for m in &mut self.m { *m /= s; }
    }
}

impl<S: Scalar> Mul<Point2<S>> for &Matrix3x3<S> {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, p: Point2<S>) -> Point2<S> {
        let x = self.m[0] * p.x + self.m[1] * p.y + self.m[2];
        let y = self.m[3] * p.x + self.m[4] * p.y + self.m[5];
        let w = self.m[6] * p.x + self.m[7] * p.y + self.m[8];
        Point2::new(x / w, y / w)
    }
}

impl<S: Scalar> Mul<&Matrix3x3<S>> for Point2<S> {
    type Output = Point2<S>;

    #[inline]
    fn mul(self, m: &Matrix3x3<S>) -> Point2<S> {
        let x = self.x * m.m[0] + self.y * m.m[3] + m.m[6];
        let y = self.x * m.m[1] + self.y * m.m[4] + m.m[7];
        let w = self.x * m.m[2] + self.y * m.m[5] + m.m[8];
        Point2::new(x / w, y / w)
    }
}

impl<S: Scalar> Mul<Vector2<S>> for &Matrix3x3<S> {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, v: Vector2<S>) -> Vector2<S> {
        let x = self.m[0] * v.x + self.m[1] * v.y;
        let y = self.m[3] * v.x + self.m[4] * v.y;
        Vector2::new(x, y)
    }
}

impl<S: Scalar> Mul<&Matrix3x3<S>> for Vector2<S> {
    type Output = Vector2<S>;

    #[inline]
    fn mul(self, m: &Matrix3x3<S>) -> Vector2<S> {
        let x = self.x * m.m[0] + self.y * m.m[3];
        let y = self.x * m.m[1] + self.y * m.m[4];
        Vector2::new(x, y)
    }
}

impl<S: Scalar> Mul<&Matrix3x3<S>> for &Matrix3x3<S> {
    type Output = Matrix3x3<S>;

    #[inline]
    fn mul(self, m: &Matrix3x3<S>) -> Matrix3x3<S> {
        Matrix3x3::new([
            self.m[0] * m.m[0] + self.m[1] * m.m[3] + self.m[2] * m.m[6],
            self.m[0] * m.m[1] + self.m[1] * m.m[4] + self.m[2] * m.m[7],
            self.m[0] * m.m[2] + self.m[1] * m.m[5] + self.m[2] * m.m[8],
            self.m[3] * m.m[0] + self.m[4] * m.m[3] + self.m[5] * m.m[6],
            self.m[3] * m.m[1] + self.m[4] * m.m[4] + self.m[5] * m.m[7],
            self.m[3] * m.m[2] + self.m[4] * m.m[5] + self.m[5] * m.m[8],
            self.m[6] * m.m[0] + self.m[7] * m.m[3] + self.m[8] * m.m[6],
            self.m[6] * m.m[1] + self.m[7] * m.m[4] + self.m[8] * m.m[7],
            self.m[6] * m.m[2] + self.m[7] * m.m[5] + self.m[8] * m.m[8],
        ])
    }
}

// ===== Transform2 ============================================================================================================================================

impl<S: Scalar> Transform2<S> {
    /// Creates and returns a new `Transform2` with a transformation matrix and its inverse.
    #[inline]
    pub fn new(forward: Matrix3x3<S>, inverse: Matrix3x3<S>) -> Transform2<S> {
        Transform2 { forward, inverse }
    }

    /// Returns a `Transform2` which represents the identity transform.
    #[inline]
    pub fn identity() -> Transform2<S> {
        let forward = Matrix3x3::identity();
        let inverse = forward.clone();
        Transform2::new(forward, inverse)
    }

    /// Returns a translation transform over a vector.
    #[inline]
    pub fn translate(v: Vector2<S>) -> Transform2<S> {
        let forward = Matrix3x3::translate(v);
        let inverse = Matrix3x3::translate(-v);
        Transform2::new(forward, inverse)
    }

    /// Returns a rotation transform which rotates around the origin.
    #[inline]
    pub fn rotate(angle: S) -> Transform2<S> {
        let forward = Matrix3x3::rotate(angle);
        let inverse = forward.transpose();
        Transform2::new(forward, inverse)
    }

    /// Returns a transform which scales by factors in the X and Y dimensions.
    #[inline]
    pub fn scale(sx: S, sy: S) -> Transform2<S> {
        let forward = Matrix3x3::scale(sx, sy);
        let inverse = Matrix3x3::scale(sx.recip(), sy.recip());
        Transform2::new(forward, inverse)
    }

    /// Returns a transform which scales uniformly in all dimensions by a factor.
    #[inline]
    pub fn scale_uniform(s: S) -> Transform2<S> {
        let forward = Matrix3x3::scale_uniform(s);
        let inverse = Matrix3x3::scale_uniform(s.recip());
        Transform2::new(forward, inverse)
    }

    /// Computes and returns a composite transform, which first applies this and then the other transform.
    #[inline]
    pub fn and_then(&self, transform: &Transform2<S>) -> Transform2<S> {
        let forward = &transform.forward * &self.forward;
        let inverse = &self.inverse * &transform.inverse;
        Transform2::new(forward, inverse)
    }

    /// Returns the inverse of this transform.
    #[inline]
    pub fn inverse(&self) -> Transform2<S> {
        let forward = self.inverse.clone();
        let inverse = self.forward.clone();
        Transform2::new(forward, inverse)
    }
}

impl<S: Scalar> TryFrom<Matrix3x3<S>> for Transform2<S> {
    type Error = NonInvertibleMatrixError;

    #[inline]
    fn try_from(forward: Matrix3x3<S>) -> Result<Transform2<S>, NonInvertibleMatrixError> {
        let inverse = forward.inverse()?;
        Ok(Transform2::new(forward, inverse))
    }
}

// ===== Point3 ================================================================================================================================================

impl<S: Scalar> Point3<S> {
    /// Creates and returns a new `Point3` with x, y and z coordinates.
    #[inline]
    pub fn new(x: S, y: S, z: S) -> Point3<S> {
        Point3 { x, y, z }
    }

    /// Returns a `Point3` which represents the origin (x = 0, y = 0 and z = 0).
    #[inline]
    pub fn origin() -> Point3<S> {
        Point3::new(S::zero(), S::zero(), S::zero())
    }

    /// Returns the dimension with the smallest extent of this point.
    #[inline]
    pub fn min_dimension(self) -> Dimension3 {
        let Point3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this point.
    #[inline]
    pub fn max_dimension(self) -> Dimension3 {
        let Point3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this point.
    #[inline]
    pub fn floor(self) -> Point3<S> {
        Point3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the element-wise ceiling of this point.
    #[inline]
    pub fn ceil(self) -> Point3<S> {
        Point3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the element-wise rounded value of this point.
    #[inline]
    pub fn round(self) -> Point3<S> {
        Point3::new(self.x.round(), self.y.round(), self.z.round())
    }

    /// Returns the element-wise truncated value of this point.
    #[inline]
    pub fn trunc(self) -> Point3<S> {
        Point3::new(self.x.trunc(), self.y.trunc(), self.z.trunc())
    }

    /// Returns the element-wise fractional value of this point.
    #[inline]
    pub fn fract(self) -> Point3<S> {
        Point3::new(self.x.fract(), self.y.fract(), self.z.fract())
    }

    /// Returns the element-wise absolute value of this point.
    #[inline]
    pub fn abs(self) -> Point3<S> {
        Point3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a point with a permutation of the elements of this point.
    #[inline]
    pub fn permute(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Point3<S> {
        Point3::new(self[dim_x], self[dim_y], self[dim_z])
    }
}

impl<S: Scalar> MinMax for Point3<S> {
    /// Returns the element-wise minimum of two points.
    #[inline]
    fn min(self, p: Point3<S>) -> Point3<S> {
        Point3::new(min(self.x, p.x), min(self.y, p.y), min(self.z, p.z))
    }

    /// Returns the element-wise maximum of two points.
    #[inline]
    fn max(self, p: Point3<S>) -> Point3<S> {
        Point3::new(max(self.x, p.x), max(self.y, p.y), max(self.z, p.z))
    }
}

impl<S: Scalar> Distance for Point3<S> {
    type Output = S;

    /// Computes and returns the distance between two points.
    #[inline]
    fn distance(self, p: Point3<S>) -> S {
        (p - self).length()
    }
}

impl<S: Scalar> RelativeDistance for Point3<S> {
    /// Checks which of the points `p1` and `p2` is closest to this point and returns the closest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn closest(self, p1: Point3<S>, p2: Point3<S>) -> Point3<S> {
        let (dp1, dp2) = (p1 - self, p2 - self);
        if dot(dp1, dp1) <= dot(dp2, dp2) { p1 } else { p2 }
    }

    /// Checks which of the points `p1` and `p2` is farthest from this point and returns the farthest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    #[inline]
    fn farthest(self, p1: Point3<S>, p2: Point3<S>) -> Point3<S> {
        let (dp1, dp2) = (p1 - self, p2 - self);
        if dot(dp1, dp1) > dot(dp2, dp2) { p1 } else { p2 }
    }
}

impl<S: Scalar> Index<Dimension3> for Point3<S> {
    type Output = S;

    #[inline]
    fn index(&self, dim: Dimension3) -> &S {
        match dim {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension3> for Point3<S> {
    #[inline]
    fn index_mut(&mut self, dim: Dimension3) -> &mut S {
        match dim {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl<S: Scalar> Add<Vector3<S>> for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn add(self, v: Vector3<S>) -> Point3<S> {
        Point3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

impl<S: Scalar> AddAssign<Vector3<S>> for Point3<S> {
    #[inline]
    fn add_assign(&mut self, v: Vector3<S>) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }
}

impl<S: Scalar> Sub<Vector3<S>> for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn sub(self, v: Vector3<S>) -> Point3<S> {
        Point3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
}

impl<S: Scalar> SubAssign<Vector3<S>> for Point3<S> {
    #[inline]
    fn sub_assign(&mut self, v: Vector3<S>) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

impl<S: Scalar> Sub<Point3<S>> for Point3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, p: Point3<S>) -> Vector3<S> {
        Vector3::new(self.x - p.x, self.y - p.y, self.z - p.z)
    }
}

impl<S: Scalar> Neg for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn neg(self) -> Point3<S> {
        Point3::new(-self.x, -self.y, -self.z)
    }
}

impl<S: Scalar> Mul<S> for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, s: S) -> Point3<S> {
        Point3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Point3f> for f32 {
    type Output = Point3f;

    #[inline]
    fn mul(self, p: Point3f) -> Point3f {
        p * self
    }
}

impl Mul<Point3d> for f64 {
    type Output = Point3d;

    #[inline]
    fn mul(self, p: Point3d) -> Point3d {
        p * self
    }
}

impl<S: Scalar> MulAssign<S> for Point3<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl<S: Scalar> Div<S> for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn div(self, s: S) -> Point3<S> {
        Point3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl<S: Scalar> DivAssign<S> for Point3<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl<S: Scalar> Transform<Point3<S>> for Transform3<S> {
    type Output = Point3<S>;

    /// Transforms a point.
    #[inline]
    fn transform(&self, p: Point3<S>) -> Point3<S> {
        &self.forward * p
    }
}

impl<S: Scalar> From<Vector3<S>> for Point3<S> {
    #[inline]
    fn from(v: Vector3<S>) -> Point3<S> {
        Point3::new(v.x, v.y, v.z)
    }
}

// ===== Vector3 ===============================================================================================================================================

impl<S: Scalar> Vector3<S> {
    /// Creates and returns a new `Vector3` with x, y and z coordinates.
    #[inline]
    pub fn new(x: S, y: S, z: S) -> Vector3<S> {
        Vector3 { x, y, z }
    }

    /// Returns a `Vector3` which represents the zero vector (x = 0, y = 0 and z = 0).
    #[inline]
    pub fn zero() -> Vector3<S> {
        Vector3::new(S::zero(), S::zero(), S::zero())
    }

    /// Returns a `Vector3` of length 1 which represents the X axis (x = 1, y = 0 and z = 0).
    #[inline]
    pub fn x_axis() -> Vector3<S> {
        Vector3::new(S::one(), S::zero(), S::zero())
    }

    /// Returns a `Vector3` of length 1 which represents the Y axis (x = 0, y = 1 and z = 0).
    #[inline]
    pub fn y_axis() -> Vector3<S> {
        Vector3::new(S::zero(), S::one(), S::zero())
    }

    /// Returns a `Vector3` of length 1 which represents the Z axis (x = 0, y = 0 and z = 1).
    #[inline]
    pub fn z_axis() -> Vector3<S> {
        Vector3::new(S::zero(), S::zero(), S::one())
    }

    /// Returns a `Vector3` of length 1 which represents the axis specified by a dimension.
    #[inline]
    pub fn axis(dim: Dimension3) -> Vector3<S> {
        match dim {
            Dimension3::X => Vector3::x_axis(),
            Dimension3::Y => Vector3::y_axis(),
            Dimension3::Z => Vector3::z_axis(),
        }
    }

    /// Creates and returns a new `Vector3` which points in the same direction as this vector, but with length 1.
    #[inline]
    pub fn normalize(self) -> Vector3<S> {
        self / self.length()
    }

    /// Returns the dimension with the smallest extent of this vector.
    #[inline]
    pub fn min_dimension(self) -> Dimension3 {
        let Vector3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this vector.
    #[inline]
    pub fn max_dimension(self) -> Dimension3 {
        let Vector3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this vector.
    #[inline]
    pub fn floor(self) -> Vector3<S> {
        Vector3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the element-wise ceiling of this vector.
    #[inline]
    pub fn ceil(self) -> Vector3<S> {
        Vector3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the element-wise rounded value of this vector.
    #[inline]
    pub fn round(self) -> Vector3<S> {
        Vector3::new(self.x.round(), self.y.round(), self.z.round())
    }

    /// Returns the element-wise truncated value of this vector.
    #[inline]
    pub fn trunc(self) -> Vector3<S> {
        Vector3::new(self.x.trunc(), self.y.trunc(), self.z.trunc())
    }

    /// Returns the element-wise fractional value of this vector.
    #[inline]
    pub fn fract(self) -> Vector3<S> {
        Vector3::new(self.x.fract(), self.y.fract(), self.z.fract())
    }

    /// Returns the element-wise absolute value of this vector.
    #[inline]
    pub fn abs(self) -> Vector3<S> {
        Vector3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a point with a permutation of the elements of this vector.
    #[inline]
    pub fn permute(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Vector3<S> {
        Vector3::new(self[dim_x], self[dim_y], self[dim_z])
    }
}

impl<S: Scalar> MinMax for Vector3<S> {
    /// Returns the element-wise minimum of two vectors.
    #[inline]
    fn min(self, v: Vector3<S>) -> Vector3<S> {
        Vector3::new(min(self.x, v.x), min(self.y, v.y), min(self.z, v.z))
    }

    /// Returns the element-wise maximum of two vectors.
    #[inline]
    fn max(self, v: Vector3<S>) -> Vector3<S> {
        Vector3::new(max(self.x, v.x), max(self.y, v.y), max(self.z, v.z))
    }
}

impl<S: Scalar> Length for Vector3<S> {
    type Output = S;

    /// Computes and returns the length of a vector.
    #[inline]
    fn length(self) -> S {
        S::sqrt(dot(self, self))
    }
}

impl<S: Scalar> RelativeLength for Vector3<S> {
    /// Returns the shortest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn shortest(self, v: Vector3<S>) -> Vector3<S> {
        if dot(self, self) <= dot(v, v) { self } else { v }
    }

    /// Returns the longest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn longest(self, v: Vector3<S>) -> Vector3<S> {
        if dot(self, self) > dot(v, v) { self } else { v }
    }
}

impl<S: Scalar> DotProduct<Vector3<S>> for Vector3<S> {
    type Output = S;

    /// Computes and returns the dot product between two vectors.
    #[inline]
    fn dot(self, v: Vector3<S>) -> S {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}

impl<S: Scalar> DotProduct<Normal3<S>> for Vector3<S> {
    type Output = S;

    /// Computes and returns the dot product between this vector and a normal.
    #[inline]
    fn dot(self, n: Normal3<S>) -> S {
        self.x * n.x + self.y * n.y + self.z * n.z
    }
}

impl<S: Scalar> CrossProduct<Vector3<S>> for Vector3<S> {
    type Output = Vector3<S>;

    /// Computes and returns the cross product between two vectors.
    #[inline]
    fn cross(self, v: Vector3<S>) -> Vector3<S> {
        Vector3::new(self.y * v.z - self.z * v.y, self.z * v.x - self.x * v.z, self.x * v.y - self.y * v.x)
    }
}

impl<S: Scalar> Index<Dimension3> for Vector3<S> {
    type Output = S;

    #[inline]
    fn index(&self, dim: Dimension3) -> &S {
        match dim {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension3> for Vector3<S> {
    #[inline]
    fn index_mut(&mut self, dim: Dimension3) -> &mut S {
        match dim {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl<S: Scalar> Add<Vector3<S>> for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn add(self, v: Vector3<S>) -> Vector3<S> {
        Vector3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

impl<S: Scalar> AddAssign<Vector3<S>> for Vector3<S> {
    #[inline]
    fn add_assign(&mut self, v: Vector3<S>) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }
}

impl<S: Scalar> Sub<Vector3<S>> for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn sub(self, v: Vector3<S>) -> Vector3<S> {
        Vector3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
}

impl<S: Scalar> SubAssign<Vector3<S>> for Vector3<S> {
    #[inline]
    fn sub_assign(&mut self, v: Vector3<S>) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

impl<S: Scalar> Neg for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn neg(self) -> Vector3<S> {
        Vector3::new(-self.x, -self.y, -self.z)
    }
}

impl<S: Scalar> Mul<S> for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, s: S) -> Vector3<S> {
        Vector3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Vector3f> for f32 {
    type Output = Vector3f;

    #[inline]
    fn mul(self, v: Vector3f) -> Vector3f {
        v * self
    }
}

impl Mul<Vector3d> for f64 {
    type Output = Vector3d;

    #[inline]
    fn mul(self, v: Vector3d) -> Vector3d {
        v * self
    }
}

impl<S: Scalar> MulAssign<S> for Vector3<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl<S: Scalar> Div<S> for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn div(self, s: S) -> Vector3<S> {
        Vector3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl<S: Scalar> DivAssign<S> for Vector3<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl<S: Scalar> Transform<Vector3<S>> for Transform3<S> {
    type Output = Vector3<S>;

    /// Transforms a vector.
    #[inline]
    fn transform(&self, v: Vector3<S>) -> Vector3<S> {
        &self.forward * v
    }
}

impl<S: Scalar> From<Point3<S>> for Vector3<S> {
    #[inline]
    fn from(p: Point3<S>) -> Vector3<S> {
        Vector3::new(p.x, p.y, p.z)
    }
}

impl<S: Scalar> From<Normal3<S>> for Vector3<S> {
    #[inline]
    fn from(n: Normal3<S>) -> Self {
        Vector3::new(n.x, n.y, n.z)
    }
}

// ===== Normal3 ===============================================================================================================================================

impl<S: Scalar> Normal3<S> {
    /// Creates and returns a new `Normal3` with x, y and z coordinates.
    #[inline]
    pub fn new(x: S, y: S, z: S) -> Normal3<S> {
        Normal3 { x, y, z }
    }

    /// Returns a `Normal3` which represents the zero normal (x = 0, y = 0 and z = 0).
    #[inline]
    pub fn zero() -> Normal3<S> {
        Normal3::new(S::zero(), S::zero(), S::zero())
    }

    /// Returns a `Normal3` of length 1 which represents the X axis (x = 1, y = 0 and z = 0).
    #[inline]
    pub fn x_axis() -> Normal3<S> {
        Normal3::new(S::one(), S::zero(), S::zero())
    }

    /// Returns a `Normal3` of length 1 which represents the Y axis (x = 0, y = 1 and z = 0).
    #[inline]
    pub fn y_axis() -> Normal3<S> {
        Normal3::new(S::zero(), S::one(), S::zero())
    }

    /// Returns a `Normal3` of length 1 which represents the Z axis (x = 0, y = 0 and z = 1).
    #[inline]
    pub fn z_axis() -> Normal3<S> {
        Normal3::new(S::zero(), S::zero(), S::one())
    }

    /// Returns a `Normal3` of length 1 which represents the axis specified by a dimension.
    #[inline]
    pub fn axis(dim: Dimension3) -> Normal3<S> {
        match dim {
            Dimension3::X => Normal3::x_axis(),
            Dimension3::Y => Normal3::y_axis(),
            Dimension3::Z => Normal3::z_axis(),
        }
    }

    /// Creates and returns a new `Normal3` which points in the same direction as this normal, but with length 1.
    #[inline]
    pub fn normalize(self) -> Normal3<S> {
        self / self.length()
    }

    /// Returns the dimension with the smallest extent of this normal.
    #[inline]
    pub fn min_dimension(self) -> Dimension3 {
        let Normal3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this normal.
    #[inline]
    pub fn max_dimension(self) -> Dimension3 {
        let Normal3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this normal.
    #[inline]
    pub fn floor(self) -> Normal3<S> {
        Normal3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the element-wise ceiling of this normal.
    #[inline]
    pub fn ceil(self) -> Normal3<S> {
        Normal3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the element-wise rounded value of this normal.
    #[inline]
    pub fn round(self) -> Normal3<S> {
        Normal3::new(self.x.round(), self.y.round(), self.z.round())
    }

    /// Returns the element-wise truncated value of this normal.
    #[inline]
    pub fn trunc(self) -> Normal3<S> {
        Normal3::new(self.x.trunc(), self.y.trunc(), self.z.trunc())
    }

    /// Returns the element-wise fractional value of this normal.
    #[inline]
    pub fn fract(self) -> Normal3<S> {
        Normal3::new(self.x.fract(), self.y.fract(), self.z.fract())
    }

    /// Returns the element-wise absolute value of this normal.
    #[inline]
    pub fn abs(self) -> Normal3<S> {
        Normal3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a point with a permutation of the elements of this normal.
    #[inline]
    pub fn permute(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Normal3<S> {
        Normal3::new(self[dim_x], self[dim_y], self[dim_z])
    }
}

impl<S: Scalar> MinMax for Normal3<S> {
    /// Returns the element-wise minimum of two normals.
    #[inline]
    fn min(self, n: Normal3<S>) -> Normal3<S> {
        Normal3::new(min(self.x, n.x), min(self.y, n.y), min(self.z, n.z))
    }

    /// Returns the element-wise maximum of two normals.
    #[inline]
    fn max(self, n: Normal3<S>) -> Normal3<S> {
        Normal3::new(max(self.x, n.x), max(self.y, n.y), max(self.z, n.z))
    }
}

impl<S: Scalar> Length for Normal3<S> {
    type Output = S;

    /// Computes and returns the length of a normal.
    #[inline]
    fn length(self) -> S {
        S::sqrt(dot(self, self))
    }
}

impl<S: Scalar> RelativeLength for Normal3<S> {
    /// Returns the shortest of two normals.
    ///
    /// This is more computationally efficient than computing the lengths of the normals and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn shortest(self, n: Normal3<S>) -> Normal3<S> {
        if dot(self, self) <= dot(n, n) { self } else { n }
    }

    /// Returns the longest of two normals.
    ///
    /// This is more computationally efficient than computing the lengths of the normals and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    #[inline]
    fn longest(self, n: Normal3<S>) -> Normal3<S> {
        if dot(self, self) > dot(n, n) { self } else { n }
    }
}

impl<S: Scalar> DotProduct<Normal3<S>> for Normal3<S> {
    type Output = S;

    /// Computes and returns the dot product between two normals.
    #[inline]
    fn dot(self, n: Normal3<S>) -> S {
        self.x * n.x + self.y * n.y + self.z * n.z
    }
}

impl<S: Scalar> DotProduct<Vector3<S>> for Normal3<S> {
    type Output = S;

    /// Computes and returns the dot product between this normal and a vector.
    #[inline]
    fn dot(self, v: Vector3<S>) -> S {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}

impl<S: Scalar> Index<Dimension3> for Normal3<S> {
    type Output = S;

    #[inline]
    fn index(&self, dim: Dimension3) -> &S {
        match dim {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl<S: Scalar> IndexMut<Dimension3> for Normal3<S> {
    #[inline]
    fn index_mut(&mut self, dim: Dimension3) -> &mut S {
        match dim {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl<S: Scalar> Add<Normal3<S>> for Normal3<S> {
    type Output = Normal3<S>;

    #[inline]
    fn add(self, n: Normal3<S>) -> Normal3<S> {
        Normal3::new(self.x + n.x, self.y + n.y, self.z + n.z)
    }
}

impl<S: Scalar> AddAssign<Normal3<S>> for Normal3<S> {
    #[inline]
    fn add_assign(&mut self, n: Normal3<S>) {
        self.x += n.x;
        self.y += n.y;
        self.z += n.z;
    }
}

impl<S: Scalar> Sub<Normal3<S>> for Normal3<S> {
    type Output = Normal3<S>;

    #[inline]
    fn sub(self, n: Normal3<S>) -> Normal3<S> {
        Normal3::new(self.x - n.x, self.y - n.y, self.z - n.z)
    }
}

impl<S: Scalar> SubAssign<Normal3<S>> for Normal3<S> {
    #[inline]
    fn sub_assign(&mut self, n: Normal3<S>) {
        self.x -= n.x;
        self.y -= n.y;
        self.z -= n.z;
    }
}

impl<S: Scalar> Neg for Normal3<S> {
    type Output = Normal3<S>;

    #[inline]
    fn neg(self) -> Normal3<S> {
        Normal3::new(-self.x, -self.y, -self.z)
    }
}

impl<S: Scalar> Mul<S> for Normal3<S> {
    type Output = Normal3<S>;

    #[inline]
    fn mul(self, s: S) -> Normal3<S> {
        Normal3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Normal3f> for f32 {
    type Output = Normal3f;

    #[inline]
    fn mul(self, n: Normal3f) -> Normal3f {
        n * self
    }
}

impl Mul<Normal3d> for f64 {
    type Output = Normal3d;

    #[inline]
    fn mul(self, n: Normal3d) -> Normal3d {
        n * self
    }
}

impl<S: Scalar> MulAssign<S> for Normal3<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl<S: Scalar> Div<S> for Normal3<S> {
    type Output = Normal3<S>;

    #[inline]
    fn div(self, s: S) -> Normal3<S> {
        Normal3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl<S: Scalar> DivAssign<S> for Normal3<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl<S: Scalar> Transform<Normal3<S>> for Transform3<S> {
    type Output = Normal3<S>;

    /// Transforms a normal.
    ///
    /// Note that transforming a normal is different from transforming a vector; normals are transformed by applying the transpose of the inverse
    /// transformation matrix. This difference is the main reason why there is a separate type for normals, which should be used instead of `Vector3`.
    #[inline]
    fn transform(&self, n: Normal3<S>) -> Normal3<S> {
        // Normals are transformed by the transpose of the inverse
        Normal3::from(Vector3::from(n) * &self.inverse)
    }
}

impl<S: Scalar> From<Vector3<S>> for Normal3<S> {
    #[inline]
    fn from(v: Vector3<S>) -> Self {
        Normal3::new(v.x, v.y, v.z)
    }
}

// ===== Ray3 ==================================================================================================================================================

impl<S: Scalar> Ray3<S> {
    /// Creates and returns a new `Ray3` with an origin point and direction vector.
    #[inline]
    pub fn new(origin: Point3<S>, direction: Vector3<S>) -> Ray3<S> {
        Ray3 { origin, direction }
    }

    /// Computes and returns a point at a distance along this ray.
    #[inline]
    pub fn at(&self, distance: S) -> Point3<S> {
        self.origin + self.direction * distance
    }
}

impl<S: Scalar> Transform<&Ray3<S>> for Transform3<S> {
    type Output = Ray3<S>;

    /// Transforms a ray.
    #[inline]
    fn transform(&self, ray: &Ray3<S>) -> Ray3<S> {
        Ray3::new(self.transform(ray.origin), self.transform(ray.direction))
    }
}

// ===== BoundingBox3 ==========================================================================================================================================

impl<S: Scalar> BoundingBox3<S> {
    /// Creates and returns a new `BoundingBox3` with minimum and maximum corner points.
    #[inline]
    pub fn new(p1: Point3<S>, p2: Point3<S>) -> BoundingBox3<S> {
        let min = min(p1, p2);
        let max = max(p1, p2);
        BoundingBox3 { min, max }
    }

    /// Returns an empty `BoundingBox3`.
    #[inline]
    pub fn empty() -> BoundingBox3<S> {
        BoundingBox3 {
            min: Point3::new(S::infinity(), S::infinity(), S::infinity()),
            max: Point3::new(S::neg_infinity(), S::neg_infinity(), S::neg_infinity()),
        }
    }

    /// Returns an infinite `BoundingBox3` which contains all of 3D space.
    #[inline]
    pub fn infinite() -> BoundingBox3<S> {
        BoundingBox3 {
            min: Point3::new(S::neg_infinity(), S::neg_infinity(), S::neg_infinity()),
            max: Point3::new(S::infinity(), S::infinity(), S::infinity()),
        }
    }

    /// Returns the width (extent in the X dimension) of this bounding box.
    #[inline]
    pub fn width(&self) -> S {
        self.max.x - self.min.x
    }

    /// Returns the height (extent in the Y dimension) of this bounding box.
    #[inline]
    pub fn height(&self) -> S {
        self.max.y - self.min.y
    }

    /// Returns the depth (extent in the Z dimension) of this bounding box.
    #[inline]
    pub fn depth(&self) -> S {
        self.max.z - self.min.z
    }

    /// Returns the extent of this bounding box in a dimension.
    #[inline]
    pub fn extent(&self, dim: Dimension3) -> S {
        match dim {
            Dimension3::X => self.width(),
            Dimension3::Y => self.height(),
            Dimension3::Z => self.depth(),
        }
    }

    /// Returns the dimension with the smallest extent of this bounding box.
    #[inline]
    pub fn min_dimension(&self) -> Dimension3 {
        let d = self.diagonal();
        if d.x <= d.y && d.x <= d.z { Dimension3::X } else if d.y <= d.z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this bounding box.
    #[inline]
    pub fn max_dimension(&self) -> Dimension3 {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z { Dimension3::X } else if d.y > d.z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the surface area of this bounding box.
    #[inline]
    pub fn surface_area(&self) -> S {
        let d = self.diagonal();
        S::two() * (d.x * d.y + d.x * d.z + d.y * d.z)
    }

    /// Returns the volume (width times height times depth) of this bounding box.
    #[inline]
    pub fn volume(&self) -> S {
        let d = self.diagonal();
        d.x * d.y * d.z
    }

    /// Returns the center point of this bounding box.
    #[inline]
    pub fn center(&self) -> Point3<S> {
        self.min + self.diagonal() * S::half()
    }

    /// Returns a corner point of this bounding box, indicated by an index (which must be between 0 and 7 inclusive).
    #[inline]
    pub fn corner(&self, index: usize) -> Point3<S> {
        debug_assert!(index < 8, "Invalid corner index: {}", index);
        let x = if index & 0b001 == 0 { self.min.x } else { self.max.x };
        let y = if index & 0b010 == 0 { self.min.y } else { self.max.y };
        let z = if index & 0b100 == 0 { self.min.z } else { self.max.z };
        Point3::new(x, y, z)
    }

    /// Returns the diagonal of this bounding box as a vector.
    #[inline]
    pub fn diagonal(&self) -> Vector3<S> {
        self.max - self.min
    }

    /// Checks if two bounding boxes overlap.
    #[inline]
    pub fn overlaps(&self, bb: &BoundingBox3<S>) -> bool {
        //@formatter:off
        self.max.x >= bb.min.x && self.min.x <= bb.max.x &&
        self.max.y >= bb.min.y && self.min.y <= bb.max.y &&
        self.max.z >= bb.min.z && self.min.z <= bb.max.z
        //@formatter:on
    }

    /// Checks if a point is inside this bounding box.
    #[inline]
    pub fn is_inside(&self, p: Point3<S>) -> bool {
        //@formatter:off
        p.x >= self.min.x && p.x <= self.max.x &&
        p.y >= self.min.y && p.y <= self.max.y &&
        p.z >= self.min.z && p.z <= self.max.z
        //@formatter:on
    }

    /// Computes the closest intersection of this bounding box with a ray within a range.
    ///
    /// Returns a `Some` containing the closest intersection, or `None` if the ray does not intersect the bounding box within the range.
    pub fn intersect_ray(&self, ray: &Ray3<S>, range: &Range<S>) -> Option<S> {
        let (start, end) = (range.start, range.end);

        let d1 = (self.min.x - ray.origin.x) / ray.direction.x;
        let d2 = (self.max.x - ray.origin.x) / ray.direction.x;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start > end {
            return None;
        }

        let d1 = (self.min.y - ray.origin.y) / ray.direction.y;
        let d2 = (self.max.y - ray.origin.y) / ray.direction.y;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start > end {
            return None;
        }

        let d1 = (self.min.z - ray.origin.z) / ray.direction.z;
        let d2 = (self.max.z - ray.origin.z) / ray.direction.z;

        let start = max(start, min(d1, d2));
        let end = min(end, max(d1, d2));

        if start <= end {
            Some(start)
        } else {
            None
        }
    }
}

impl<S: Scalar> Union<&BoundingBox3<S>> for BoundingBox3<S> {
    type Output = BoundingBox3<S>;

    /// Computes and returns the union between two bounding boxes.
    ///
    /// The union is the smallest bounding box that contains both bounding boxes.
    #[inline]
    fn union(self, bb: &BoundingBox3<S>) -> BoundingBox3<S> {
        BoundingBox3::new(min(self.min, bb.min), max(self.max, bb.max))
    }
}

impl<S: Scalar> Union<Point3<S>> for BoundingBox3<S> {
    type Output = BoundingBox3<S>;

    /// Computes and returns the union between this bounding box and a point.
    ///
    /// The union is the smallest bounding box that contains both the bounding box and the point.
    #[inline]
    fn union(self, p: Point3<S>) -> BoundingBox3<S> {
        BoundingBox3::new(min(self.min, p), max(self.max, p))
    }
}

impl<S: Scalar> Intersection<&BoundingBox3<S>> for BoundingBox3<S> {
    type Output = BoundingBox3<S>;

    /// Computes and returns the intersection between two bounding boxes.
    ///
    /// The intersection is the largest bounding box that contains the region where the two bounding boxes overlap.
    ///
    /// Returns `Some` when the bounding boxes overlap; `None` if the bounding boxes do not overlap.
    #[inline]
    fn intersection(self, bb: &BoundingBox3<S>) -> Option<BoundingBox3<S>> {
        if self.overlaps(bb) {
            Some(BoundingBox3::new(max(self.min, bb.min), min(self.max, bb.max)))
        } else {
            None
        }
    }
}

impl<S: Scalar> Transform<&BoundingBox3<S>> for Transform3<S> {
    type Output = BoundingBox3<S>;

    /// Transforms a bounding box.
    fn transform(&self, bb: &BoundingBox3<S>) -> BoundingBox3<S> {
        let o = self.transform(bb.min);
        let d = self.transform(bb.diagonal());

        let (mut min_corner, mut max_corner) = (o, o);
        for i in 1..8 {
            let mut corner = o;
            if i & 0b001 != 0 { corner.x += d.x; }
            if i & 0b010 != 0 { corner.y += d.y; }
            if i & 0b100 != 0 { corner.z += d.z; }

            min_corner = min(min_corner, corner);
            max_corner = max(max_corner, corner);
        }

        BoundingBox3::new(min_corner, max_corner)
    }
}

// ===== Matrix4x4 =============================================================================================================================================

impl<S: Scalar> Matrix4x4<S> {
    /// Creates and returns a new `Matrix4x4` with the specified elements.
    #[inline]
    pub fn new(m: [S; 16]) -> Matrix4x4<S> {
        Matrix4x4 { m }
    }

    /// Returns a `Matrix4x4` which represents the identity matrix.
    #[inline]
    pub fn identity() -> Matrix4x4<S> {
        let (o, i) = (S::zero(), S::one());

        Matrix4x4::new([
            i, o, o, o,
            o, i, o, o,
            o, o, i, o,
            o, o, o, i,
        ])
    }

    /// Returns a translation matrix which translates over a vector.
    #[inline]
    pub fn translate(v: Vector3<S>) -> Matrix4x4<S> {
        let (o, i) = (S::zero(), S::one());

        Matrix4x4::new([
            i, o, o, v.x,
            o, i, o, v.y,
            o, o, i, v.z,
            o, o, o, i,
        ])
    }

    /// Returns a rotation matrix which rotates around the X axis.
    #[inline]
    pub fn rotate_x(angle: S) -> Matrix4x4<S> {
        let (o, i) = (S::zero(), S::one());
        let (sin, cos) = angle.sin_cos();

        Matrix4x4::new([
            i, o, o, o,
            o, cos, -sin, o,
            o, sin, cos, o,
            o, o, o, i,
        ])
    }

    /// Returns a rotation matrix which rotates around the Y axis.
    #[inline]
    pub fn rotate_y(angle: S) -> Matrix4x4<S> {
        let (o, i) = (S::zero(), S::one());
        let (sin, cos) = angle.sin_cos();

        Matrix4x4::new([
            cos, o, sin, o,
            o, i, o, o,
            -sin, o, cos, o,
            o, o, o, i,
        ])
    }

    /// Returns a rotation matrix which rotates around the Y axis.
    #[inline]
    pub fn rotate_z(angle: S) -> Matrix4x4<S> {
        let (o, i) = (S::zero(), S::one());
        let (sin, cos) = angle.sin_cos();

        Matrix4x4::new([
            cos, -sin, o, o,
            sin, cos, o, o,
            o, o, i, o,
            o, o, o, i,
        ])
    }

    /// Returns a rotation matrix which rotates around an axis.
    #[inline]
    pub fn rotate_axis(axis: Vector3<S>, angle: S) -> Matrix4x4<S> {
        let a = axis.normalize();

        let (o, i) = (S::zero(), S::one());
        let (s, c) = angle.sin_cos();
        let cc = S::one() - c;

        let (t1, t2, t3) = (a.x * a.y * cc, a.x * a.z * cc, a.y * a.z * cc);
        let (u1, u2, u3) = (a.x * s, a.y * s, a.z * s);

        Matrix4x4::new([
            a.x * a.x * cc + c, t1 - u3, t2 + u2, o,
            t1 + u3, a.y * a.y * cc + c, t3 - u1, o,
            t2 - u2, t3 + u1, a.z * a.z * cc + c, o,
            o, o, o, i,
        ])
    }

    /// Returns a matrix which scales by factors in the X, Y and Z dimensions.
    #[inline]
    pub fn scale(sx: S, sy: S, sz: S) -> Matrix4x4<S> {
        let (o, i) = (S::zero(), S::one());

        Matrix4x4::new([
            sx, o, o, o,
            o, sy, o, o,
            o, o, sz, o,
            o, o, o, i,
        ])
    }

    /// Returns a matrix which scales uniformly in all dimensions by a factor.
    #[inline]
    pub fn scale_uniform(s: S) -> Matrix4x4<S> {
        let (o, i) = (S::zero(), S::one());

        Matrix4x4::new([
            s, o, o, o,
            o, s, o, o,
            o, o, s, o,
            o, o, o, i,
        ])
    }

    /// Returns the inverse of a look-at transformation matrix which looks from a point at a target, with an 'up' direction.
    #[inline]
    pub fn inverse_look_at(from: Point3<S>, target: Point3<S>, up: Vector3<S>) -> Matrix4x4<S> {
        let (o, i) = (S::zero(), S::one());
        let direction = (target - from).normalize();
        let right = up.normalize().cross(direction).normalize();
        let new_up = direction.cross(right);

        Matrix4x4::new([
            right.x, new_up.x, direction.x, from.x,
            right.y, new_up.y, direction.y, from.y,
            right.z, new_up.z, direction.z, from.z,
            o, o, o, i,
        ])
    }

    /// Returns an element at a row and column of the matrix.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> S {
        debug_assert!(row < 4, "Invalid row index: {}", row);
        debug_assert!(col < 4, "Invalid column index: {}", row);
        self.m[row * 4 + col]
    }

    /// Returns a mutable reference to an element at a row and column of the matrix.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut S {
        debug_assert!(row < 4, "Invalid row index: {}", row);
        debug_assert!(col < 4, "Invalid column index: {}", row);
        &mut self.m[row * 4 + col]
    }

    /// Sets the value of an element at a row and column of the matrix.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: S) {
        debug_assert!(row < 4, "Invalid row index: {}", row);
        debug_assert!(col < 4, "Invalid column index: {}", row);
        self.m[row * 4 + col] = value;
    }

    /// Returns the transpose of this matrix.
    #[inline]
    pub fn transpose(&self) -> Matrix4x4<S> {
        Matrix4x4::new([
            self.m[0], self.m[4], self.m[8], self.m[12],
            self.m[1], self.m[5], self.m[9], self.m[13],
            self.m[2], self.m[6], self.m[10], self.m[14],
            self.m[3], self.m[7], self.m[11], self.m[15],
        ])
    }

    /// Computes and returns the inverse of this matrix.
    ///
    /// If this matrix is singular, a `NonInvertibleMatrixError` is returned.
    pub fn inverse(&self) -> Result<Matrix4x4<S>, NonInvertibleMatrixError> {
        let cofactor = |i, j| {
            let sub = |row, col| self.get(if row < i { row } else { row + 1 }, if col < j { col } else { col + 1 });

            let sign = if (i + j) % 2 == 0 { S::one() } else { -S::one() };

            sign * (sub(0, 0) * sub(1, 1) * sub(2, 2) + sub(0, 1) * sub(1, 2) * sub(2, 0) + sub(0, 2) * sub(1, 0) * sub(2, 1)
                - sub(0, 0) * sub(1, 2) * sub(2, 1) - sub(0, 1) * sub(1, 0) * sub(2, 2) - sub(0, 2) * sub(1, 1) * sub(2, 0))
        };

        let adjugate = Matrix4x4::new([
            cofactor(0, 0), cofactor(1, 0), cofactor(2, 0), cofactor(3, 0),
            cofactor(0, 1), cofactor(1, 1), cofactor(2, 1), cofactor(3, 1),
            cofactor(0, 2), cofactor(1, 2), cofactor(2, 2), cofactor(3, 2),
            cofactor(0, 3), cofactor(1, 3), cofactor(2, 3), cofactor(3, 3),
        ]);

        let det = self.m[0] * adjugate.m[0] + self.m[1] * adjugate.m[4] + self.m[2] * adjugate.m[8] + self.m[3] * adjugate.m[12];

        if det != S::zero() {
            Ok(&adjugate * det.recip())
        } else {
            Err(NonInvertibleMatrixError)
        }
    }
}

impl<S: Scalar> Mul<S> for &Matrix4x4<S> {
    type Output = Matrix4x4<S>;

    #[inline]
    fn mul(self, s: S) -> Matrix4x4<S> {
        Matrix4x4::new(array::from_fn(|i| self.m[i] * s))
    }
}

impl Mul<&Matrix4x4f> for f32 {
    type Output = Matrix4x4f;

    #[inline]
    fn mul(self, m: &Matrix4x4f) -> Matrix4x4f {
        m * self
    }
}

impl Mul<&Matrix4x4d> for f64 {
    type Output = Matrix4x4d;

    #[inline]
    fn mul(self, m: &Matrix4x4d) -> Matrix4x4d {
        m * self
    }
}

impl<S: Scalar> MulAssign<S> for &mut Matrix4x4<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        for m in &mut self.m { *m *= s; }
    }
}

impl<S: Scalar> Div<S> for &Matrix4x4<S> {
    type Output = Matrix4x4<S>;

    #[inline]
    fn div(self, s: S) -> Matrix4x4<S> {
        Matrix4x4::new(array::from_fn(|i| self.m[i] / s))
    }
}

impl<S: Scalar> DivAssign<S> for &mut Matrix4x4<S> {
    #[inline]
    fn div_assign(&mut self, s: S) {
        for m in &mut self.m { *m /= s; }
    }
}

impl<S: Scalar> Mul<Point3<S>> for &Matrix4x4<S> {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, p: Point3<S>) -> Point3<S> {
        let x = self.m[0] * p.x + self.m[1] * p.y + self.m[2] * p.z + self.m[3];
        let y = self.m[4] * p.x + self.m[5] * p.y + self.m[6] * p.z + self.m[7];
        let z = self.m[8] * p.x + self.m[9] * p.y + self.m[10] * p.z + self.m[11];
        let w = self.m[12] * p.x + self.m[13] * p.y + self.m[14] * p.z + self.m[15];
        Point3::new(x / w, y / w, z / w)
    }
}

impl<S: Scalar> Mul<&Matrix4x4<S>> for Point3<S> {
    type Output = Point3<S>;

    #[inline]
    fn mul(self, m: &Matrix4x4<S>) -> Point3<S> {
        let x = self.x * m.m[0] + self.y * m.m[4] + self.z * m.m[8] + m.m[12];
        let y = self.x * m.m[1] + self.y * m.m[5] + self.z * m.m[9] + m.m[13];
        let z = self.x * m.m[2] + self.y * m.m[6] + self.z * m.m[10] + m.m[14];
        let w = self.x * m.m[3] + self.y * m.m[7] + self.z * m.m[11] + m.m[15];
        Point3::new(x / w, y / w, z / w)
    }
}

impl<S: Scalar> Mul<Vector3<S>> for &Matrix4x4<S> {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, v: Vector3<S>) -> Vector3<S> {
        let x = self.m[0] * v.x + self.m[1] * v.y + self.m[2] * v.z;
        let y = self.m[4] * v.x + self.m[5] * v.y + self.m[6] * v.z;
        let z = self.m[8] * v.x + self.m[9] * v.y + self.m[10] * v.z;
        Vector3::new(x, y, z)
    }
}

impl<S: Scalar> Mul<&Matrix4x4<S>> for Vector3<S> {
    type Output = Vector3<S>;

    #[inline]
    fn mul(self, m: &Matrix4x4<S>) -> Vector3<S> {
        let x = self.x * m.m[0] + self.y * m.m[4] + self.z * m.m[8];
        let y = self.x * m.m[1] + self.y * m.m[5] + self.z * m.m[9];
        let z = self.x * m.m[2] + self.y * m.m[6] + self.z * m.m[10];
        Vector3::new(x, y, z)
    }
}

impl<S: Scalar> Mul<&Matrix4x4<S>> for &Matrix4x4<S> {
    type Output = Matrix4x4<S>;

    #[inline]
    fn mul(self, m: &Matrix4x4<S>) -> Matrix4x4<S> {
        Matrix4x4::new([
            self.m[0] * m.m[0] + self.m[1] * m.m[4] + self.m[2] * m.m[8] + self.m[3] * m.m[12],
            self.m[0] * m.m[1] + self.m[1] * m.m[5] + self.m[2] * m.m[9] + self.m[3] * m.m[13],
            self.m[0] * m.m[2] + self.m[1] * m.m[6] + self.m[2] * m.m[10] + self.m[3] * m.m[14],
            self.m[0] * m.m[3] + self.m[1] * m.m[7] + self.m[2] * m.m[11] + self.m[3] * m.m[15],
            self.m[4] * m.m[0] + self.m[5] * m.m[4] + self.m[6] * m.m[8] + self.m[7] * m.m[12],
            self.m[4] * m.m[1] + self.m[5] * m.m[5] + self.m[6] * m.m[9] + self.m[7] * m.m[13],
            self.m[4] * m.m[2] + self.m[5] * m.m[6] + self.m[6] * m.m[10] + self.m[7] * m.m[14],
            self.m[4] * m.m[3] + self.m[5] * m.m[7] + self.m[6] * m.m[11] + self.m[7] * m.m[15],
            self.m[8] * m.m[0] + self.m[9] * m.m[4] + self.m[10] * m.m[8] + self.m[11] * m.m[12],
            self.m[8] * m.m[1] + self.m[9] * m.m[5] + self.m[10] * m.m[9] + self.m[11] * m.m[13],
            self.m[8] * m.m[2] + self.m[9] * m.m[6] + self.m[10] * m.m[10] + self.m[11] * m.m[14],
            self.m[8] * m.m[3] + self.m[9] * m.m[7] + self.m[10] * m.m[11] + self.m[11] * m.m[15],
            self.m[12] * m.m[0] + self.m[13] * m.m[4] + self.m[14] * m.m[8] + self.m[15] * m.m[12],
            self.m[12] * m.m[1] + self.m[13] * m.m[5] + self.m[14] * m.m[9] + self.m[15] * m.m[13],
            self.m[12] * m.m[2] + self.m[13] * m.m[6] + self.m[14] * m.m[10] + self.m[15] * m.m[14],
            self.m[12] * m.m[3] + self.m[13] * m.m[7] + self.m[14] * m.m[11] + self.m[15] * m.m[15],
        ])
    }
}

// ===== Transform3 ============================================================================================================================================

impl<S: Scalar> Transform3<S> {
    /// Creates and returns a new `Transform3` with a transformation matrix and its inverse.
    #[inline]
    pub fn new(forward: Matrix4x4<S>, inverse: Matrix4x4<S>) -> Transform3<S> {
        Transform3 { forward, inverse }
    }

    /// Returns a `Transform3` which represents the identity transform.
    #[inline]
    pub fn identity() -> Transform3<S> {
        let forward = Matrix4x4::identity();
        let inverse = forward.clone();
        Transform3::new(forward, inverse)
    }

    /// Returns a translation transform over a vector.
    #[inline]
    pub fn translate(v: Vector3<S>) -> Transform3<S> {
        let forward = Matrix4x4::translate(v);
        let inverse = Matrix4x4::translate(-v);
        Transform3::new(forward, inverse)
    }

    /// Returns a rotation transform which rotates around the X axis.
    #[inline]
    pub fn rotate_x(angle: S) -> Transform3<S> {
        let forward = Matrix4x4::rotate_x(angle);
        let inverse = forward.transpose();
        Transform3::new(forward, inverse)
    }

    /// Returns a rotation transform which rotates around the Y axis.
    #[inline]
    pub fn rotate_y(angle: S) -> Transform3<S> {
        let forward = Matrix4x4::rotate_y(angle);
        let inverse = forward.transpose();
        Transform3::new(forward, inverse)
    }

    /// Returns a rotation transform which rotates around the Z axis.
    #[inline]
    pub fn rotate_z(angle: S) -> Transform3<S> {
        let forward = Matrix4x4::rotate_z(angle);
        let inverse = forward.transpose();
        Transform3::new(forward, inverse)
    }

    /// Returns a rotation transform which rotates around an axis.
    #[inline]
    pub fn rotate_axis(axis: Vector3<S>, angle: S) -> Transform3<S> {
        let forward = Matrix4x4::rotate_axis(axis, angle);
        let inverse = forward.transpose();
        Transform3::new(forward, inverse)
    }

    /// Returns a transform which scales by factors in the X, Y and Z dimensions.
    #[inline]
    pub fn scale(sx: S, sy: S, sz: S) -> Transform3<S> {
        let forward = Matrix4x4::scale(sx, sy, sz);
        let inverse = Matrix4x4::scale(sx.recip(), sy.recip(), sz.recip());
        Transform3::new(forward, inverse)
    }

    /// Returns a transform which scales uniformly in all dimensions by a factor.
    #[inline]
    pub fn scale_uniform(s: S) -> Transform3<S> {
        let forward = Matrix4x4::scale_uniform(s);
        let inverse = Matrix4x4::scale_uniform(s.recip());
        Transform3::new(forward, inverse)
    }

    // TODO: factory methods for perspective and orthographic projection matrices and transforms

    /// Returns a look-at transform which looks from a point at a target, with an 'up' direction.
    #[inline]
    pub fn look_at(from: Point3<S>, target: Point3<S>, up: Vector3<S>) -> Result<Transform3<S>, NonInvertibleMatrixError> {
        let inverse = Matrix4x4::inverse_look_at(from, target, up);
        let forward = inverse.inverse()?;
        Ok(Transform3::new(forward, inverse))
    }

    /// Computes and returns a composite transform, which first applies this and then the other transform.
    #[inline]
    pub fn and_then(&self, transform: &Transform3<S>) -> Transform3<S> {
        let forward = &transform.forward * &self.forward;
        let inverse = &self.inverse * &transform.inverse;
        Transform3::new(forward, inverse)
    }

    /// Returns the inverse of this transform.
    #[inline]
    pub fn inverse(&self) -> Transform3<S> {
        let forward = self.inverse.clone();
        let inverse = self.forward.clone();
        Transform3::new(forward, inverse)
    }
}

impl<S: Scalar> TryFrom<Matrix4x4<S>> for Transform3<S> {
    type Error = NonInvertibleMatrixError;

    #[inline]
    fn try_from(forward: Matrix4x4<S>) -> Result<Transform3<S>, NonInvertibleMatrixError> {
        let inverse = forward.inverse()?;
        Ok(Transform3::new(forward, inverse))
    }
}

// ===== Tests =================================================================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_min() {
        assert_eq!(min(-1.0, 2.0), -1.0);
    }

    #[test]
    fn scalar_max() {
        assert_eq!(max(-2.0, 1.0), 1.0);
    }

    #[test]
    fn point2_new() {
        let p = Point2d::new(-1.0, 2.0);
        assert_eq!(p.x, -1.0);
        assert_eq!(p.y, 2.0);
    }

    #[test]
    fn point2_origin() {
        let p = Point2d::origin();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn point2_min_dimension() {
        assert_eq!(Point2d::new(-1.0, 2.0).min_dimension(), Dimension2::X);
        assert_eq!(Point2d::new(-3.0, 2.0).min_dimension(), Dimension2::Y);
    }

    #[test]
    fn point2_max_dimension() {
        assert_eq!(Point2d::new(-1.0, 2.0).max_dimension(), Dimension2::Y);
        assert_eq!(Point2d::new(-3.0, 2.0).max_dimension(), Dimension2::X);
    }

    #[test]
    fn point2_floor() {
        assert_eq!(Point2d::new(-1.3, 2.6).floor(), Point2d::new(-2.0, 2.0));
    }

    #[test]
    fn point2_ceil() {
        assert_eq!(Point2d::new(-1.3, 2.6).ceil(), Point2d::new(-1.0, 3.0));
    }

    #[test]
    fn point2_round() {
        assert_eq!(Point2d::new(-1.6, 2.3).round(), Point2d::new(-2.0, 2.0));
    }

    #[test]
    fn point2_trunc() {
        assert_eq!(Point2d::new(-1.3, 2.6).trunc(), Point2d::new(-1.0, 2.0));
    }

    #[test]
    fn point2_fract() {
        assert_eq!(Point2d::new(-1.25, 2.5).fract(), Point2d::new(-0.25, 0.5));
    }

    #[test]
    fn point2_abs() {
        assert_eq!(Point2d::new(-1.3, 2.6).abs(), Point2d::new(1.3, 2.6));
        assert_eq!(Point2d::new(1.3, -2.6).abs(), Point2d::new(1.3, 2.6));
    }

    #[test]
    fn point2_permute() {
        assert_eq!(Point2d::new(1.0, 2.0).permute(Dimension2::X, Dimension2::X), Point2d::new(1.0, 1.0));
        assert_eq!(Point2d::new(1.0, 2.0).permute(Dimension2::X, Dimension2::Y), Point2d::new(1.0, 2.0));
        assert_eq!(Point2d::new(1.0, 2.0).permute(Dimension2::Y, Dimension2::X), Point2d::new(2.0, 1.0));
        assert_eq!(Point2d::new(1.0, 2.0).permute(Dimension2::Y, Dimension2::Y), Point2d::new(2.0, 2.0));
    }

    #[test]
    fn point2_min() {
        assert_eq!(min(Point2d::new(-1.0, 2.0), Point2d::new(-3.0, 2.5)), Point2d::new(-3.0, 2.0));
    }

    #[test]
    fn point2_max() {
        assert_eq!(max(Point2d::new(-1.0, 2.0), Point2d::new(-3.0, 2.5)), Point2d::new(-1.0, 2.5));
    }

    #[test]
    fn point2_distance() {
        assert_eq!(distance(Point2d::new(4.0, 1.0), Point2d::new(1.0, 5.0)), 5.0);
    }

    #[test]
    fn point2_closest() {
        let p1 = Point2d::new(4.0, 1.0);
        let p2 = Point2d::new(1.0, 5.0);
        assert_eq!(Point2d::new(-1.0, 2.0).closest(p1, p2), p2);
    }

    #[test]
    fn point2_farthest() {
        let p1 = Point2d::new(4.0, 1.0);
        let p2 = Point2d::new(1.0, 5.0);
        assert_eq!(Point2d::new(-1.0, 2.0).farthest(p1, p2), p1);
    }

    #[test]
    fn point2_index() {
        let p = Point2d::new(1.0, 2.0);
        assert_eq!(p[Dimension2::X], 1.0);
        assert_eq!(p[Dimension2::Y], 2.0);
    }

    #[test]
    fn point2_index_mut() {
        let mut p = Point2d::new(1.0, 2.0);
        p[Dimension2::X] = 3.0;
        p[Dimension2::Y] = -1.0;
        assert_eq!(p, Point2d::new(3.0, -1.0));
    }

    #[test]
    fn point2_add_vector2() {
        let p = Point2d::new(1.0, 2.0);
        let v = Vector2d::new(-0.5, 1.5);
        assert_eq!(p + v, Point2d::new(0.5, 3.5));
    }

    #[test]
    fn point2_add_assign_vector2() {
        let mut p = Point2d::new(1.0, 2.0);
        let v = Vector2d::new(-0.5, 1.5);
        p += v;
        assert_eq!(p, Point2d::new(0.5, 3.5));
    }

    #[test]
    fn point2_sub_vector2() {
        let p = Point2d::new(1.0, 2.0);
        let v = Vector2d::new(-0.5, 1.5);
        assert_eq!(p - v, Point2d::new(1.5, 0.5));
    }

    #[test]
    fn point2_sub_assign_vector2() {
        let mut p = Point2d::new(1.0, 2.0);
        let v = Vector2d::new(-0.5, 1.5);
        p -= v;
        assert_eq!(p, Point2d::new(1.5, 0.5));
    }

    #[test]
    fn point2_sub_point2() {
        let p1 = Point2d::new(4.0, 2.0);
        let p2 = Point2d::new(1.0, 5.0);
        assert_eq!(p1 - p2, Vector2d::new(3.0, -3.0));
    }

    #[test]
    fn point2_neg() {
        assert_eq!(-Point2d::new(1.0, -2.0), Point2d::new(-1.0, 2.0));
    }

    #[test]
    fn point2_mul_scalar() {
        assert_eq!(Point2d::new(2.5, -1.5) * 2.0, Point2d::new(5.0, -3.0));
    }

    #[test]
    fn scalar_mul_point2() {
        assert_eq!(2.0 * Point2d::new(2.5, -1.5), Point2d::new(5.0, -3.0));
    }

    #[test]
    fn point2_mul_assign_scalar() {
        let mut p = Point2d::new(2.5, -1.5);
        p *= 2.0;
        assert_eq!(p, Point2d::new(5.0, -3.0));
    }

    #[test]
    fn point2_div_scalar() {
        assert_eq!(Point2d::new(2.5, -1.5) / 2.0, Point2d::new(1.25, -0.75));
    }

    #[test]
    fn point2_div_assign_scalar() {
        let mut p = Point2d::new(2.5, -1.5);
        p /= 2.0;
        assert_eq!(p, Point2d::new(1.25, -0.75));
    }

    #[test]
    fn point2_from_vector2() {
        let p = Point2d::from(Vector2d::new(1.0, 2.0));
        assert_eq!(p, Point2d::new(1.0, 2.0));
    }

    #[test]
    fn vector2_new() {
        let v = Vector2d::new(-1.0, 2.0);
        assert_eq!(v.x, -1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn vector2_zero() {
        let v = Vector2d::zero();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
    }

    #[test]
    fn vector2_x_axis() {
        let v = Vector2d::x_axis();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 0.0);
    }

    #[test]
    fn vector2_y_axis() {
        let v = Vector2d::y_axis();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 1.0);
    }

    #[test]
    fn vector2_axis() {
        assert_eq!(Vector2d::axis(Dimension2::X), Vector2d::x_axis());
        assert_eq!(Vector2d::axis(Dimension2::Y), Vector2d::y_axis());
    }

    #[test]
    fn vector2_normalize() {
        let v = Vector2d::new(3.0, -2.0);
        assert_eq!(v.normalize(), v / f64::sqrt(13.0));
    }

    #[test]
    fn vector2_min_dimension() {
        assert_eq!(Vector2d::new(-1.0, 2.0).min_dimension(), Dimension2::X);
        assert_eq!(Vector2d::new(-3.0, 2.0).min_dimension(), Dimension2::Y);
    }

    #[test]
    fn vector2_max_dimension() {
        assert_eq!(Vector2d::new(-1.0, 2.0).max_dimension(), Dimension2::Y);
        assert_eq!(Vector2d::new(-3.0, 2.0).max_dimension(), Dimension2::X);
    }

    #[test]
    fn vector2_floor() {
        assert_eq!(Vector2d::new(-1.3, 2.6).floor(), Vector2d::new(-2.0, 2.0));
    }

    #[test]
    fn vector2_ceil() {
        assert_eq!(Vector2d::new(-1.3, 2.6).ceil(), Vector2d::new(-1.0, 3.0));
    }

    #[test]
    fn vector2_round() {
        assert_eq!(Vector2d::new(-1.6, 2.3).round(), Vector2d::new(-2.0, 2.0));
    }

    #[test]
    fn vector2_trunc() {
        assert_eq!(Vector2d::new(-1.3, 2.6).trunc(), Vector2d::new(-1.0, 2.0));
    }

    #[test]
    fn vector2_fract() {
        assert_eq!(Vector2d::new(-1.25, 2.5).fract(), Vector2d::new(-0.25, 0.5));
    }

    #[test]
    fn vector2_abs() {
        assert_eq!(Vector2d::new(-1.3, 2.6).abs(), Vector2d::new(1.3, 2.6));
    }

    #[test]
    fn vector2_permute() {
        assert_eq!(Vector2d::new(1.0, 2.0).permute(Dimension2::X, Dimension2::X), Vector2d::new(1.0, 1.0));
        assert_eq!(Vector2d::new(1.0, 2.0).permute(Dimension2::X, Dimension2::Y), Vector2d::new(1.0, 2.0));
        assert_eq!(Vector2d::new(1.0, 2.0).permute(Dimension2::Y, Dimension2::X), Vector2d::new(2.0, 1.0));
        assert_eq!(Vector2d::new(1.0, 2.0).permute(Dimension2::Y, Dimension2::Y), Vector2d::new(2.0, 2.0));
    }

    #[test]
    fn vector2_min() {
        assert_eq!(min(Vector2d::new(-1.0, 2.0), Vector2d::new(-3.0, 2.5)), Vector2d::new(-3.0, 2.0));
    }

    #[test]
    fn vector2_max() {
        assert_eq!(max(Vector2d::new(-1.0, 2.0), Vector2d::new(-3.0, 2.5)), Vector2d::new(-1.0, 2.5));
    }

    #[test]
    fn vector2_length() {
        assert_eq!(Vector2d::new(3.0, 4.0).length(), 5.0);
    }

    #[test]
    fn vector2_shortest() {
        let v1 = Vector2d::new(-1.0, -3.0);
        let v2 = Vector2d::new(2.0, 1.5);
        assert_eq!(shortest(v1, v2), v2);
    }

    #[test]
    fn vector2_longest() {
        let v1 = Vector2d::new(-1.0, -3.0);
        let v2 = Vector2d::new(2.0, 1.5);
        assert_eq!(longest(v1, v2), v1);
    }

    #[test]
    fn vector2_dot_vector2() {
        let v1 = Vector2d::new(-1.0, -3.0);
        let v2 = Vector2d::new(2.0, 1.5);
        assert_eq!(dot(v1, v2), -6.5);
    }

    #[test]
    fn vector2_index() {
        let v = Vector2d::new(1.0, 2.0);
        assert_eq!(v[Dimension2::X], 1.0);
        assert_eq!(v[Dimension2::Y], 2.0);
    }

    #[test]
    fn vector2_index_mut() {
        let mut v = Vector2d::new(1.0, 2.0);
        v[Dimension2::X] = 3.0;
        v[Dimension2::Y] = -1.0;
        assert_eq!(v, Vector2d::new(3.0, -1.0));
    }

    #[test]
    fn vector2_add_vector2() {
        let v1 = Vector2d::new(1.0, 2.0);
        let v2 = Vector2d::new(-0.5, 1.5);
        assert_eq!(v1 + v2, Vector2d::new(0.5, 3.5));
    }

    #[test]
    fn vector2_add_assign_vector2() {
        let mut v1 = Vector2d::new(1.0, 2.0);
        let v2 = Vector2d::new(-0.5, 1.5);
        v1 += v2;
        assert_eq!(v1, Vector2d::new(0.5, 3.5));
    }

    #[test]
    fn vector2_sub_vector2() {
        let v1 = Vector2d::new(1.0, 2.0);
        let v2 = Vector2d::new(-0.5, 1.5);
        assert_eq!(v1 - v2, Vector2d::new(1.5, 0.5));
    }

    #[test]
    fn vector2_sub_assign_vector2() {
        let mut v1 = Vector2d::new(1.0, 2.0);
        let v2 = Vector2d::new(-0.5, 1.5);
        v1 -= v2;
        assert_eq!(v1, Vector2d::new(1.5, 0.5));
    }

    #[test]
    fn vector2_neg() {
        assert_eq!(-Vector2d::new(1.0, -2.0), Vector2d::new(-1.0, 2.0));
    }

    #[test]
    fn vector2_mul_scalar() {
        assert_eq!(Vector2d::new(2.5, -1.5) * 2.0, Vector2d::new(5.0, -3.0));
    }

    #[test]
    fn scalar_mul_vector2() {
        assert_eq!(2.0 * Vector2d::new(2.5, -1.5), Vector2d::new(5.0, -3.0));
    }

    #[test]
    fn vector2_mul_assign_scalar() {
        let mut v = Vector2d::new(2.5, -1.5);
        v *= 2.0;
        assert_eq!(v, Vector2d::new(5.0, -3.0));
    }

    #[test]
    fn vector2_div_scalar() {
        assert_eq!(Vector2d::new(2.5, -1.5) / 2.0, Vector2d::new(1.25, -0.75));
    }

    #[test]
    fn vector2_div_assign_scalar() {
        let mut v = Vector2d::new(2.5, -1.5);
        v /= 2.0;
        assert_eq!(v, Vector2d::new(1.25, -0.75));
    }

    #[test]
    fn vector2_from_point2() {
        let v = Vector2d::from(Point2d::new(1.0, 2.0));
        assert_eq!(v, Vector2d::new(1.0, 2.0));
    }

    #[test]
    fn ray2_new() {
        let r = Ray2d::new(Point2d::new(1.0, 2.0), Vector2d::new(-1.5, 0.5));
        assert_eq!(r.origin, Point2d::new(1.0, 2.0));
        assert_eq!(r.direction, Vector2d::new(-1.5, 0.5));
    }

    #[test]
    fn ray2_at() {
        let r = Ray2d::new(Point2d::new(1.0, 2.0), Vector2d::new(-1.5, 0.5));
        assert_eq!(r.at(2.5), Point2d::new(-2.75, 3.25));
    }

    #[test]
    fn bounding_box2_new() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert_eq!(bb.min, Point2d::new(1.0, -0.5));
        assert_eq!(bb.max, Point2d::new(5.0, 4.0));
    }

    #[test]
    fn bounding_box2_empty() {
        let bb = BoundingBox2d::empty();
        assert_eq!(bb.min, Point2d::new(f64::INFINITY, f64::INFINITY));
        assert_eq!(bb.max, Point2d::new(f64::NEG_INFINITY, f64::NEG_INFINITY));
    }

    #[test]
    fn bounding_box2_infinite() {
        let bb = BoundingBox2d::infinite();
        assert_eq!(bb.min, Point2d::new(f64::NEG_INFINITY, f64::NEG_INFINITY));
        assert_eq!(bb.max, Point2d::new(f64::INFINITY, f64::INFINITY));
    }

    #[test]
    fn bounding_box2_width() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert_eq!(bb.width(), 4.0);
    }

    #[test]
    fn bounding_box2_height() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert_eq!(bb.height(), 4.5);
    }

    #[test]
    fn bounding_box2_extent() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert_eq!(bb.extent(Dimension2::X), 4.0);
        assert_eq!(bb.extent(Dimension2::Y), 4.5);
    }

    #[test]
    fn bounding_box2_min_dimension() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert_eq!(bb.min_dimension(), Dimension2::X);
    }

    #[test]
    fn bounding_box2_max_dimension() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert_eq!(bb.max_dimension(), Dimension2::Y);
    }

    #[test]
    fn bounding_box2_area() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert_eq!(bb.area(), 18.0);
    }

    #[test]
    fn bounding_box2_center() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert_eq!(bb.center(), Point2d::new(3.0, 1.75));
    }

    #[test]
    fn bounding_box2_corner() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert_eq!(bb.corner(0), Point2d::new(1.0, -0.5));
        assert_eq!(bb.corner(1), Point2d::new(5.0, -0.5));
        assert_eq!(bb.corner(2), Point2d::new(1.0, 4.0));
        assert_eq!(bb.corner(3), Point2d::new(5.0, 4.0));
    }

    #[test]
    fn bounding_box2_diagonal() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert_eq!(bb.diagonal(), Vector2d::new(4.0, 4.5));
    }

    #[test]
    fn bounding_box2_overlaps() {
        let bb1 = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        let bb2 = BoundingBox2d::new(Point2d::new(-1.0, 2.0), Point2d::new(3.0, 6.0));
        let bb3 = BoundingBox2d::new(Point2d::new(3.5, 1.5), Point2d::new(6.0, 5.0));
        assert!(bb1.overlaps(&bb2));
        assert!(!bb2.overlaps(&bb3));
    }

    #[test]
    fn bounding_box2_is_inside() {
        let bb = BoundingBox2d::new(Point2d::new(1.0, -0.5), Point2d::new(5.0, 4.0));
        assert!(!bb.is_inside(Point2d::new(0.0, 2.0)));
        assert!(bb.is_inside(Point2d::new(2.0, 3.0)));
        assert!(!bb.is_inside(Point2d::new(4.0, 5.0)));
    }

    #[test]
    fn bounding_box2_intersect_ray() {
        // TODO: Needs more elaborate tests, with rays going in different directions and different hit and miss cases
    }

    #[test]
    fn bounding_box2_union_bounding_box2() {
        // TODO: Test different cases with and without overlap
    }

    #[test]
    fn bounding_box2_union_point2() {
        // TODO: Test different cases with point inside and outside bounding box
    }

    #[test]
    fn bounding_box2_intersection_bounding_box2() {
        // TODO: Test different cases with and without overlap
    }

    #[test]
    fn matrix3x3_new() {
        let m = Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        //@formatter:off
        assert_eq!(m.get(0, 0), 1.0); assert_eq!(m.get(0, 1), 2.0); assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0); assert_eq!(m.get(1, 1), 5.0); assert_eq!(m.get(1, 2), 6.0);
        assert_eq!(m.get(2, 0), 7.0); assert_eq!(m.get(2, 1), 8.0); assert_eq!(m.get(2, 2), 9.0);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_identity() {
        let m = Matrix3x3d::identity();
        //@formatter:off
        assert_eq!(m.get(0, 0), 1.0); assert_eq!(m.get(0, 1), 0.0); assert_eq!(m.get(0, 2), 0.0);
        assert_eq!(m.get(1, 0), 0.0); assert_eq!(m.get(1, 1), 1.0); assert_eq!(m.get(1, 2), 0.0);
        assert_eq!(m.get(2, 0), 0.0); assert_eq!(m.get(2, 1), 0.0); assert_eq!(m.get(2, 2), 1.0);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_translate() {
        let m = Matrix3x3d::translate(Vector2d::new(-2.0, 3.0));
        //@formatter:off
        assert_eq!(m.get(0, 0), 1.0); assert_eq!(m.get(0, 1), 0.0); assert_eq!(m.get(0, 2), -2.0);
        assert_eq!(m.get(1, 0), 0.0); assert_eq!(m.get(1, 1), 1.0); assert_eq!(m.get(1, 2), 3.0);
        assert_eq!(m.get(2, 0), 0.0); assert_eq!(m.get(2, 1), 0.0); assert_eq!(m.get(2, 2), 1.0);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_rotate() {
        let angle = std::f64::consts::FRAC_PI_6;
        let m = Matrix3x3d::rotate(angle);
        //@formatter:off
        assert_eq!(m.get(0, 0), angle.cos()); assert_eq!(m.get(0, 1), -angle.sin()); assert_eq!(m.get(0, 2), 0.0);
        assert_eq!(m.get(1, 0), angle.sin()); assert_eq!(m.get(1, 1), angle.cos()); assert_eq!(m.get(1, 2), 0.0);
        assert_eq!(m.get(2, 0), 0.0); assert_eq!(m.get(2, 1), 0.0); assert_eq!(m.get(2, 2), 1.0);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_scale() {
        let m = Matrix3x3d::scale(-2.0, 2.0);
        //@formatter:off
        assert_eq!(m.get(0, 0), -2.0); assert_eq!(m.get(0, 1), 0.0); assert_eq!(m.get(0, 2), 0.0);
        assert_eq!(m.get(1, 0), 0.0); assert_eq!(m.get(1, 1), 2.0); assert_eq!(m.get(1, 2), 0.0);
        assert_eq!(m.get(2, 0), 0.0); assert_eq!(m.get(2, 1), 0.0); assert_eq!(m.get(2, 2), 1.0);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_scale_uniform() {
        let m = Matrix3x3d::scale_uniform(2.5);
        //@formatter:off
        assert_eq!(m.get(0, 0), 2.5); assert_eq!(m.get(0, 1), 0.0); assert_eq!(m.get(0, 2), 0.0);
        assert_eq!(m.get(1, 0), 0.0); assert_eq!(m.get(1, 1), 2.5); assert_eq!(m.get(1, 2), 0.0);
        assert_eq!(m.get(2, 0), 0.0); assert_eq!(m.get(2, 1), 0.0); assert_eq!(m.get(2, 2), 1.0);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_get_mut() {
        let mut m = Matrix3x3d::identity();
        *m.get_mut(0, 0) = 2.0;
        *m.get_mut(0, 1) = 3.0;
        *m.get_mut(1, 0) = -2.0;
        *m.get_mut(2, 2) = 4.0;
        //@formatter:off
        assert_eq!(m.get(0, 0), 2.0); assert_eq!(m.get(0, 1), 3.0); assert_eq!(m.get(0, 2), 0.0);
        assert_eq!(m.get(1, 0), -2.0); assert_eq!(m.get(1, 1), 1.0); assert_eq!(m.get(1, 2), 0.0);
        assert_eq!(m.get(2, 0), 0.0); assert_eq!(m.get(2, 1), 0.0); assert_eq!(m.get(2, 2), 4.0);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_set() {
        let mut m = Matrix3x3d::identity();
        m.set(0, 0, 2.0);
        m.set(0, 1, 3.0);
        m.set(1, 0, -2.0);
        m.set(2, 2, 4.0);
        //@formatter:off
        assert_eq!(m.get(0, 0), 2.0); assert_eq!(m.get(0, 1), 3.0); assert_eq!(m.get(0, 2), 0.0);
        assert_eq!(m.get(1, 0), -2.0); assert_eq!(m.get(1, 1), 1.0); assert_eq!(m.get(1, 2), 0.0);
        assert_eq!(m.get(2, 0), 0.0); assert_eq!(m.get(2, 1), 0.0); assert_eq!(m.get(2, 2), 4.0);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_transpose() {
        let m = Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).transpose();
        //@formatter:off
        assert_eq!(m.get(0, 0), 1.0); assert_eq!(m.get(0, 1), 4.0); assert_eq!(m.get(0, 2), 7.0);
        assert_eq!(m.get(1, 0), 2.0); assert_eq!(m.get(1, 1), 5.0); assert_eq!(m.get(1, 2), 8.0);
        assert_eq!(m.get(2, 0), 3.0); assert_eq!(m.get(2, 1), 6.0); assert_eq!(m.get(2, 2), 9.0);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_inverse() {
        // TODO: Implement test
    }

    #[test]
    fn matrix3x3_mul_scalar() {
        let m = &Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]) * 2.5;
        //@formatter:off
        assert_eq!(m.get(0, 0), 2.5); assert_eq!(m.get(0, 1), 5.0); assert_eq!(m.get(0, 2), 7.5);
        assert_eq!(m.get(1, 0), 10.0); assert_eq!(m.get(1, 1), 12.5); assert_eq!(m.get(1, 2), 15.0);
        assert_eq!(m.get(2, 0), 17.5); assert_eq!(m.get(2, 1), 20.0); assert_eq!(m.get(2, 2), 22.5);
        //@formatter:on
    }

    #[test]
    fn scalar_mul_matrix3x3() {
        let m = 2.5 * &Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        //@formatter:off
        assert_eq!(m.get(0, 0), 2.5); assert_eq!(m.get(0, 1), 5.0); assert_eq!(m.get(0, 2), 7.5);
        assert_eq!(m.get(1, 0), 10.0); assert_eq!(m.get(1, 1), 12.5); assert_eq!(m.get(1, 2), 15.0);
        assert_eq!(m.get(2, 0), 17.5); assert_eq!(m.get(2, 1), 20.0); assert_eq!(m.get(2, 2), 22.5);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_mul_assign_scalar() {
        let mut m = &mut Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        m *= 2.5;
        //@formatter:off
        assert_eq!(m.get(0, 0), 2.5); assert_eq!(m.get(0, 1), 5.0); assert_eq!(m.get(0, 2), 7.5);
        assert_eq!(m.get(1, 0), 10.0); assert_eq!(m.get(1, 1), 12.5); assert_eq!(m.get(1, 2), 15.0);
        assert_eq!(m.get(2, 0), 17.5); assert_eq!(m.get(2, 1), 20.0); assert_eq!(m.get(2, 2), 22.5);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_div_scalar() {
        let m = &Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]) / 2.5;
        //@formatter:off
        assert_eq!(m.get(0, 0), 0.4); assert_eq!(m.get(0, 1), 0.8); assert_eq!(m.get(0, 2), 1.2);
        assert_eq!(m.get(1, 0), 1.6); assert_eq!(m.get(1, 1), 2.0); assert_eq!(m.get(1, 2), 2.4);
        assert_eq!(m.get(2, 0), 2.8); assert_eq!(m.get(2, 1), 3.2); assert_eq!(m.get(2, 2), 3.6);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_div_assign_scalar() {
        let mut m = &mut Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        m /= 2.5;
        //@formatter:off
        assert_eq!(m.get(0, 0), 0.4); assert_eq!(m.get(0, 1), 0.8); assert_eq!(m.get(0, 2), 1.2);
        assert_eq!(m.get(1, 0), 1.6); assert_eq!(m.get(1, 1), 2.0); assert_eq!(m.get(1, 2), 2.4);
        assert_eq!(m.get(2, 0), 2.8); assert_eq!(m.get(2, 1), 3.2); assert_eq!(m.get(2, 2), 3.6);
        //@formatter:on
    }

    #[test]
    fn matrix3x3_mul_point2() {
        let m = Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(&m * Point2d::new(-1.0, -2.0), Point2d::new(2.0 / 14.0, 8.0 / 14.0));
    }

    #[test]
    fn point2_mul_matrix3x3() {
        let m = Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(Point2d::new(-1.0, -2.0) * &m, Point2d::new(2.0 / 6.0, 4.0 / 6.0));
    }

    #[test]
    fn matrix3x3_mul_vector2() {
        let m = Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(&m * Vector2d::new(-1.0, -2.0), Vector2d::new(-5.0, -14.0));
    }

    #[test]
    fn vector2_mul_matrix3x3() {
        let m = Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(Vector2d::new(-1.0, -2.0) * &m, Vector2d::new(-9.0, -12.0));
    }

    #[test]
    fn matrix3x3_mul_matrix3x3() {
        let m1 = Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let m2 = Matrix3x3d::new([0.25, -0.75, 0.5, -0.5, 0.25, -0.75, 1.25, -1.75, 1.5]);
        let m = &m1 * &m2;
        //@formatter:off
        assert_eq!(m.get(0, 0), 3.0); assert_eq!(m.get(0, 1), -5.5); assert_eq!(m.get(0, 2), 3.5);
        assert_eq!(m.get(1, 0), 6.0); assert_eq!(m.get(1, 1), -12.25); assert_eq!(m.get(1, 2), 7.25);
        assert_eq!(m.get(2, 0), 9.0); assert_eq!(m.get(2, 1), -19.0); assert_eq!(m.get(2, 2), 11.0);
        //@formatter:on
    }

    #[test]
    fn transform2_new() {
        let m1 = Matrix3x3d::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let m2 = Matrix3x3d::new([0.25, -0.75, 0.5, -0.5, 0.25, -0.75, 1.25, -1.75, 1.5]);
        let t = Transform2d::new(m1.clone(), m2.clone());
        assert_eq!(t.forward, m1);
        assert_eq!(t.inverse, m2);
    }

    #[test]
    fn transform2_identity() {
        let t = Transform2d::identity();
        assert_eq!(t.forward, Matrix3x3d::identity());
        assert_eq!(t.inverse, Matrix3x3d::identity());
    }

    #[test]
    fn transform2_translate() {
        let v = Vector2d::new(-3.0, 4.0);
        let t = Transform2d::translate(v);
        assert_eq!(t.forward, Matrix3x3d::translate(v));
        assert_eq!(t.inverse, Matrix3x3d::translate(-v));
    }

    #[test]
    fn transform2_rotate() {
        let angle = std::f64::consts::FRAC_PI_6;
        let t = Transform2d::rotate(angle);
        assert_eq!(t.forward, Matrix3x3d::rotate(angle));
        assert_eq!(t.inverse, Matrix3x3d::rotate(-angle));
    }

    #[test]
    fn transform2_scale() {
        let t = Transform2d::scale(3.25, 2.5);
        assert_eq!(t.forward, Matrix3x3d::scale(3.25, 2.5));
        assert_eq!(t.inverse, Matrix3x3d::scale(1.0 / 3.25, 1.0 / 2.5));
    }

    #[test]
    fn transform2_scale_uniform() {
        let t = Transform2d::scale_uniform(4.0);
        assert_eq!(t.forward, Matrix3x3d::scale_uniform(4.0));
        assert_eq!(t.inverse, Matrix3x3d::scale_uniform(1.0 / 4.0));
    }

    #[test]
    fn transform2_and_then() {
        let angle = std::f64::consts::FRAC_PI_8;
        let v = Vector2d::new(-2.0, 3.0);
        let t = Transform2d::rotate(angle).and_then(&Transform2d::translate(v));
        assert_eq!(t.forward, &Matrix3x3d::translate(v) * &Matrix3x3d::rotate(angle));
        assert_eq!(t.inverse, &Matrix3x3d::rotate(-angle) * &Matrix3x3d::translate(-v));
    }

    #[test]
    fn transform2_inverse() {
        let t1 = Transform2d::rotate(std::f64::consts::FRAC_PI_8).and_then(&Transform2d::translate(Vector2d::new(-2.0, 3.0)));
        let t2 = t1.inverse();
        assert_eq!(t1.forward, t2.inverse);
        assert_eq!(t1.inverse, t2.forward);
    }

    #[test]
    fn transform2_from_matrix() {
        let m = Matrix3x3d::translate(Vector2d::new(-2.0, 3.0));
        let t = Transform2d::try_from(m.clone()).unwrap();
        assert_eq!(t.forward, m);
        assert_eq!(t.inverse, m.inverse().unwrap());
    }

    #[test]
    fn transform_point2() {
        let angle = std::f64::consts::FRAC_PI_8;
        let v = Vector2d::new(-2.0, 3.0);
        let t = Transform2d::rotate(angle).and_then(&Transform2d::translate(v));
        let p = t.transform(Point2d::new(-1.0, 1.0));
        assert_eq!(p, &(&Matrix3x3d::translate(v) * &Matrix3x3d::rotate(angle)) * Point2d::new(-1.0, 1.0));
    }

    #[test]
    fn transform_vector2() {
        let angle = std::f64::consts::FRAC_PI_8;
        let scale = 2.25;
        let t = Transform2d::scale_uniform(scale).and_then(&Transform2d::rotate(angle));
        let v = t.transform(Vector2d::new(-1.0, 1.0));
        assert_eq!(v, &(&Matrix3x3d::rotate(angle) * &Matrix3x3d::scale_uniform(scale)) * Vector2d::new(-1.0, 1.0));
    }

    #[test]
    fn transform_ray2() {
        let angle = std::f64::consts::FRAC_PI_8;
        let v = Vector2d::new(-2.0, 3.0);
        let t = Transform2d::rotate(angle).and_then(&Transform2d::translate(v));
        let origin = Point2d::new(-2.0, 1.5);
        let direction = Vector2d::new(3.5, 2.25);
        let r = t.transform(&Ray2d::new(origin, direction));
        assert_eq!(r, Ray2d::new(t.transform(origin), t.transform(direction)));
    }

    #[test]
    fn transform_bounding_box2() {
        // TODO
    }

    #[test]
    fn point3_new() {
        let p = Point3d::new(-1.0, 2.0, 3.0);
        assert_eq!(p.x, -1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn point3_origin() {
        let p = Point3d::origin();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
        assert_eq!(p.z, 0.0);
    }

    #[test]
    fn point3_min_dimension() {
        assert_eq!(Point3d::new(-1.0, 2.0, 3.0).min_dimension(), Dimension3::X);
        assert_eq!(Point3d::new(-3.0, 2.0, 3.0).min_dimension(), Dimension3::Y);
        assert_eq!(Point3d::new(-1.0, 2.0, 0.5).min_dimension(), Dimension3::Z);
    }

    #[test]
    fn point3_max_dimension() {
        assert_eq!(Point3d::new(-1.0, 2.0, 0.0).max_dimension(), Dimension3::Y);
        assert_eq!(Point3d::new(-3.0, 2.0, 0.0).max_dimension(), Dimension3::X);
        assert_eq!(Point3d::new(-1.0, 2.0, -2.5).max_dimension(), Dimension3::Z);
    }

    #[test]
    fn point3_floor() {
        assert_eq!(Point3d::new(-1.3, 2.6, 4.5).floor(), Point3d::new(-2.0, 2.0, 4.0));
    }

    #[test]
    fn point3_ceil() {
        assert_eq!(Point3d::new(-1.3, 2.6, 4.5).ceil(), Point3d::new(-1.0, 3.0, 5.0));
    }

    #[test]
    fn point3_round() {
        assert_eq!(Point3d::new(-1.6, 2.3, 4.5).round(), Point3d::new(-2.0, 2.0, 5.0));
    }

    #[test]
    fn point3_trunc() {
        assert_eq!(Point3d::new(-1.3, 2.6, 4.5).trunc(), Point3d::new(-1.0, 2.0, 4.0));
    }

    #[test]
    fn point3_fract() {
        assert_eq!(Point3d::new(-1.25, 2.5, 4.75).fract(), Point3d::new(-0.25, 0.5, 0.75));
    }

    #[test]
    fn point3_abs() {
        assert_eq!(Point3d::new(-1.3, 2.6, -4.5).abs(), Point3d::new(1.3, 2.6, 4.5));
        assert_eq!(Point3d::new(1.3, -2.6, 4.5).abs(), Point3d::new(1.3, 2.6, 4.5));
    }

    #[test]
    fn point3_permute() {
        assert_eq!(Point3d::new(1.0, 2.0, 3.0).permute(Dimension3::X, Dimension3::X, Dimension3::X), Point3d::new(1.0, 1.0, 1.0));
        assert_eq!(Point3d::new(1.0, 2.0, 3.0).permute(Dimension3::X, Dimension3::Y, Dimension3::Z), Point3d::new(1.0, 2.0, 3.0));
        assert_eq!(Point3d::new(1.0, 2.0, 3.0).permute(Dimension3::Y, Dimension3::Z, Dimension3::X), Point3d::new(2.0, 3.0, 1.0));
        assert_eq!(Point3d::new(1.0, 2.0, 3.0).permute(Dimension3::Z, Dimension3::X, Dimension3::Y), Point3d::new(3.0, 1.0, 2.0));
    }

    #[test]
    fn point3_min() {
        assert_eq!(min(Point3d::new(-1.0, 2.0, 3.0), Point3d::new(-3.0, 2.5, 1.5)), Point3d::new(-3.0, 2.0, 1.5));
    }

    #[test]
    fn point3_max() {
        assert_eq!(max(Point3d::new(-1.0, 2.0, 3.0), Point3d::new(-3.0, 2.5, 1.5)), Point3d::new(-1.0, 2.5, 3.0));
    }

    #[test]
    fn point3_distance() {
        assert_eq!(distance(Point3d::new(2.0, 7.0, 1.0), Point3d::new(3.0, 4.0, -1.0)), f64::sqrt(14.0));
    }

    #[test]
    fn point3_closest() {
        let p1 = Point3d::new(4.0, 1.0, 3.0);
        let p2 = Point3d::new(1.0, 5.0, -2.0);
        assert_eq!(Point3d::new(-1.0, 2.0, 0.0).closest(p1, p2), p2);
    }

    #[test]
    fn point3_farthest() {
        let p1 = Point3d::new(4.0, 1.0, 3.0);
        let p2 = Point3d::new(1.0, 5.0, -2.0);
        assert_eq!(Point3d::new(-1.0, 2.0, 0.0).farthest(p1, p2), p1);
    }

    #[test]
    fn point3_index() {
        let p = Point3d::new(1.0, 2.0, 3.0);
        assert_eq!(p[Dimension3::X], 1.0);
        assert_eq!(p[Dimension3::Y], 2.0);
        assert_eq!(p[Dimension3::Z], 3.0);
    }

    #[test]
    fn point3_index_mut() {
        let mut p = Point3d::new(1.0, 2.0, 3.0);
        p[Dimension3::X] = 3.0;
        p[Dimension3::Y] = -1.0;
        p[Dimension3::Z] = 2.0;
        assert_eq!(p, Point3d::new(3.0, -1.0, 2.0));
    }

    #[test]
    fn point3_add_vector3() {
        let p = Point3d::new(1.0, 2.0, 3.0);
        let v = Vector3d::new(-0.5, 1.5, 2.5);
        assert_eq!(p + v, Point3d::new(0.5, 3.5, 5.5));
    }

    #[test]
    fn point3_add_assign_vector3() {
        let mut p = Point3d::new(1.0, 2.0, 3.0);
        let v = Vector3d::new(-0.5, 1.5, 2.5);
        p += v;
        assert_eq!(p, Point3d::new(0.5, 3.5, 5.5));
    }

    #[test]
    fn point3_sub_vector3() {
        let p = Point3d::new(1.0, 2.0, 3.0);
        let v = Vector3d::new(-0.5, 1.5, 2.75);
        assert_eq!(p - v, Point3d::new(1.5, 0.5, 0.25));
    }

    #[test]
    fn point3_sub_assign_vector3() {
        let mut p = Point3d::new(1.0, 2.0, 3.0);
        let v = Vector3d::new(-0.5, 1.5, 2.75);
        p -= v;
        assert_eq!(p, Point3d::new(1.5, 0.5, 0.25));
    }

    #[test]
    fn point3_sub_point3() {
        let p1 = Point3d::new(4.0, 2.0, 1.0);
        let p2 = Point3d::new(1.0, 5.0, 2.0);
        assert_eq!(p1 - p2, Vector3d::new(3.0, -3.0, -1.0));
    }

    #[test]
    fn point3_neg() {
        assert_eq!(-Point3d::new(1.0, -2.0, 3.0), Point3d::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn point3_mul_scalar() {
        assert_eq!(Point3d::new(2.5, -1.5, 3.0) * 2.0, Point3d::new(5.0, -3.0, 6.0));
    }

    #[test]
    fn scalar_mul_point3() {
        assert_eq!(2.0 * Point3d::new(2.5, -1.5, 3.0), Point3d::new(5.0, -3.0, 6.0));
    }

    #[test]
    fn point3_mul_assign_scalar() {
        let mut p = Point3d::new(2.5, -1.5, 3.0);
        p *= 2.0;
        assert_eq!(p, Point3d::new(5.0, -3.0, 6.0));
    }

    #[test]
    fn point3_div_scalar() {
        assert_eq!(Point3d::new(2.5, -1.5, 3.0) / 2.0, Point3d::new(1.25, -0.75, 1.5));
    }

    #[test]
    fn point3_div_assign_scalar() {
        let mut p = Point3d::new(2.5, -1.5, 3.0);
        p /= 2.0;
        assert_eq!(p, Point3d::new(1.25, -0.75, 1.5));
    }

    #[test]
    fn point3_from_vector3() {
        let p = Point3d::from(Vector3d::new(1.0, 2.0, 3.0));
        assert_eq!(p, Point3d::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn vector3_new() {
        let v = Vector3d::new(-1.0, 2.0, -3.0);
        assert_eq!(v.x, -1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, -3.0);
    }

    #[test]
    fn vector3_zero() {
        let v = Vector3d::zero();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn vector3_x_axis() {
        let v = Vector3d::x_axis();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn vector3_y_axis() {
        let v = Vector3d::y_axis();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 1.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn vector3_z_axis() {
        let v = Vector3d::z_axis();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 1.0);
    }

    #[test]
    fn vector3_axis() {
        assert_eq!(Vector3d::axis(Dimension3::X), Vector3d::x_axis());
        assert_eq!(Vector3d::axis(Dimension3::Y), Vector3d::y_axis());
        assert_eq!(Vector3d::axis(Dimension3::Z), Vector3d::z_axis());
    }

    #[test]
    fn vector3_normalize() {
        let v = Vector3d::new(3.0, -2.0, 1.0);
        assert_eq!(v.normalize(), v / f64::sqrt(14.0));
    }

    #[test]
    fn vector3_min_dimension() {
        assert_eq!(Vector3d::new(-1.0, 2.0, 3.0).min_dimension(), Dimension3::X);
        assert_eq!(Vector3d::new(-3.0, 2.0, 2.5).min_dimension(), Dimension3::Y);
    }

    #[test]
    fn vector3_max_dimension() {
        assert_eq!(Vector3d::new(-1.0, 2.0, 0.5).max_dimension(), Dimension3::Y);
        assert_eq!(Vector3d::new(-3.0, 2.0, 0.5).max_dimension(), Dimension3::X);
        assert_eq!(Vector3d::new(-3.0, 2.0, 4.0).max_dimension(), Dimension3::Z);
    }

    #[test]
    fn vector3_floor() {
        assert_eq!(Vector3d::new(-1.3, 2.6, 3.75).floor(), Vector3d::new(-2.0, 2.0, 3.0));
    }

    #[test]
    fn vector3_ceil() {
        assert_eq!(Vector3d::new(-1.3, 2.6, 3.75).ceil(), Vector3d::new(-1.0, 3.0, 4.0));
    }

    #[test]
    fn vector3_round() {
        assert_eq!(Vector3d::new(-1.6, 2.3, 3.75).round(), Vector3d::new(-2.0, 2.0, 4.0));
    }

    #[test]
    fn vector3_trunc() {
        assert_eq!(Vector3d::new(-1.3, 2.6, 3.75).trunc(), Vector3d::new(-1.0, 2.0, 3.0));
    }

    #[test]
    fn vector3_fract() {
        assert_eq!(Vector3d::new(-1.25, 2.5, 3.75).fract(), Vector3d::new(-0.25, 0.5, 0.75));
    }

    #[test]
    fn vector3_abs() {
        assert_eq!(Vector3d::new(-1.3, 2.6, -2.0).abs(), Vector3d::new(1.3, 2.6, 2.0));
    }

    #[test]
    fn vector3_permute() {
        assert_eq!(Vector3d::new(1.0, 2.0, 3.0).permute(Dimension3::X, Dimension3::X, Dimension3::X), Vector3d::new(1.0, 1.0, 1.0));
        assert_eq!(Vector3d::new(1.0, 2.0, 3.0).permute(Dimension3::X, Dimension3::Y, Dimension3::Z), Vector3d::new(1.0, 2.0, 3.0));
        assert_eq!(Vector3d::new(1.0, 2.0, 3.0).permute(Dimension3::Y, Dimension3::Z, Dimension3::X), Vector3d::new(2.0, 3.0, 1.0));
        assert_eq!(Vector3d::new(1.0, 2.0, 3.0).permute(Dimension3::Z, Dimension3::X, Dimension3::Y), Vector3d::new(3.0, 1.0, 2.0));
    }

    #[test]
    fn vector3_min() {
        assert_eq!(min(Vector3d::new(-1.0, 2.0, 3.0), Vector3d::new(-3.0, 2.5, 3.5)), Vector3d::new(-3.0, 2.0, 3.0));
    }

    #[test]
    fn vector3_max() {
        assert_eq!(max(Vector3d::new(-1.0, 2.0, 3.0), Vector3d::new(-3.0, 2.5, 3.5)), Vector3d::new(-1.0, 2.5, 3.5));
    }

    #[test]
    fn vector3_length() {
        assert_eq!(Vector3d::new(2.0, 3.0, 6.0).length(), 7.0);
    }

    #[test]
    fn vector3_shortest() {
        let v1 = Vector3d::new(-1.0, -3.0, 0.5);
        let v2 = Vector3d::new(2.0, 1.5, 0.5);
        assert_eq!(shortest(v1, v2), v2);
    }

    #[test]
    fn vector3_longest() {
        let v1 = Vector3d::new(-1.0, -3.0, 0.5);
        let v2 = Vector3d::new(2.0, 1.5, 0.5);
        assert_eq!(longest(v1, v2), v1);
    }

    #[test]
    fn vector3_dot_vector3() {
        let v1 = Vector3d::new(-1.0, -3.0, 2.5);
        let v2 = Vector3d::new(2.0, 1.5, 0.5);
        assert_eq!(dot(v1, v2), -5.25);
    }

    #[test]
    fn vector3_index() {
        let v = Vector3d::new(1.0, 2.0, 3.0);
        assert_eq!(v[Dimension3::X], 1.0);
        assert_eq!(v[Dimension3::Y], 2.0);
        assert_eq!(v[Dimension3::Z], 3.0);
    }

    #[test]
    fn vector3_index_mut() {
        let mut v = Vector3d::new(1.0, 2.0, 3.0);
        v[Dimension3::X] = 3.0;
        v[Dimension3::Y] = -1.0;
        v[Dimension3::Z] = 2.5;
        assert_eq!(v, Vector3d::new(3.0, -1.0, 2.5));
    }

    #[test]
    fn vector3_add_vector3() {
        let v1 = Vector3d::new(1.0, 2.0, 3.0);
        let v2 = Vector3d::new(-0.5, 1.5, 3.0);
        assert_eq!(v1 + v2, Vector3d::new(0.5, 3.5, 6.0));
    }

    #[test]
    fn vector3_add_assign_vector3() {
        let mut v1 = Vector3d::new(1.0, 2.0, 3.0);
        let v2 = Vector3d::new(-0.5, 1.5, 3.0);
        v1 += v2;
        assert_eq!(v1, Vector3d::new(0.5, 3.5, 6.0));
    }

    #[test]
    fn vector3_sub_vector3() {
        let v1 = Vector3d::new(1.0, 2.0, 3.0);
        let v2 = Vector3d::new(-0.5, 1.5, 3.5);
        assert_eq!(v1 - v2, Vector3d::new(1.5, 0.5, -0.5));
    }

    #[test]
    fn vector3_sub_assign_vector3() {
        let mut v1 = Vector3d::new(1.0, 2.0, 3.0);
        let v2 = Vector3d::new(-0.5, 1.5, 3.5);
        v1 -= v2;
        assert_eq!(v1, Vector3d::new(1.5, 0.5, -0.5));
    }

    #[test]
    fn vector3_neg() {
        assert_eq!(-Vector3d::new(1.0, -2.0, 3.0), Vector3d::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn vector3_mul_scalar() {
        assert_eq!(Vector3d::new(2.5, -1.5, 4.0) * 2.0, Vector3d::new(5.0, -3.0, 8.0));
    }

    #[test]
    fn scalar_mul_vector3() {
        assert_eq!(2.0 * Vector3d::new(2.5, -1.5, 4.0), Vector3d::new(5.0, -3.0, 8.0));
    }

    #[test]
    fn vector3_mul_assign_scalar() {
        let mut v = Vector3d::new(2.5, -1.5, 4.0);
        v *= 2.0;
        assert_eq!(v, Vector3d::new(5.0, -3.0, 8.0));
    }

    #[test]
    fn vector3_div_scalar() {
        assert_eq!(Vector3d::new(2.5, -1.5, 4.0) / 2.0, Vector3d::new(1.25, -0.75, 2.0));
    }

    #[test]
    fn vector3_div_assign_scalar() {
        let mut v = Vector3d::new(2.5, -1.5, 4.0);
        v /= 2.0;
        assert_eq!(v, Vector3d::new(1.25, -0.75, 2.0));
    }

    #[test]
    fn vector3_from_point3() {
        let v = Vector3d::from(Point3d::new(1.0, 2.0, 3.0));
        assert_eq!(v, Vector3d::new(1.0, 2.0, 3.0));
    }

    // TODO: Tests for BoundingBox3, Matrix4x4, Transform3
}

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

use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, Sub, SubAssign};
use std::sync::Arc;

use array_macro::array;

/// Scalar type used for the elements of points, vectors and matrices.
pub type Scalar = f32;

/// Trait for types that have `min()` and `max()` methods.
pub trait MinMax {
    /// Compares and returns the minimum of two values.
    fn min(self, other: Self) -> Self;

    /// Compares and returns the maximum of two values.
    fn max(self, other: Self) -> Self;
}

/// Compares and returns the minimum of two values.
pub fn min<T: MinMax>(a: T, b: T) -> T {
    a.min(b)
}

/// Compares and returns the maximum of two values.
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
pub fn shortest<T: RelativeLength>(a: T, b: T) -> T {
    a.shortest(b)
}

/// Returns the longest of two values.
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

/// Dimension in 2D space.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Dimension2 { X, Y }

/// Point in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point2 {
    pub x: Scalar,
    pub y: Scalar,
}

/// Vector in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector2 {
    pub x: Scalar,
    pub y: Scalar,
}

/// Ray in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Ray2 {
    pub origin: Point2,
    pub direction: Vector2,
}

/// Axis-aligned bounding box in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct BoundingBox2 {
    pub min: Point2,
    pub max: Point2,
}

/// Matrix with 3 rows and 3 columns for transformations in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix3x3 {
    m: [Scalar; 9]
}

/// Transform for transformations in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Transform2 {
    pub forward: Arc<Matrix3x3>,
    pub inverse: Arc<Matrix3x3>,
}

/// Dimension in 3D space.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Dimension3 { X, Y, Z }

/// Point in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Point3 {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
}

/// Vector in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector3 {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
}

/// Surface normal in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Normal3 {
    pub x: Scalar,
    pub y: Scalar,
    pub z: Scalar,
}

/// Ray in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Ray3 {
    pub origin: Point3,
    pub direction: Vector3,
}

/// Axis-aligned bounding box in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct BoundingBox3 {
    pub min: Point3,
    pub max: Point3,
}

/// Matrix with 4 rows and 4 columns for transformations in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix4x4 {
    m: [Scalar; 16]
}

/// Transform for transformations in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Transform3 {
    pub forward: Arc<Matrix4x4>,
    pub inverse: Arc<Matrix4x4>,
}

// ===== Scalar ================================================================================================================================================

impl MinMax for Scalar {
    /// Compares and returns the minimum of two scalars.
    fn min(self, s: Scalar) -> Scalar {
        Scalar::min(self, s)
    }

    /// Compares and returns the maximum of two scalars.
    fn max(self, s: Scalar) -> Scalar {
        Scalar::max(self, s)
    }
}

// ===== Point2 ================================================================================================================================================

impl Point2 {
    /// Creates and returns a new `Point2` with x and y coordinates.
    pub fn new(x: Scalar, y: Scalar) -> Point2 {
        Point2 { x, y }
    }

    /// Returns a `Point2` which represents the origin (x = 0 and y = 0).
    pub fn origin() -> Point2 {
        Point2::new(0.0, 0.0)
    }

    /// Returns the dimension with the smallest extent of this point.
    pub fn min_dimension(self) -> Dimension2 {
        let Point2 { x, y } = self.abs();
        if x <= y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this point.
    pub fn max_dimension(self) -> Dimension2 {
        let Point2 { x, y } = self.abs();
        if x > y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the element-wise floor of this point.
    pub fn floor(self) -> Point2 {
        Point2::new(self.x.floor(), self.y.floor())
    }

    /// Returns the element-wise ceiling of this point.
    pub fn ceil(self) -> Point2 {
        Point2::new(self.x.ceil(), self.y.ceil())
    }

    /// Returns the element-wise rounded value of this point.
    pub fn round(self) -> Point2 {
        Point2::new(self.x.round(), self.y.round())
    }

    /// Returns the element-wise truncated value of this point.
    pub fn trunc(self) -> Point2 {
        Point2::new(self.x.trunc(), self.y.trunc())
    }

    /// Returns the element-wise fractional value of this point.
    pub fn fract(self) -> Point2 {
        Point2::new(self.x.fract(), self.y.fract())
    }

    /// Returns the element-wise absolute value of this point.
    pub fn abs(self) -> Point2 {
        Point2::new(self.x.abs(), self.y.abs())
    }

    /// Returns a point with a permutation of the elements of this point.
    pub fn permute(self, dim_x: Dimension2, dim_y: Dimension2) -> Point2 {
        Point2::new(self[dim_x], self[dim_y])
    }
}

impl MinMax for Point2 {
    /// Returns the element-wise minimum of two points.
    fn min(self, p: Point2) -> Point2 {
        Point2::new(min(self.x, p.x), min(self.y, p.y))
    }

    /// Returns the element-wise maximum of two points.
    fn max(self, p: Point2) -> Point2 {
        Point2::new(max(self.x, p.x), max(self.y, p.y))
    }
}

impl Distance for Point2 {
    type Output = Scalar;

    /// Computes and returns the distance between two points.
    fn distance(self, p: Point2) -> Scalar {
        (p - self).length()
    }
}

impl RelativeDistance for Point2 {
    /// Checks which of the points `p1` and `p2` is closest to this point and returns the closest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    fn closest(self, p1: Point2, p2: Point2) -> Point2 {
        let (dp1, dp2) = (p1 - self, p2 - self);
        if dot(dp1, dp1) <= dot(dp2, dp2) { p1 } else { p2 }
    }

    /// Checks which of the points `p1` and `p2` is farthest from this point and returns the farthest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    fn farthest(self, p1: Point2, p2: Point2) -> Point2 {
        let (dp1, dp2) = (p1 - self, p2 - self);
        if dot(dp1, dp1) > dot(dp2, dp2) { p1 } else { p2 }
    }
}

impl Index<Dimension2> for Point2 {
    type Output = Scalar;

    fn index(&self, dim: Dimension2) -> &Scalar {
        match dim {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl IndexMut<Dimension2> for Point2 {
    fn index_mut(&mut self, dim: Dimension2) -> &mut Scalar {
        match dim {
            Dimension2::X => &mut self.x,
            Dimension2::Y => &mut self.y,
        }
    }
}

impl Add<Vector2> for Point2 {
    type Output = Point2;

    fn add(self, v: Vector2) -> Point2 {
        Point2::new(self.x + v.x, self.y + v.y)
    }
}

impl AddAssign<Vector2> for Point2 {
    fn add_assign(&mut self, v: Vector2) {
        self.x += v.x;
        self.y += v.y;
    }
}

impl Sub<Vector2> for Point2 {
    type Output = Point2;

    fn sub(self, v: Vector2) -> Point2 {
        Point2::new(self.x - v.x, self.y - v.y)
    }
}

impl SubAssign<Vector2> for Point2 {
    fn sub_assign(&mut self, v: Vector2) {
        self.x -= v.x;
        self.y -= v.y;
    }
}

impl Sub<Point2> for Point2 {
    type Output = Vector2;

    fn sub(self, p: Point2) -> Vector2 {
        Vector2::new(self.x - p.x, self.y - p.y)
    }
}

impl Neg for Point2 {
    type Output = Point2;

    fn neg(self) -> Point2 {
        Point2::new(-self.x, -self.y)
    }
}

impl Mul<Scalar> for Point2 {
    type Output = Point2;

    fn mul(self, s: Scalar) -> Point2 {
        Point2::new(self.x * s, self.y * s)
    }
}

impl Mul<Point2> for Scalar {
    type Output = Point2;

    fn mul(self, p: Point2) -> Point2 {
        p * self
    }
}

impl MulAssign<Scalar> for Point2 {
    fn mul_assign(&mut self, s: Scalar) {
        self.x *= s;
        self.y *= s;
    }
}

impl Div<Scalar> for Point2 {
    type Output = Point2;

    fn div(self, s: Scalar) -> Point2 {
        Point2::new(self.x / s, self.y / s)
    }
}

impl DivAssign<Scalar> for Point2 {
    fn div_assign(&mut self, s: Scalar) {
        self.x /= s;
        self.y /= s;
    }
}

impl Transform<Point2> for Transform2 {
    type Output = Point2;

    /// Transforms a point.
    fn transform(&self, p: Point2) -> Point2 {
        &*self.forward * p
    }
}

impl From<Vector2> for Point2 {
    fn from(v: Vector2) -> Point2 {
        Point2::new(v.x, v.y)
    }
}

// ===== Vector2 ===============================================================================================================================================

impl Vector2 {
    /// Creates and returns a new `Vector2` with x and y coordinates.
    pub fn new(x: Scalar, y: Scalar) -> Vector2 {
        Vector2 { x, y }
    }

    /// Returns a `Vector2` which represents the zero vector (x = 0 and y = 0).
    pub fn zero() -> Vector2 {
        Vector2::new(0.0, 0.0)
    }

    /// Returns a `Vector2` of length 1 which represents the X axis (x = 1 and y = 0).
    pub fn x_axis() -> Vector2 {
        Vector2::new(1.0, 0.0)
    }

    /// Returns a `Vector2` of length 1 which represents the Y axis (x = 0 and y = 1).
    pub fn y_axis() -> Vector2 {
        Vector2::new(0.0, 1.0)
    }

    /// Returns a `Vector2` of length 1 which represents the axis specified by a dimension.
    pub fn axis(dim: Dimension2) -> Vector2 {
        match dim {
            Dimension2::X => Vector2::x_axis(),
            Dimension2::Y => Vector2::y_axis(),
        }
    }

    /// Creates and returns a new `Vector2` which points in the same direction as this vector, but with length 1.
    pub fn normalize(self) -> Vector2 {
        self / self.length()
    }

    /// Returns the dimension with the smallest extent of this vector.
    pub fn min_dimension(self) -> Dimension2 {
        let Vector2 { x, y } = self.abs();
        if x <= y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this vector.
    pub fn max_dimension(self) -> Dimension2 {
        let Vector2 { x, y } = self.abs();
        if x > y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the element-wise floor of this vector.
    pub fn floor(self) -> Vector2 {
        Vector2::new(self.x.floor(), self.y.floor())
    }

    /// Returns the element-wise ceiling of this vector.
    pub fn ceil(self) -> Vector2 {
        Vector2::new(self.x.ceil(), self.y.ceil())
    }

    /// Returns the element-wise rounded value of this vector.
    pub fn round(self) -> Vector2 {
        Vector2::new(self.x.round(), self.y.round())
    }

    /// Returns the element-wise truncated value of this vector.
    pub fn trunc(self) -> Vector2 {
        Vector2::new(self.x.trunc(), self.y.trunc())
    }

    /// Returns the element-wise fractional value of this vector.
    pub fn fract(self) -> Vector2 {
        Vector2::new(self.x.fract(), self.y.fract())
    }

    /// Returns the element-wise absolute value of this vector.
    pub fn abs(self) -> Vector2 {
        Vector2::new(self.x.abs(), self.y.abs())
    }

    /// Returns a point with a permutation of the elements of this vector.
    pub fn permute(self, dim_x: Dimension2, dim_y: Dimension2) -> Vector2 {
        Vector2::new(self[dim_x], self[dim_y])
    }
}

impl MinMax for Vector2 {
    /// Returns the element-wise minimum of two vectors.
    fn min(self, v: Vector2) -> Vector2 {
        Vector2::new(min(self.x, v.x), min(self.y, v.y))
    }

    /// Returns the element-wise maximum of two vectors.
    fn max(self, v: Vector2) -> Vector2 {
        Vector2::new(max(self.x, v.x), max(self.y, v.y))
    }
}

impl Length for Vector2 {
    type Output = Scalar;

    /// Computes and returns the length of a vector.
    fn length(self) -> Scalar {
        Scalar::sqrt(dot(self, self))
    }
}

impl RelativeLength for Vector2 {
    /// Returns the shortest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    fn shortest(self, v: Vector2) -> Vector2 {
        if dot(self, self) <= dot(v, v) { self } else { v }
    }

    /// Returns the longest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    fn longest(self, v: Vector2) -> Vector2 {
        if dot(self, self) > dot(v, v) { self } else { v }
    }
}

impl DotProduct<Vector2> for Vector2 {
    type Output = Scalar;

    /// Computes and returns the dot product between two vectors.
    fn dot(self, v: Vector2) -> Scalar {
        self.x * v.x + self.y * v.y
    }
}

impl Index<Dimension2> for Vector2 {
    type Output = Scalar;

    fn index(&self, dim: Dimension2) -> &Scalar {
        match dim {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl IndexMut<Dimension2> for Vector2 {
    fn index_mut(&mut self, dim: Dimension2) -> &mut Scalar {
        match dim {
            Dimension2::X => &mut self.x,
            Dimension2::Y => &mut self.y,
        }
    }
}

impl Add<Vector2> for Vector2 {
    type Output = Vector2;

    fn add(self, v: Vector2) -> Vector2 {
        Vector2::new(self.x + v.x, self.y + v.y)
    }
}

impl AddAssign<Vector2> for Vector2 {
    fn add_assign(&mut self, v: Vector2) {
        self.x += v.x;
        self.y += v.y;
    }
}

impl Sub<Vector2> for Vector2 {
    type Output = Vector2;

    fn sub(self, v: Vector2) -> Vector2 {
        Vector2::new(self.x - v.x, self.y - v.y)
    }
}

impl SubAssign<Vector2> for Vector2 {
    fn sub_assign(&mut self, v: Vector2) {
        self.x -= v.x;
        self.y -= v.y;
    }
}

impl Neg for Vector2 {
    type Output = Vector2;

    fn neg(self) -> Vector2 {
        Vector2::new(-self.x, -self.y)
    }
}

impl Mul<Scalar> for Vector2 {
    type Output = Vector2;

    fn mul(self, s: Scalar) -> Vector2 {
        Vector2::new(self.x * s, self.y * s)
    }
}

impl Mul<Vector2> for Scalar {
    type Output = Vector2;

    fn mul(self, v: Vector2) -> Vector2 {
        v * self
    }
}

impl MulAssign<Scalar> for Vector2 {
    fn mul_assign(&mut self, s: Scalar) {
        self.x *= s;
        self.y *= s;
    }
}

impl Div<Scalar> for Vector2 {
    type Output = Vector2;

    fn div(self, s: Scalar) -> Vector2 {
        Vector2::new(self.x / s, self.y / s)
    }
}

impl DivAssign<Scalar> for Vector2 {
    fn div_assign(&mut self, s: Scalar) {
        self.x /= s;
        self.y /= s;
    }
}

impl Transform<Vector2> for Transform2 {
    type Output = Vector2;

    /// Transforms a vector.
    fn transform(&self, v: Vector2) -> Vector2 {
        &*self.forward * v
    }
}

impl From<Point2> for Vector2 {
    fn from(p: Point2) -> Vector2 {
        Vector2::new(p.x, p.y)
    }
}

// ===== Ray2 ==================================================================================================================================================

impl Ray2 {
    /// Creates and returns a new `Ray2` with an origin point and direction vector.
    pub fn new(origin: Point2, direction: Vector2) -> Ray2 {
        Ray2 { origin, direction }
    }

    /// Computes and returns a point at a distance along this ray.
    pub fn at(&self, distance: Scalar) -> Point2 {
        self.origin + self.direction * distance
    }
}

impl Transform<&Ray2> for Transform2 {
    type Output = Ray2;

    /// Transforms a ray.
    fn transform(&self, ray: &Ray2) -> Ray2 {
        Ray2::new(self.transform(ray.origin), self.transform(ray.direction))
    }
}

// ===== BoundingBox2 ==========================================================================================================================================

impl BoundingBox2 {
    /// Creates and returns a new `BoundingBox2` with minimum and maximum corner points.
    pub fn new(min: Point2, max: Point2) -> BoundingBox2 {
        BoundingBox2 { min, max }
    }

    /// Returns an empty `BoundingBox2`.
    pub fn empty() -> BoundingBox2 {
        BoundingBox2::new(Point2::new(Scalar::INFINITY, Scalar::INFINITY), Point2::new(Scalar::NEG_INFINITY, Scalar::NEG_INFINITY))
    }

    /// Returns an infinite `BoundingBox2` which contains all of 2D space.
    pub fn infinite() -> BoundingBox2 {
        BoundingBox2::new(Point2::new(Scalar::NEG_INFINITY, Scalar::NEG_INFINITY), Point2::new(Scalar::INFINITY, Scalar::INFINITY))
    }

    /// Returns the width (extent in the X dimension) of this bounding box.
    pub fn width(&self) -> Scalar {
        self.max.x - self.min.x
    }

    /// Returns the height (extent in the Y dimension) of this bounding box.
    pub fn height(&self) -> Scalar {
        self.max.y - self.min.y
    }

    /// Returns the extent of this bounding box in a dimension.
    pub fn extent(&self, dim: Dimension2) -> Scalar {
        match dim {
            Dimension2::X => self.width(),
            Dimension2::Y => self.height(),
        }
    }

    /// Returns the dimension with the smallest extent of this bounding box.
    pub fn min_dimension(&self) -> Dimension2 {
        let d = self.diagonal();
        if d.x <= d.y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the largest extent of this bounding box.
    pub fn max_dimension(&self) -> Dimension2 {
        let d = self.diagonal();
        if d.x > d.y { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the area (width times height) of this bounding box.
    pub fn area(&self) -> Scalar {
        let d = self.diagonal();
        d.x * d.y
    }

    /// Returns the center point of this bounding box.
    pub fn center(&self) -> Point2 {
        self.min + self.diagonal() * 0.5
    }

    /// Returns a corner point of this bounding box, indicated by an index (which must be between 0 and 3 inclusive).
    pub fn corner(&self, index: usize) -> Point2 {
        debug_assert!(index < 4, "Invalid corner index: {}", index);
        let x = if index & 0b01 == 0 { self.min.x } else { self.max.x };
        let y = if index & 0b10 == 0 { self.min.y } else { self.max.y };
        Point2::new(x, y)
    }

    /// Returns the diagonal of this bounding box as a vector.
    pub fn diagonal(&self) -> Vector2 {
        self.max - self.min
    }

    /// Checks if two bounding boxes overlap.
    pub fn overlaps(&self, bb: &BoundingBox2) -> bool {
        //@formatter:off
        self.max.x >= bb.min.x && self.min.x <= bb.max.x &&
            self.max.y >= bb.min.y && self.min.y <= bb.max.y
        //@formatter:on
    }

    /// Checks if a point is inside this bounding box.
    pub fn is_inside(&self, p: Point2) -> bool {
        //@formatter:off
        p.x >= self.min.x && p.x <= self.max.x &&
            p.y >= self.min.y && p.y <= self.max.y
        //@formatter:on
    }

    /// Computes the closest intersection of this bounding box with a ray within a range.
    ///
    /// Returns a `Some` containing the closest intersection, or `None` if the ray does not intersect the bounding box within the range.
    pub fn intersect_ray(&self, ray: &Ray2, range: &Range<Scalar>) -> Option<Scalar> {
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

impl Union<&BoundingBox2> for BoundingBox2 {
    type Output = BoundingBox2;

    /// Computes and returns the union between two bounding boxes.
    ///
    /// The union is the smallest bounding box that contains both bounding boxes.
    fn union(self, bb: &BoundingBox2) -> BoundingBox2 {
        BoundingBox2::new(min(self.min, bb.min), max(self.max, bb.max))
    }
}

impl Union<Point2> for BoundingBox2 {
    type Output = BoundingBox2;

    /// Computes and returns the union between this bounding box and a point.
    ///
    /// The union is the smallest bounding box that contains both the bounding box and the point.
    fn union(self, p: Point2) -> BoundingBox2 {
        BoundingBox2::new(min(self.min, p), max(self.max, p))
    }
}

impl Intersection<&BoundingBox2> for BoundingBox2 {
    type Output = BoundingBox2;

    /// Computes and returns the intersection between two bounding boxes.
    ///
    /// The intersection is the largest bounding box that contains the region where the two bounding boxes overlap.
    ///
    /// Returns `Some` when the bounding boxes overlap; `None` if the bounding boxes do not overlap.
    fn intersection(self, bb: &BoundingBox2) -> Option<BoundingBox2> {
        if self.overlaps(bb) {
            Some(BoundingBox2::new(max(self.min, bb.min), min(self.max, bb.max)))
        } else {
            None
        }
    }
}

impl Transform<&BoundingBox2> for Transform2 {
    type Output = BoundingBox2;

    /// Transforms a bounding box.
    fn transform(&self, bb: &BoundingBox2) -> BoundingBox2 {
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

impl Matrix3x3 {
    /// Creates and returns a new `Matrix3x3` with the specified elements.
    pub fn new(m: [Scalar; 9]) -> Matrix3x3 {
        Matrix3x3 { m }
    }

    /// Returns a `Matrix3x3` which represents the identity matrix.
    pub fn identity() -> Matrix3x3 {
        Matrix3x3::new([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ])
    }

    /// Returns a translation matrix which translates over a vector.
    pub fn translate(v: Vector2) -> Matrix3x3 {
        Matrix3x3::new([
            1.0, 0.0, v.x,
            0.0, 1.0, v.y,
            0.0, 0.0, 1.0,
        ])
    }

    /// Returns a rotation matrix which rotates around the origin.
    pub fn rotate(angle: Scalar) -> Matrix3x3 {
        let (sin, cos) = angle.sin_cos();

        Matrix3x3::new([
            cos, -sin, 0.0,
            sin, cos, 0.0,
            0.0, 0.0, 1.0,
        ])
    }

    /// Returns a matrix which scales by factors in the X and Y dimensions.
    pub fn scale(sx: Scalar, sy: Scalar) -> Matrix3x3 {
        Matrix3x3::new([
            sx, 0.0, 0.0,
            0.0, sy, 0.0,
            0.0, 0.0, 1.0,
        ])
    }

    /// Returns a matrix which scales uniformly in all dimensions by a factor.
    pub fn scale_uniform(s: Scalar) -> Matrix3x3 {
        Matrix3x3::new([
            s, 0.0, 0.0,
            0.0, s, 0.0,
            0.0, 0.0, 1.0,
        ])
    }

    /// Returns an element at a row and column of the matrix.
    pub fn get(&self, row: usize, col: usize) -> Scalar {
        self.m[row * 3 + col]
    }

    /// Returns a mutable reference to an element at a row and column of the matrix.
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut Scalar {
        &mut self.m[row * 3 + col]
    }

    /// Sets the value of an element at a row and column of the matrix.
    pub fn set(&mut self, row: usize, col: usize, value: Scalar) {
        self.m[row * 3 + col] = value;
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(&self) -> Matrix3x3 {
        Matrix3x3::new([
            self.m[0], self.m[3], self.m[6],
            self.m[1], self.m[4], self.m[7],
            self.m[2], self.m[5], self.m[8],
        ])
    }

    /// Computes and returns the inverse of this matrix.
    ///
    /// If this matrix is singular, a `NonInvertibleMatrixError` is returned.
    pub fn inverse(&self) -> Result<Matrix3x3, NonInvertibleMatrixError> {
        let det = self.m[0] * self.m[4] * self.m[8] + self.m[1] * self.m[5] * self.m[6] + self.m[2] * self.m[3] * self.m[7]
            - self.m[2] * self.m[4] * self.m[6] - self.m[1] * self.m[3] * self.m[8] - self.m[0] * self.m[5] * self.m[7];

        if det != 0.0 {
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

impl Mul<Scalar> for &Matrix3x3 {
    type Output = Matrix3x3;

    fn mul(self, s: Scalar) -> Matrix3x3 {
        Matrix3x3::new(array![|i| self.m[i] * s; 9])
    }
}

impl Mul<&Matrix3x3> for Scalar {
    type Output = Matrix3x3;

    fn mul(self, m: &Matrix3x3) -> Matrix3x3 {
        m * self
    }
}

impl MulAssign<Scalar> for &mut Matrix3x3 {
    fn mul_assign(&mut self, s: Scalar) {
        for m in &mut self.m { *m *= s; }
    }
}

impl Div<Scalar> for &Matrix3x3 {
    type Output = Matrix3x3;

    fn div(self, s: Scalar) -> Matrix3x3 {
        Matrix3x3::new(array![|i| self.m[i] / s; 9])
    }
}

impl DivAssign<Scalar> for &mut Matrix3x3 {
    fn div_assign(&mut self, s: Scalar) {
        for m in &mut self.m { *m /= s; }
    }
}

impl Mul<Point2> for &Matrix3x3 {
    type Output = Point2;

    fn mul(self, p: Point2) -> Point2 {
        let x = self.m[0] * p.x + self.m[1] * p.y + self.m[2];
        let y = self.m[3] * p.x + self.m[4] * p.y + self.m[5];
        let w = self.m[6] * p.x + self.m[7] * p.y + self.m[8];
        Point2::new(x / w, y / w)
    }
}

impl Mul<&Matrix3x3> for Point2 {
    type Output = Point2;

    fn mul(self, m: &Matrix3x3) -> Point2 {
        let x = self.x * m.m[0] + self.y * m.m[3] + m.m[6];
        let y = self.x * m.m[1] + self.y * m.m[4] + m.m[7];
        let w = self.x * m.m[2] + self.y * m.m[5] + m.m[8];
        Point2::new(x / w, y / w)
    }
}

impl Mul<Vector2> for &Matrix3x3 {
    type Output = Vector2;

    fn mul(self, v: Vector2) -> Vector2 {
        let x = self.m[0] * v.x + self.m[1] * v.y;
        let y = self.m[3] * v.x + self.m[4] * v.y;
        Vector2::new(x, y)
    }
}

impl Mul<&Matrix3x3> for Vector2 {
    type Output = Vector2;

    fn mul(self, m: &Matrix3x3) -> Vector2 {
        let x = self.x * m.m[0] + self.y * m.m[3];
        let y = self.x * m.m[1] + self.y * m.m[4];
        Vector2::new(x, y)
    }
}

impl Mul<&Matrix3x3> for &Matrix3x3 {
    type Output = Matrix3x3;

    fn mul(self, m: &Matrix3x3) -> Matrix3x3 {
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

impl Transform2 {
    /// Creates and returns a new `Transform2` with a transformation matrix and its inverse.
    pub fn new(forward: Arc<Matrix3x3>, inverse: Arc<Matrix3x3>) -> Transform2 {
        Transform2 { forward, inverse }
    }

    /// Creates and returns a transform from a transformation matrix; computes the inverse of the matrix.
    pub fn from_matrix(forward: Matrix3x3) -> Result<Transform2, NonInvertibleMatrixError> {
        let inverse = forward.inverse()?;
        Ok(Transform2::new(Arc::new(forward), Arc::new(inverse)))
    }

    /// Returns a `Transform2` which represents the identity transform.
    pub fn identity() -> Transform2 {
        let forward = Arc::new(Matrix3x3::identity());
        let inverse = forward.clone();
        Transform2::new(forward, inverse)
    }

    /// Returns a translation transform over a vector.
    pub fn translate(v: Vector2) -> Transform2 {
        Transform2::new(Arc::new(Matrix3x3::translate(v)), Arc::new(Matrix3x3::translate(-v)))
    }

    /// Returns a rotation transform which rotates around the origin.
    pub fn rotate(angle: Scalar) -> Transform2 {
        let forward = Matrix3x3::rotate(angle);
        let inverse = forward.transpose();
        Transform2::new(Arc::new(forward), Arc::new(inverse))
    }

    /// Returns a transform which scales by factors in the X and Y dimensions.
    pub fn scale(sx: Scalar, sy: Scalar) -> Transform2 {
        Transform2::new(Arc::new(Matrix3x3::scale(sx, sy)), Arc::new(Matrix3x3::scale(sx.recip(), sy.recip())))
    }

    /// Returns a transform which scales uniformly in all dimensions by a factor.
    pub fn scale_uniform(s: Scalar) -> Transform2 {
        Transform2::new(Arc::new(Matrix3x3::scale_uniform(s)), Arc::new(Matrix3x3::scale_uniform(s.recip())))
    }

    /// Computes and returns a composite transform, which first applies this and then the other transform.
    pub fn and_then(&self, transform: &Transform2) -> Transform2 {
        Transform2::new(Arc::new(&*transform.forward * &*self.forward), Arc::new(&*self.inverse * &*transform.inverse))
    }

    /// Returns the inverse of this transform.
    pub fn inverse(&self) -> Transform2 {
        Transform2::new(self.inverse.clone(), self.forward.clone())
    }
}

// ===== Point3 ================================================================================================================================================

impl Point3 {
    /// Creates and returns a new `Point3` with x, y and z coordinates.
    pub fn new(x: Scalar, y: Scalar, z: Scalar) -> Point3 {
        Point3 { x, y, z }
    }

    /// Returns a `Point3` which represents the origin (x = 0, y = 0 and z = 0).
    pub fn origin() -> Point3 {
        Point3::new(0.0, 0.0, 0.0)
    }

    /// Returns the dimension with the smallest extent of this point.
    pub fn min_dimension(self) -> Dimension3 {
        let Point3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this point.
    pub fn max_dimension(self) -> Dimension3 {
        let Point3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this point.
    pub fn floor(self) -> Point3 {
        Point3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the element-wise ceiling of this point.
    pub fn ceil(self) -> Point3 {
        Point3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the element-wise rounded value of this point.
    pub fn round(self) -> Point3 {
        Point3::new(self.x.round(), self.y.round(), self.z.round())
    }

    /// Returns the element-wise truncated value of this point.
    pub fn trunc(self) -> Point3 {
        Point3::new(self.x.trunc(), self.y.trunc(), self.z.trunc())
    }

    /// Returns the element-wise fractional value of this point.
    pub fn fract(self) -> Point3 {
        Point3::new(self.x.fract(), self.y.fract(), self.z.fract())
    }

    /// Returns the element-wise absolute value of this point.
    pub fn abs(self) -> Point3 {
        Point3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a point with a permutation of the elements of this point.
    pub fn permute(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Point3 {
        Point3::new(self[dim_x], self[dim_y], self[dim_z])
    }
}

impl MinMax for Point3 {
    /// Returns the element-wise minimum of two points.
    fn min(self, p: Point3) -> Point3 {
        Point3::new(min(self.x, p.x), min(self.y, p.y), min(self.z, p.z))
    }

    /// Returns the element-wise maximum of two points.
    fn max(self, p: Point3) -> Point3 {
        Point3::new(max(self.x, p.x), max(self.y, p.y), max(self.z, p.z))
    }
}

impl Distance for Point3 {
    type Output = Scalar;

    /// Computes and returns the distance between two points.
    fn distance(self, p: Point3) -> Scalar {
        (p - self).length()
    }
}

impl RelativeDistance for Point3 {
    /// Checks which of the points `p1` and `p2` is closest to this point and returns the closest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    fn closest(self, p1: Point3, p2: Point3) -> Point3 {
        let (dp1, dp2) = (p1 - self, p2 - self);
        if dot(dp1, dp1) <= dot(dp2, dp2) { p1 } else { p2 }
    }

    /// Checks which of the points `p1` and `p2` is farthest from this point and returns the farthest one.
    ///
    /// This is more computationally efficient than computing the distance between this point and the points `p1` and `p2` and comparing the distances,
    /// because square root operations that are needed for computing the distances are avoided.
    fn farthest(self, p1: Point3, p2: Point3) -> Point3 {
        let (dp1, dp2) = (p1 - self, p2 - self);
        if dot(dp1, dp1) > dot(dp2, dp2) { p1 } else { p2 }
    }
}

impl Index<Dimension3> for Point3 {
    type Output = Scalar;

    fn index(&self, dim: Dimension3) -> &Scalar {
        match dim {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl IndexMut<Dimension3> for Point3 {
    fn index_mut(&mut self, dim: Dimension3) -> &mut Scalar {
        match dim {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl Add<Vector3> for Point3 {
    type Output = Point3;

    fn add(self, v: Vector3) -> Point3 {
        Point3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

impl AddAssign<Vector3> for Point3 {
    fn add_assign(&mut self, v: Vector3) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }
}

impl Sub<Vector3> for Point3 {
    type Output = Point3;

    fn sub(self, v: Vector3) -> Point3 {
        Point3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
}

impl SubAssign<Vector3> for Point3 {
    fn sub_assign(&mut self, v: Vector3) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

impl Sub<Point3> for Point3 {
    type Output = Vector3;

    fn sub(self, p: Point3) -> Vector3 {
        Vector3::new(self.x - p.x, self.y - p.y, self.z - p.z)
    }
}

impl Neg for Point3 {
    type Output = Point3;

    fn neg(self) -> Point3 {
        Point3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<Scalar> for Point3 {
    type Output = Point3;

    fn mul(self, s: Scalar) -> Point3 {
        Point3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Point3> for Scalar {
    type Output = Point3;

    fn mul(self, p: Point3) -> Point3 {
        p * self
    }
}

impl MulAssign<Scalar> for Point3 {
    fn mul_assign(&mut self, s: Scalar) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl Div<Scalar> for Point3 {
    type Output = Point3;

    fn div(self, s: Scalar) -> Point3 {
        Point3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl DivAssign<Scalar> for Point3 {
    fn div_assign(&mut self, s: Scalar) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl Transform<Point3> for Transform3 {
    type Output = Point3;

    /// Transforms a point.
    fn transform(&self, p: Point3) -> Point3 {
        &*self.forward * p
    }
}

impl From<Vector3> for Point3 {
    fn from(v: Vector3) -> Point3 {
        Point3::new(v.x, v.y, v.z)
    }
}

// ===== Vector3 ===============================================================================================================================================

impl Vector3 {
    /// Creates and returns a new `Vector3` with x, y and z coordinates.
    pub fn new(x: Scalar, y: Scalar, z: Scalar) -> Vector3 {
        Vector3 { x, y, z }
    }

    /// Returns a `Vector3` which represents the zero vector (x = 0, y = 0 and z = 0).
    pub fn zero() -> Vector3 {
        Vector3::new(0.0, 0.0, 0.0)
    }

    /// Returns a `Vector3` of length 1 which represents the X axis (x = 1, y = 0 and z = 0).
    pub fn x_axis() -> Vector3 {
        Vector3::new(1.0, 0.0, 0.0)
    }

    /// Returns a `Vector3` of length 1 which represents the Y axis (x = 0, y = 1 and z = 0).
    pub fn y_axis() -> Vector3 {
        Vector3::new(0.0, 1.0, 0.0)
    }

    /// Returns a `Vector3` of length 1 which represents the Z axis (x = 0, y = 0 and z = 1).
    pub fn z_axis() -> Vector3 {
        Vector3::new(0.0, 0.0, 1.0)
    }

    /// Returns a `Vector3` of length 1 which represents the axis specified by a dimension.
    pub fn axis(dim: Dimension3) -> Vector3 {
        match dim {
            Dimension3::X => Vector3::x_axis(),
            Dimension3::Y => Vector3::y_axis(),
            Dimension3::Z => Vector3::z_axis(),
        }
    }

    /// Creates and returns a new `Vector3` which points in the same direction as this vector, but with length 1.
    pub fn normalize(self) -> Vector3 {
        self / self.length()
    }

    /// Returns the dimension with the smallest extent of this vector.
    pub fn min_dimension(self) -> Dimension3 {
        let Vector3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this vector.
    pub fn max_dimension(self) -> Dimension3 {
        let Vector3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this vector.
    pub fn floor(self) -> Vector3 {
        Vector3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the element-wise ceiling of this vector.
    pub fn ceil(self) -> Vector3 {
        Vector3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the element-wise rounded value of this vector.
    pub fn round(self) -> Vector3 {
        Vector3::new(self.x.round(), self.y.round(), self.z.round())
    }

    /// Returns the element-wise truncated value of this vector.
    pub fn trunc(self) -> Vector3 {
        Vector3::new(self.x.trunc(), self.y.trunc(), self.z.trunc())
    }

    /// Returns the element-wise fractional value of this vector.
    pub fn fract(self) -> Vector3 {
        Vector3::new(self.x.fract(), self.y.fract(), self.z.fract())
    }

    /// Returns the element-wise absolute value of this vector.
    pub fn abs(self) -> Vector3 {
        Vector3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a point with a permutation of the elements of this vector.
    pub fn permute(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Vector3 {
        Vector3::new(self[dim_x], self[dim_y], self[dim_z])
    }
}

impl MinMax for Vector3 {
    /// Returns the element-wise minimum of two vectors.
    fn min(self, v: Vector3) -> Vector3 {
        Vector3::new(min(self.x, v.x), min(self.y, v.y), min(self.z, v.z))
    }

    /// Returns the element-wise maximum of two vectors.
    fn max(self, v: Vector3) -> Vector3 {
        Vector3::new(max(self.x, v.x), max(self.y, v.y), max(self.z, v.z))
    }
}

impl Length for Vector3 {
    type Output = Scalar;

    /// Computes and returns the length of a vector.
    fn length(self) -> Scalar {
        Scalar::sqrt(dot(self, self))
    }
}

impl RelativeLength for Vector3 {
    /// Returns the shortest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    fn shortest(self, v: Vector3) -> Vector3 {
        if dot(self, self) <= dot(v, v) { self } else { v }
    }

    /// Returns the longest of two vectors.
    ///
    /// This is more computationally efficient than computing the lengths of the vectors and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    fn longest(self, v: Vector3) -> Vector3 {
        if dot(self, self) > dot(v, v) { self } else { v }
    }
}

impl DotProduct<Vector3> for Vector3 {
    type Output = Scalar;

    /// Computes and returns the dot product between two vectors.
    fn dot(self, v: Vector3) -> Scalar {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}

impl DotProduct<Normal3> for Vector3 {
    type Output = Scalar;

    /// Computes and returns the dot product between this vector and a normal.
    fn dot(self, n: Normal3) -> Scalar {
        self.x * n.x + self.y * n.y + self.z * n.z
    }
}

impl CrossProduct<Vector3> for Vector3 {
    type Output = Vector3;

    /// Computes and returns the cross product between two vectors.
    fn cross(self, v: Vector3) -> Vector3 {
        Vector3::new(self.y * v.z - self.z * v.y, self.z * v.x - self.x * v.z, self.x * v.y - self.y * v.x)
    }
}

impl Index<Dimension3> for Vector3 {
    type Output = Scalar;

    fn index(&self, dim: Dimension3) -> &Scalar {
        match dim {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl IndexMut<Dimension3> for Vector3 {
    fn index_mut(&mut self, dim: Dimension3) -> &mut Scalar {
        match dim {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl Add<Vector3> for Vector3 {
    type Output = Vector3;

    fn add(self, v: Vector3) -> Vector3 {
        Vector3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

impl AddAssign<Vector3> for Vector3 {
    fn add_assign(&mut self, v: Vector3) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }
}

impl Sub<Vector3> for Vector3 {
    type Output = Vector3;

    fn sub(self, v: Vector3) -> Vector3 {
        Vector3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
}

impl SubAssign<Vector3> for Vector3 {
    fn sub_assign(&mut self, v: Vector3) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

impl Neg for Vector3 {
    type Output = Vector3;

    fn neg(self) -> Vector3 {
        Vector3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<Scalar> for Vector3 {
    type Output = Vector3;

    fn mul(self, s: Scalar) -> Vector3 {
        Vector3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Vector3> for Scalar {
    type Output = Vector3;

    fn mul(self, v: Vector3) -> Vector3 {
        v * self
    }
}

impl MulAssign<Scalar> for Vector3 {
    fn mul_assign(&mut self, s: Scalar) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl Div<Scalar> for Vector3 {
    type Output = Vector3;

    fn div(self, s: Scalar) -> Vector3 {
        Vector3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl DivAssign<Scalar> for Vector3 {
    fn div_assign(&mut self, s: Scalar) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl Transform<Vector3> for Transform3 {
    type Output = Vector3;

    /// Transforms a vector.
    fn transform(&self, v: Vector3) -> Vector3 {
        &*self.forward * v
    }
}

impl From<Point3> for Vector3 {
    fn from(p: Point3) -> Vector3 {
        Vector3::new(p.x, p.y, p.z)
    }
}

impl From<Normal3> for Vector3 {
    fn from(n: Normal3) -> Self {
        Vector3::new(n.x, n.y, n.z)
    }
}

// ===== Normal3 ===============================================================================================================================================

impl Normal3 {
    /// Creates and returns a new `Normal3` with x, y and z coordinates.
    pub fn new(x: Scalar, y: Scalar, z: Scalar) -> Normal3 {
        Normal3 { x, y, z }
    }

    /// Returns a `Normal3` which represents the zero normal (x = 0, y = 0 and z = 0).
    pub fn zero() -> Normal3 {
        Normal3::new(0.0, 0.0, 0.0)
    }

    /// Returns a `Normal3` of length 1 which represents the X axis (x = 1, y = 0 and z = 0).
    pub fn x_axis() -> Normal3 {
        Normal3::new(1.0, 0.0, 0.0)
    }

    /// Returns a `Normal3` of length 1 which represents the Y axis (x = 0, y = 1 and z = 0).
    pub fn y_axis() -> Normal3 {
        Normal3::new(0.0, 1.0, 0.0)
    }

    /// Returns a `Normal3` of length 1 which represents the Z axis (x = 0, y = 0 and z = 1).
    pub fn z_axis() -> Normal3 {
        Normal3::new(0.0, 0.0, 1.0)
    }

    /// Returns a `Normal3` of length 1 which represents the axis specified by a dimension.
    pub fn axis(dim: Dimension3) -> Normal3 {
        match dim {
            Dimension3::X => Normal3::x_axis(),
            Dimension3::Y => Normal3::y_axis(),
            Dimension3::Z => Normal3::z_axis(),
        }
    }

    /// Creates and returns a new `Normal3` which points in the same direction as this normal, but with length 1.
    pub fn normalize(self) -> Normal3 {
        self / self.length()
    }

    /// Returns the dimension with the smallest extent of this normal.
    pub fn min_dimension(self) -> Dimension3 {
        let Normal3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this normal.
    pub fn max_dimension(self) -> Dimension3 {
        let Normal3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this normal.
    pub fn floor(self) -> Normal3 {
        Normal3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the element-wise ceiling of this normal.
    pub fn ceil(self) -> Normal3 {
        Normal3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the element-wise rounded value of this normal.
    pub fn round(self) -> Normal3 {
        Normal3::new(self.x.round(), self.y.round(), self.z.round())
    }

    /// Returns the element-wise truncated value of this normal.
    pub fn trunc(self) -> Normal3 {
        Normal3::new(self.x.trunc(), self.y.trunc(), self.z.trunc())
    }

    /// Returns the element-wise fractional value of this normal.
    pub fn fract(self) -> Normal3 {
        Normal3::new(self.x.fract(), self.y.fract(), self.z.fract())
    }

    /// Returns the element-wise absolute value of this normal.
    pub fn abs(self) -> Normal3 {
        Normal3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a point with a permutation of the elements of this normal.
    pub fn permute(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Normal3 {
        Normal3::new(self[dim_x], self[dim_y], self[dim_z])
    }
}

impl MinMax for Normal3 {
    /// Returns the element-wise minimum of two normals.
    fn min(self, n: Normal3) -> Normal3 {
        Normal3::new(min(self.x, n.x), min(self.y, n.y), min(self.z, n.z))
    }

    /// Returns the element-wise maximum of two normals.
    fn max(self, n: Normal3) -> Normal3 {
        Normal3::new(max(self.x, n.x), max(self.y, n.y), max(self.z, n.z))
    }
}

impl Length for Normal3 {
    type Output = Scalar;

    /// Computes and returns the length of a normal.
    fn length(self) -> Scalar {
        Scalar::sqrt(dot(self, self))
    }
}

impl RelativeLength for Normal3 {
    /// Returns the shortest of two normals.
    ///
    /// This is more computationally efficient than computing the lengths of the normals and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    fn shortest(self, n: Normal3) -> Normal3 {
        if dot(self, self) <= dot(n, n) { self } else { n }
    }

    /// Returns the longest of two normals.
    ///
    /// This is more computationally efficient than computing the lengths of the normals and comparing them,
    /// because square root operations that are needed for computing the lengths are avoided.
    fn longest(self, n: Normal3) -> Normal3 {
        if dot(self, self) > dot(n, n) { self } else { n }
    }
}

impl DotProduct<Normal3> for Normal3 {
    type Output = Scalar;

    /// Computes and returns the dot product between two normals.
    fn dot(self, n: Normal3) -> Scalar {
        self.x * n.x + self.y * n.y + self.z * n.z
    }
}

impl DotProduct<Vector3> for Normal3 {
    type Output = Scalar;

    /// Computes and returns the dot product between this normal and a vector.
    fn dot(self, v: Vector3) -> Scalar {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}

impl Index<Dimension3> for Normal3 {
    type Output = Scalar;

    fn index(&self, dim: Dimension3) -> &Scalar {
        match dim {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl IndexMut<Dimension3> for Normal3 {
    fn index_mut(&mut self, dim: Dimension3) -> &mut Scalar {
        match dim {
            Dimension3::X => &mut self.x,
            Dimension3::Y => &mut self.y,
            Dimension3::Z => &mut self.z,
        }
    }
}

impl Add<Normal3> for Normal3 {
    type Output = Normal3;

    fn add(self, n: Normal3) -> Normal3 {
        Normal3::new(self.x + n.x, self.y + n.y, self.z + n.z)
    }
}

impl AddAssign<Normal3> for Normal3 {
    fn add_assign(&mut self, n: Normal3) {
        self.x += n.x;
        self.y += n.y;
        self.z += n.z;
    }
}

impl Sub<Normal3> for Normal3 {
    type Output = Normal3;

    fn sub(self, n: Normal3) -> Normal3 {
        Normal3::new(self.x - n.x, self.y - n.y, self.z - n.z)
    }
}

impl SubAssign<Normal3> for Normal3 {
    fn sub_assign(&mut self, n: Normal3) {
        self.x -= n.x;
        self.y -= n.y;
        self.z -= n.z;
    }
}

impl Neg for Normal3 {
    type Output = Normal3;

    fn neg(self) -> Normal3 {
        Normal3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<Scalar> for Normal3 {
    type Output = Normal3;

    fn mul(self, s: Scalar) -> Normal3 {
        Normal3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Normal3> for Scalar {
    type Output = Normal3;

    fn mul(self, n: Normal3) -> Normal3 {
        n * self
    }
}

impl MulAssign<Scalar> for Normal3 {
    fn mul_assign(&mut self, s: Scalar) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl Div<Scalar> for Normal3 {
    type Output = Normal3;

    fn div(self, s: Scalar) -> Normal3 {
        Normal3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl DivAssign<Scalar> for Normal3 {
    fn div_assign(&mut self, s: Scalar) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl Transform<Normal3> for Transform3 {
    type Output = Normal3;

    /// Transforms a normal.
    ///
    /// Note that transforming a normal is different from transforming a vector; normals are transformed by applying the transpose of the inverse
    /// transformation matrix. This difference is the main reason why there is a separate type for normals, which should be used instead of `Vector3`.
    fn transform(&self, n: Normal3) -> Normal3 {
        // Normals are transformed by the transpose of the inverse
        Normal3::from(Vector3::from(n) * &*self.inverse)
    }
}

impl From<Vector3> for Normal3 {
    fn from(v: Vector3) -> Self {
        Normal3::new(v.x, v.y, v.z)
    }
}

// ===== Ray3 ==================================================================================================================================================

impl Ray3 {
    /// Creates and returns a new `Ray3` with an origin point and direction vector.
    pub fn new(origin: Point3, direction: Vector3) -> Ray3 {
        Ray3 { origin, direction }
    }

    /// Computes and returns a point at a distance along this ray.
    pub fn at(&self, distance: Scalar) -> Point3 {
        self.origin + self.direction * distance
    }
}

impl Transform<&Ray3> for Transform3 {
    type Output = Ray3;

    /// Transforms a ray.
    fn transform(&self, ray: &Ray3) -> Ray3 {
        Ray3::new(self.transform(ray.origin), self.transform(ray.direction))
    }
}

// ===== BoundingBox3 ==========================================================================================================================================

impl BoundingBox3 {
    /// Creates and returns a new `BoundingBox3` with minimum and maximum corner points.
    pub fn new(min: Point3, max: Point3) -> BoundingBox3 {
        BoundingBox3 { min, max }
    }

    /// Returns an empty `BoundingBox3`.
    pub fn empty() -> BoundingBox3 {
        BoundingBox3::new(
            Point3::new(Scalar::INFINITY, Scalar::INFINITY, Scalar::INFINITY),
            Point3::new(Scalar::NEG_INFINITY, Scalar::NEG_INFINITY, Scalar::NEG_INFINITY),
        )
    }

    /// Returns an infinite `BoundingBox3` which contains all of 3D space.
    pub fn infinite() -> BoundingBox3 {
        BoundingBox3::new(
            Point3::new(Scalar::NEG_INFINITY, Scalar::NEG_INFINITY, Scalar::NEG_INFINITY),
            Point3::new(Scalar::INFINITY, Scalar::INFINITY, Scalar::INFINITY),
        )
    }

    /// Returns the width (extent in the X dimension) of this bounding box.
    pub fn width(&self) -> Scalar {
        self.max.x - self.min.x
    }

    /// Returns the height (extent in the Y dimension) of this bounding box.
    pub fn height(&self) -> Scalar {
        self.max.y - self.min.y
    }

    /// Returns the depth (extent in the Z dimension) of this bounding box.
    pub fn depth(&self) -> Scalar {
        self.max.z - self.min.z
    }

    /// Returns the extent of this bounding box in a dimension.
    pub fn extent(&self, dim: Dimension3) -> Scalar {
        match dim {
            Dimension3::X => self.width(),
            Dimension3::Y => self.height(),
            Dimension3::Z => self.depth(),
        }
    }

    /// Returns the dimension with the smallest extent of this bounding box.
    pub fn min_dimension(&self) -> Dimension3 {
        let d = self.diagonal();
        if d.x <= d.y && d.x <= d.z { Dimension3::X } else if d.y <= d.z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the largest extent of this bounding box.
    pub fn max_dimension(&self) -> Dimension3 {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z { Dimension3::X } else if d.y > d.z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the surface area of this bounding box.
    pub fn surface_area(&self) -> Scalar {
        let d = self.diagonal();
        2.0 * (d.x * d.y + d.x * d.z + d.y * d.z)
    }

    /// Returns the volume (width times height times depth) of this bounding box.
    pub fn volume(&self) -> Scalar {
        let d = self.diagonal();
        d.x * d.y * d.z
    }

    /// Returns the center point of this bounding box.
    pub fn center(&self) -> Point3 {
        self.min + self.diagonal() * 0.5
    }

    /// Returns a corner point of this bounding box, indicated by an index (which must be between 0 and 7 inclusive).
    pub fn corner(&self, index: usize) -> Point3 {
        debug_assert!(index < 8, "Invalid corner index: {}", index);
        let x = if index & 0b001 == 0 { self.min.x } else { self.max.x };
        let y = if index & 0b010 == 0 { self.min.y } else { self.max.y };
        let z = if index & 0b100 == 0 { self.min.z } else { self.max.z };
        Point3::new(x, y, z)
    }

    /// Returns the diagonal of this bounding box as a vector.
    pub fn diagonal(&self) -> Vector3 {
        self.max - self.min
    }

    /// Checks if two bounding boxes overlap.
    pub fn overlaps(&self, bb: &BoundingBox3) -> bool {
        //@formatter:off
        self.max.x >= bb.min.x && self.min.x <= bb.max.x &&
            self.max.y >= bb.min.y && self.min.y <= bb.max.y &&
            self.max.z >= bb.min.z && self.min.z <= bb.max.z
        //@formatter:on
    }

    /// Checks if a point is inside this bounding box.
    pub fn is_inside(&self, p: Point3) -> bool {
        //@formatter:off
        p.x >= self.min.x && p.x <= self.max.x &&
            p.y >= self.min.y && p.y <= self.max.y &&
            p.z >= self.min.z && p.z <= self.max.z
        //@formatter:on
    }

    /// Computes the closest intersection of this bounding box with a ray within a range.
    ///
    /// Returns a `Some` containing the closest intersection, or `None` if the ray does not intersect the bounding box within the range.
    pub fn intersect_ray(&self, ray: &Ray3, range: &Range<Scalar>) -> Option<Scalar> {
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

impl Union<&BoundingBox3> for BoundingBox3 {
    type Output = BoundingBox3;

    /// Computes and returns the union between two bounding boxes.
    ///
    /// The union is the smallest bounding box that contains both bounding boxes.
    fn union(self, bb: &BoundingBox3) -> BoundingBox3 {
        BoundingBox3::new(min(self.min, bb.min), max(self.max, bb.max))
    }
}

impl Union<Point3> for BoundingBox3 {
    type Output = BoundingBox3;

    /// Computes and returns the union between this bounding box and a point.
    ///
    /// The union is the smallest bounding box that contains both the bounding box and the point.
    fn union(self, p: Point3) -> BoundingBox3 {
        BoundingBox3::new(min(self.min, p), max(self.max, p))
    }
}

impl Intersection<&BoundingBox3> for BoundingBox3 {
    type Output = BoundingBox3;

    /// Computes and returns the intersection between two bounding boxes.
    ///
    /// The intersection is the largest bounding box that contains the region where the two bounding boxes overlap.
    ///
    /// Returns `Some` when the bounding boxes overlap; `None` if the bounding boxes do not overlap.
    fn intersection(self, bb: &BoundingBox3) -> Option<BoundingBox3> {
        if self.overlaps(bb) {
            Some(BoundingBox3::new(max(self.min, bb.min), min(self.max, bb.max)))
        } else {
            None
        }
    }
}

impl Transform<&BoundingBox3> for Transform3 {
    type Output = BoundingBox3;

    /// Transforms a bounding box.
    fn transform(&self, bb: &BoundingBox3) -> BoundingBox3 {
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

impl Matrix4x4 {
    /// Creates and returns a new `Matrix4x4` with the specified elements.
    pub fn new(m: [Scalar; 16]) -> Matrix4x4 {
        Matrix4x4 { m }
    }

    /// Returns a `Matrix4x4` which represents the identity matrix.
    pub fn identity() -> Matrix4x4 {
        Matrix4x4::new([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a translation matrix which translates over a vector.
    pub fn translate(v: Vector3) -> Matrix4x4 {
        Matrix4x4::new([
            1.0, 0.0, 0.0, v.x,
            0.0, 1.0, 0.0, v.y,
            0.0, 0.0, 1.0, v.z,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a rotation matrix which rotates around the X axis.
    pub fn rotate_x(angle: Scalar) -> Matrix4x4 {
        let (sin, cos) = angle.sin_cos();

        Matrix4x4::new([
            1.0, 0.0, 0.0, 0.0,
            0.0, cos, -sin, 0.0,
            0.0, sin, cos, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a rotation matrix which rotates around the Y axis.
    pub fn rotate_y(angle: Scalar) -> Matrix4x4 {
        let (sin, cos) = angle.sin_cos();

        Matrix4x4::new([
            cos, 0.0, sin, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sin, 0.0, cos, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a rotation matrix which rotates around the Y axis.
    pub fn rotate_z(angle: Scalar) -> Matrix4x4 {
        let (sin, cos) = angle.sin_cos();

        Matrix4x4::new([
            cos, -sin, 0.0, 0.0,
            sin, cos, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a rotation matrix which rotates around an axis.
    pub fn rotate_axis(axis: Vector3, angle: Scalar) -> Matrix4x4 {
        let a = axis.normalize();

        let (s, c) = angle.sin_cos();
        let cc = 1.0 - c;

        let (t1, t2, t3) = (a.x * a.y * cc, a.x * a.z * cc, a.y * a.z * cc);
        let (u1, u2, u3) = (a.x * s, a.y * s, a.z * s);

        Matrix4x4::new([
            a.x * a.x * cc + c, t1 - u3, t2 + u2, 0.0,
            t1 + u3, a.y * a.y * cc + c, t3 - u1, 0.0,
            t2 - u2, t3 + u1, a.z * a.z * cc + c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a matrix which scales by factors in the X, Y and Z dimensions.
    pub fn scale(sx: Scalar, sy: Scalar, sz: Scalar) -> Matrix4x4 {
        Matrix4x4::new([
            sx, 0.0, 0.0, 0.0,
            0.0, sy, 0.0, 0.0,
            0.0, 0.0, sz, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a matrix which scales uniformly in all dimensions by a factor.
    pub fn scale_uniform(s: Scalar) -> Matrix4x4 {
        Matrix4x4::new([
            s, 0.0, 0.0, 0.0,
            0.0, s, 0.0, 0.0,
            0.0, 0.0, s, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns the inverse of a look-at transformation matrix which looks from a point at a target, with an 'up' direction.
    pub fn inverse_look_at(from: Point3, target: Point3, up: Vector3) -> Matrix4x4 {
        let direction = (target - from).normalize();
        let right = up.normalize().cross(direction).normalize();
        let new_up = direction.cross(right);

        Matrix4x4::new([
            right.x, new_up.x, direction.x, from.x,
            right.y, new_up.y, direction.y, from.y,
            right.z, new_up.z, direction.z, from.z,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns an element at a row and column of the matrix.
    pub fn get(&self, row: usize, col: usize) -> Scalar {
        self.m[row * 4 + col]
    }

    /// Returns a mutable reference to an element at a row and column of the matrix.
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut Scalar {
        &mut self.m[row * 4 + col]
    }

    /// Sets the value of an element at a row and column of the matrix.
    pub fn set(&mut self, row: usize, col: usize, value: Scalar) {
        self.m[row * 4 + col] = value;
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(&self) -> Matrix4x4 {
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
    pub fn inverse(&self) -> Result<Matrix4x4, NonInvertibleMatrixError> {
        let cofactor = |i, j| {
            let sub = |row, col| self.get(if row < i { row } else { row + 1 }, if col < j { col } else { col + 1 });

            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };

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

        if det != 0.0 {
            Ok(&adjugate * det.recip())
        } else {
            Err(NonInvertibleMatrixError)
        }
    }
}

impl Mul<Scalar> for &Matrix4x4 {
    type Output = Matrix4x4;

    fn mul(self, s: Scalar) -> Matrix4x4 {
        Matrix4x4::new(array![|i| self.m[i] * s; 16])
    }
}

impl Mul<&Matrix4x4> for Scalar {
    type Output = Matrix4x4;

    fn mul(self, m: &Matrix4x4) -> Matrix4x4 {
        m * self
    }
}

impl MulAssign<Scalar> for &mut Matrix4x4 {
    fn mul_assign(&mut self, s: Scalar) {
        for m in &mut self.m { *m *= s; }
    }
}

impl Div<Scalar> for &Matrix4x4 {
    type Output = Matrix4x4;

    fn div(self, s: Scalar) -> Matrix4x4 {
        Matrix4x4::new(array![|i| self.m[i] / s; 16])
    }
}

impl DivAssign<Scalar> for &mut Matrix4x4 {
    fn div_assign(&mut self, s: Scalar) {
        for m in &mut self.m { *m /= s; }
    }
}

impl Mul<Point3> for &Matrix4x4 {
    type Output = Point3;

    fn mul(self, p: Point3) -> Point3 {
        let x = self.m[0] * p.x + self.m[1] * p.y + self.m[2] * p.z + self.m[3];
        let y = self.m[4] * p.x + self.m[5] * p.y + self.m[6] * p.z + self.m[7];
        let z = self.m[8] * p.x + self.m[9] * p.y + self.m[10] * p.z + self.m[11];
        let w = self.m[12] * p.x + self.m[13] * p.y + self.m[14] * p.z + self.m[15];
        Point3::new(x / w, y / w, z / w)
    }
}

impl Mul<&Matrix4x4> for Point3 {
    type Output = Point3;

    fn mul(self, m: &Matrix4x4) -> Point3 {
        let x = self.x * m.m[0] + self.y * m.m[4] + self.z * m.m[8] + m.m[12];
        let y = self.x * m.m[1] + self.y * m.m[5] + self.z * m.m[9] + m.m[13];
        let z = self.x * m.m[2] + self.y * m.m[6] + self.z * m.m[10] + m.m[14];
        let w = self.x * m.m[3] + self.y * m.m[7] + self.z * m.m[11] + m.m[15];
        Point3::new(x / w, y / w, z / w)
    }
}

impl Mul<Vector3> for &Matrix4x4 {
    type Output = Vector3;

    fn mul(self, v: Vector3) -> Vector3 {
        let x = self.m[0] * v.x + self.m[1] * v.y + self.m[2] * v.z;
        let y = self.m[4] * v.x + self.m[5] * v.y + self.m[6] * v.z;
        let z = self.m[8] * v.x + self.m[9] * v.y + self.m[10] * v.z;
        Vector3::new(x, y, z)
    }
}

impl Mul<&Matrix4x4> for Vector3 {
    type Output = Vector3;

    fn mul(self, m: &Matrix4x4) -> Vector3 {
        let x = self.x * m.m[0] + self.y * m.m[4] + self.z * m.m[8];
        let y = self.x * m.m[1] + self.y * m.m[5] + self.z * m.m[9];
        let z = self.x * m.m[2] + self.y * m.m[6] + self.z * m.m[10];
        Vector3::new(x, y, z)
    }
}

impl Mul<&Matrix4x4> for &Matrix4x4 {
    type Output = Matrix4x4;

    fn mul(self, m: &Matrix4x4) -> Matrix4x4 {
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

impl Transform3 {
    /// Creates and returns a new `Transform3` with a transformation matrix and its inverse.
    pub fn new(forward: Arc<Matrix4x4>, inverse: Arc<Matrix4x4>) -> Transform3 {
        Transform3 { forward, inverse }
    }

    /// Creates and returns a transform from a transformation matrix; computes the inverse of the matrix.
    pub fn from_matrix(forward: Matrix4x4) -> Result<Transform3, NonInvertibleMatrixError> {
        let inverse = forward.inverse()?;
        Ok(Transform3::new(Arc::new(forward), Arc::new(inverse)))
    }

    /// Returns a `Transform3` which represents the identity transform.
    pub fn identity() -> Transform3 {
        let forward = Arc::new(Matrix4x4::identity());
        let inverse = forward.clone();
        Transform3::new(forward, inverse)
    }

    /// Returns a translation transform over a vector.
    pub fn translate(v: Vector3) -> Transform3 {
        Transform3::new(Arc::new(Matrix4x4::translate(v)), Arc::new(Matrix4x4::translate(-v)))
    }

    /// Returns a rotation transform which rotates around the X axis.
    pub fn rotate_x(angle: Scalar) -> Transform3 {
        let forward = Matrix4x4::rotate_x(angle);
        let inverse = forward.transpose();
        Transform3::new(Arc::new(forward), Arc::new(inverse))
    }

    /// Returns a rotation transform which rotates around the Y axis.
    pub fn rotate_y(angle: Scalar) -> Transform3 {
        let forward = Matrix4x4::rotate_y(angle);
        let inverse = forward.transpose();
        Transform3::new(Arc::new(forward), Arc::new(inverse))
    }

    /// Returns a rotation transform which rotates around the Z axis.
    pub fn rotate_z(angle: Scalar) -> Transform3 {
        let forward = Matrix4x4::rotate_z(angle);
        let inverse = forward.transpose();
        Transform3::new(Arc::new(forward), Arc::new(inverse))
    }

    /// Returns a rotation transform which rotates around an axis.
    pub fn rotate_axis(axis: Vector3, angle: Scalar) -> Transform3 {
        let forward = Matrix4x4::rotate_axis(axis, angle);
        let inverse = forward.transpose();
        Transform3::new(Arc::new(forward), Arc::new(inverse))
    }

    /// Returns a transform which scales by factors in the X, Y and Z dimensions.
    pub fn scale(sx: Scalar, sy: Scalar, sz: Scalar) -> Transform3 {
        Transform3::new(Arc::new(Matrix4x4::scale(sx, sy, sz)), Arc::new(Matrix4x4::scale(sx.recip(), sy.recip(), sz.recip())))
    }

    /// Returns a transform which scales uniformly in all dimensions by a factor.
    pub fn scale_uniform(s: Scalar) -> Transform3 {
        Transform3::new(Arc::new(Matrix4x4::scale_uniform(s)), Arc::new(Matrix4x4::scale_uniform(s.recip())))
    }

    // TODO: factory methods for perspective and orthographic projection matrices and transforms

    /// Returns a look-at transform which looks from a point at a target, with an 'up' direction.
    pub fn look_at(from: Point3, target: Point3, up: Vector3) -> Result<Transform3, NonInvertibleMatrixError> {
        let inverse = Matrix4x4::inverse_look_at(from, target, up);
        let forward = inverse.inverse()?;
        Ok(Transform3::new(Arc::new(forward), Arc::new(inverse)))
    }

    /// Computes and returns a composite transform, which first applies this and then the other transform.
    pub fn and_then(&self, transform: &Transform3) -> Transform3 {
        Transform3::new(Arc::new(&*transform.forward * &*self.forward), Arc::new(&*self.inverse * &*transform.inverse))
    }

    /// Returns the inverse of this transform.
    pub fn inverse(&self) -> Transform3 {
        Transform3::new(self.inverse.clone(), self.forward.clone())
    }
}

// ===== Unit tests ============================================================================================================================================

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn todo() {
        // TODO: Add unit tests
    }
}

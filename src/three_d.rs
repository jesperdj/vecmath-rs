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

use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::Arc;

use array_macro::array;

use crate::NonInvertibleMatrixError;

/// Dimension in 3D space.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Dimension3 { X, Y, Z }

/// Point in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Vector in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Normal vector in 3D space.
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct Normal3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// TODO: Quaternion

/// Transformation matrix for transformations in 3D space.
#[derive(Clone, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct Matrix4x4 {
    m: [f32; 16]
}

/// Transformation in 3D space; holds shared pointers to a transformation matrix and its inverse.
#[derive(Clone, PartialEq, Debug)]
pub struct Transform3 {
    pub forward: Arc<Matrix4x4>,
    pub inverse: Arc<Matrix4x4>,
}

/// Implement this trait for `Transform3` with a specific type to be able to transform values of that type with a `Transform3`.
pub trait ApplyTransform3<T> {
    type Output;

    fn transform(&self, input: T) -> Self::Output;
}

// ===== Point3 ================================================================================================================================================

impl Point3 {
    /// Creates a new `Point3` with x, y and z coordinate values.
    pub const fn new(x: f32, y: f32, z: f32) -> Point3 {
        Point3 { x, y, z }
    }

    /// Returns a `Point3` which represents the origin: x = 0, y = 0 and z = 0.
    pub const fn origin() -> Point3 {
        Point3::new(0.0, 0.0, 0.0)
    }

    /// Computes the distance between two points.
    ///
    /// If you only need to compare relative distances, consider using [`distance_squared`](#method.distance_squared) instead, which avoids a costly
    /// square root computation.
    pub fn distance(self, p: Point3) -> f32 {
        (self - p).length()
    }

    /// Computes the square of the distance between two points.
    ///
    /// Use this method instead of [`distance`](#method.distance) when possible, to avoid a costly square root computation.
    pub fn distance_squared(self, p: Point3) -> f32 {
        (self - p).length_squared()
    }

    /// Returns the element-wise minimum between this point and the point `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Point3;
    ///
    /// let p1 = Point3::new(2.0, 1.0, 3.0);
    /// let p2 = Point3::new(4.0, -1.5, 0.5);
    /// assert_eq!(p1.min(p2), Point3::new(2.0, -1.5, 0.5));
    /// ```
    pub fn min(self, p: Point3) -> Point3 {
        Point3::new(f32::min(self.x, p.x), f32::min(self.y, p.y), f32::min(self.z, p.z))
    }

    /// Returns the element-wise maximum between this point and the point `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Point3;
    ///
    /// let p1 = Point3::new(2.0, 1.0, 3.0);
    /// let p2 = Point3::new(4.0, -1.5, 0.5);
    /// assert_eq!(p1.max(p2), Point3::new(4.0, 1.0, 3.0));
    /// ```
    pub fn max(self, p: Point3) -> Point3 {
        Point3::new(f32::max(self.x, p.x), f32::max(self.y, p.y), f32::max(self.z, p.z))
    }

    /// Returns the dimension with the minimum extent of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension3, Point3};
    ///
    /// let p = Point3::new(2.0, -3.0, 1.0);
    /// assert_eq!(p.min_dimension(), Dimension3::Z);
    /// ```
    pub fn min_dimension(self) -> Dimension3 {
        let Point3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the maximum extent of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension3, Point3};
    ///
    /// let p = Point3::new(2.0, -3.0, 1.0);
    /// assert_eq!(p.max_dimension(), Dimension3::Y);
    /// ```
    pub fn max_dimension(self) -> Dimension3 {
        let Point3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Point3;
    ///
    /// let p = Point3::new(1.5, -2.25, 0.75);
    /// assert_eq!(p.floor(), Point3::new(1.0, -3.0, 0.0));
    /// ```
    pub fn floor(self) -> Point3 {
        Point3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the element-wise ceiling of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Point3;
    ///
    /// let p = Point3::new(1.5, -2.25, 0.75);
    /// assert_eq!(p.ceil(), Point3::new(2.0, -2.0, 1.0));
    /// ```
    pub fn ceil(self) -> Point3 {
        Point3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the element-wise absolute value of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Point3;
    ///
    /// let p = Point3::new(3.5, -2.0, -0.5);
    /// assert_eq!(p.abs(), Point3::new(3.5, 2.0, 0.5));
    /// ```
    pub fn abs(self) -> Point3 {
        Point3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a point with a permutation of the elements of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension3, Point3};
    ///
    /// let p = Point3::new(1.0, 2.0, 3.0);
    /// assert_eq!(p.permute(Dimension3::Y, Dimension3::Z, Dimension3::X), Point3::new(2.0, 3.0, 1.0));
    /// ```
    pub fn permute(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Point3 {
        Point3::new(self[dim_x], self[dim_y], self[dim_z])
    }
}

impl Index<Dimension3> for Point3 {
    type Output = f32;

    fn index(&self, index: Dimension3) -> &f32 {
        match index {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl IndexMut<Dimension3> for Point3 {
    fn index_mut(&mut self, index: Dimension3) -> &mut f32 {
        match index {
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

impl Mul<f32> for Point3 {
    type Output = Point3;

    fn mul(self, s: f32) -> Point3 {
        Point3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Point3> for f32 {
    type Output = Point3;

    fn mul(self, p: Point3) -> Point3 {
        p * self
    }
}

impl MulAssign<f32> for Point3 {
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl Div<f32> for Point3 {
    type Output = Point3;

    fn div(self, s: f32) -> Point3 {
        Point3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl DivAssign<f32> for Point3 {
    fn div_assign(&mut self, s: f32) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl From<Vector3> for Point3 {
    fn from(v: Vector3) -> Point3 {
        Point3::new(v.x, v.y, v.z)
    }
}

// ===== Vector3 ===============================================================================================================================================

impl Vector3 {
    /// Creates a new `Vector3` with x, y and z coordinate values.
    pub const fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }

    /// Returns a `Vector3` which represents the zero vector: x = 0, y = 0 and z = 0.
    pub const fn zero() -> Vector3 {
        Vector3::new(0.0, 0.0, 0.0)
    }

    /// Returns a `Vector3` which represents the x axis: x = 1, y = 0 and z = 0.
    pub const fn x_axis() -> Vector3 {
        Vector3::new(1.0, 0.0, 0.0)
    }

    /// Returns a `Vector3` which represents the y axis: x = 0, y = 1 and z = 0.
    pub const fn y_axis() -> Vector3 {
        Vector3::new(0.0, 1.0, 0.0)
    }

    /// Returns a `Vector3` which represents the z axis: x = 0, y = 0 and z = 1.
    pub const fn z_axis() -> Vector3 {
        Vector3::new(0.0, 0.0, 1.0)
    }

    /// Returns a `Vector3` which represents the axis specified by the dimension `dim`.
    pub fn axis(dim: Dimension3) -> Vector3 {
        match dim {
            Dimension3::X => Vector3::x_axis(),
            Dimension3::Y => Vector3::y_axis(),
            Dimension3::Z => Vector3::z_axis(),
        }
    }

    /// Returns the length of this vector.
    ///
    /// If you only need to compare relative vector lengths, consider using [`length_squared`](#method.length_squared) instead, which avoids a costly
    /// square root computation.
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the square of the length of this vector.
    ///
    /// Use this method instead of [`length`](#method.length) when possible, to avoid a costly square root computation.
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Returns a new vector which points in the same direction as this vector and which has length 1.
    pub fn normalize(self) -> Vector3 {
        self / self.length()
    }

    /// Computes the dot product between this vector and the vector `v`.
    pub fn dot(self, v: Vector3) -> f32 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// Computes the cross product between this vector and the vector `v`.
    pub fn cross(self, v: Vector3) -> Vector3 {
        Vector3::new(self.y * v.z - self.z * v.y, self.z * v.x - self.x * v.z, self.x * v.y - self.y * v.x)
    }

    /// Returns the element-wise minimum between this vector and the vector `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Vector3;
    ///
    /// let v1 = Vector3::new(2.0, 1.0, 3.0);
    /// let v2 = Vector3::new(4.0, -1.5, 0.5);
    /// assert_eq!(v1.min(v2), Vector3::new(2.0, -1.5, 0.5));
    /// ```
    pub fn min(self, v: Vector3) -> Vector3 {
        Vector3::new(f32::min(self.x, v.x), f32::min(self.y, v.y), f32::min(self.z, v.z))
    }

    /// Returns the element-wise maximum between this vector and the vector `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Vector3;
    ///
    /// let v1 = Vector3::new(2.0, 1.0, 3.0);
    /// let v2 = Vector3::new(4.0, -1.5, 0.5);
    /// assert_eq!(v1.max(v2), Vector3::new(4.0, 1.0, 3.0));
    /// ```
    pub fn max(self, v: Vector3) -> Vector3 {
        Vector3::new(f32::max(self.x, v.x), f32::max(self.y, v.y), f32::max(self.z, v.z))
    }

    /// Returns the dimension with the minimum extent of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension3, Vector3};
    ///
    /// let v = Vector3::new(2.0, -3.0, 1.0);
    /// assert_eq!(v.min_dimension(), Dimension3::Z);
    /// ```
    pub fn min_dimension(self) -> Dimension3 {
        let Vector3 { x, y, z } = self.abs();
        if x <= y && x <= z { Dimension3::X } else if y <= z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the dimension with the maximum extent of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension3, Vector3};
    ///
    /// let v = Vector3::new(2.0, -3.0, 1.0);
    /// assert_eq!(v.max_dimension(), Dimension3::Y);
    /// ```
    pub fn max_dimension(self) -> Dimension3 {
        let Vector3 { x, y, z } = self.abs();
        if x > y && x > z { Dimension3::X } else if y > z { Dimension3::Y } else { Dimension3::Z }
    }

    /// Returns the element-wise floor of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Vector3;
    ///
    /// let v = Vector3::new(1.5, -2.25, 0.75);
    /// assert_eq!(v.floor(), Vector3::new(1.0, -3.0, 0.0));
    /// ```
    pub fn floor(self) -> Vector3 {
        Vector3::new(self.x.floor(), self.y.floor(), self.z.floor())
    }

    /// Returns the element-wise ceiling of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Vector3;
    ///
    /// let v = Vector3::new(1.5, -2.25, 0.75);
    /// assert_eq!(v.ceil(), Vector3::new(2.0, -2.0, 1.0));
    /// ```
    pub fn ceil(self) -> Vector3 {
        Vector3::new(self.x.ceil(), self.y.ceil(), self.z.ceil())
    }

    /// Returns the element-wise absolute value of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Vector3;
    ///
    /// let v = Vector3::new(3.5, -2.0, -0.5);
    /// assert_eq!(v.abs(), Vector3::new(3.5, 2.0, 0.5));
    /// ```
    pub fn abs(self) -> Vector3 {
        Vector3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a vector with a permutation of the elements of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension3, Vector3};
    ///
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// assert_eq!(v.permute(Dimension3::Y, Dimension3::Z, Dimension3::X), Vector3::new(2.0, 3.0, 1.0));
    /// ```
    pub fn permute(self, dim_x: Dimension3, dim_y: Dimension3, dim_z: Dimension3) -> Vector3 {
        Vector3::new(self[dim_x], self[dim_y], self[dim_z])
    }

    /// Generates a coordinate system from this vector.
    ///
    /// Two vectors are generated that are perpendicular to this vector and to each other.
    ///
    /// Note: This function assumes that the vector you call is on is normalized.
    pub fn coordinate_system(self) -> (Vector3, Vector3) {
        let v2 = if self.x.abs() > self.y.abs() {
            Vector3::new(-self.z, 0.0, self.x) / (self.x * self.x + self.z * self.z).sqrt()
        } else {
            Vector3::new(0.0, self.z, -self.y) / (self.y * self.y + self.z * self.z).sqrt()
        };

        (v2, self.cross(v2))
    }
}

impl Index<Dimension3> for Vector3 {
    type Output = f32;

    fn index(&self, index: Dimension3) -> &f32 {
        match index {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl IndexMut<Dimension3> for Vector3 {
    fn index_mut(&mut self, index: Dimension3) -> &mut f32 {
        match index {
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

impl Mul<f32> for Vector3 {
    type Output = Vector3;

    fn mul(self, s: f32) -> Vector3 {
        Vector3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Vector3> for f32 {
    type Output = Vector3;

    fn mul(self, v: Vector3) -> Self::Output {
        v * self
    }
}

impl MulAssign<f32> for Vector3 {
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl Div<f32> for Vector3 {
    type Output = Vector3;

    fn div(self, s: f32) -> Self::Output {
        Vector3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl DivAssign<f32> for Vector3 {
    fn div_assign(&mut self, s: f32) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl From<Point3> for Vector3 {
    fn from(p: Point3) -> Self {
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
    /// Creates a new `Normal3` with x, y and z coordinate values.
    pub const fn new(x: f32, y: f32, z: f32) -> Normal3 {
        Normal3 { x, y, z }
    }

    /// Returns a `Normal3` which represents the zero vector: x = 0, y = 0 and z = 0.
    pub const fn zero() -> Normal3 {
        Normal3::new(0.0, 0.0, 0.0)
    }

    /// Returns a `Normal3` which represents the x axis: x = 1, y = 0 and z = 0.
    pub const fn x_axis() -> Normal3 {
        Normal3::new(1.0, 0.0, 0.0)
    }

    /// Returns a `Normal3` which represents the y axis: x = 0, y = 1 and z = 0.
    pub const fn y_axis() -> Normal3 {
        Normal3::new(0.0, 1.0, 0.0)
    }

    /// Returns a `Normal3` which represents the z axis: x = 0, y = 0 and z = 1.
    pub const fn z_axis() -> Normal3 {
        Normal3::new(0.0, 0.0, 1.0)
    }

    /// Returns a `Normal3` which represents the axis specified by the dimension `dim`.
    pub fn axis(dim: Dimension3) -> Normal3 {
        match dim {
            Dimension3::X => Normal3::x_axis(),
            Dimension3::Y => Normal3::y_axis(),
            Dimension3::Z => Normal3::z_axis(),
        }
    }

    /// Returns the length of this normal.
    ///
    /// If you only need to compare relative normal lengths, consider using [`length_squared`](#method.length_squared) instead, which avoids a costly
    /// square root computation.
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the square of the length of this normal.
    ///
    /// Use this method instead of [`length`](#method.length) when possible, to avoid a costly square root computation.
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Returns a new normal which points in the same direction as this normal and which has length 1.
    pub fn normalize(self) -> Normal3 {
        self / self.length()
    }

    /// Computes the dot product between this normal and the vector `v`.
    pub fn dot(self, v: Vector3) -> f32 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}

impl Index<Dimension3> for Normal3 {
    type Output = f32;

    fn index(&self, index: Dimension3) -> &f32 {
        match index {
            Dimension3::X => &self.x,
            Dimension3::Y => &self.y,
            Dimension3::Z => &self.z,
        }
    }
}

impl IndexMut<Dimension3> for Normal3 {
    fn index_mut(&mut self, index: Dimension3) -> &mut f32 {
        match index {
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

impl Add<Vector3> for Normal3 {
    type Output = Normal3;

    fn add(self, v: Vector3) -> Normal3 {
        Normal3::new(self.x + v.x, self.y + v.y, self.z + v.z)
    }
}

impl AddAssign<Vector3> for Normal3 {
    fn add_assign(&mut self, v: Vector3) {
        self.x += v.x;
        self.y += v.y;
        self.z += v.z;
    }
}

impl Sub<Vector3> for Normal3 {
    type Output = Normal3;

    fn sub(self, v: Vector3) -> Normal3 {
        Normal3::new(self.x - v.x, self.y - v.y, self.z - v.z)
    }
}

impl SubAssign<Vector3> for Normal3 {
    fn sub_assign(&mut self, v: Vector3) {
        self.x -= v.x;
        self.y -= v.y;
        self.z -= v.z;
    }
}

impl Neg for Normal3 {
    type Output = Normal3;

    fn neg(self) -> Normal3 {
        Normal3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f32> for Normal3 {
    type Output = Normal3;

    fn mul(self, s: f32) -> Normal3 {
        Normal3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Normal3> for f32 {
    type Output = Normal3;

    fn mul(self, n: Normal3) -> Self::Output {
        n * self
    }
}

impl MulAssign<f32> for Normal3 {
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
        self.z *= s;
    }
}

impl Div<f32> for Normal3 {
    type Output = Normal3;

    fn div(self, s: f32) -> Self::Output {
        Normal3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl DivAssign<f32> for Normal3 {
    fn div_assign(&mut self, s: f32) {
        self.x /= s;
        self.y /= s;
        self.z /= s;
    }
}

impl From<Vector3> for Normal3 {
    fn from(v: Vector3) -> Self {
        Normal3::new(v.x, v.y, v.z)
    }
}

// ===== Matrix4x4 =============================================================================================================================================

impl Matrix4x4 {
    /// Creates a new `Matrix4x4` with the given elements.
    ///
    /// Elements in a matrix are stored in row-major order.
    pub fn new(m: [f32; 16]) -> Matrix4x4 {
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

    /// Returns a `Matrix4x4` that translates along the vector `v`.
    pub fn translate(v: Vector3) -> Matrix4x4 {
        Matrix4x4::new([
            1.0, 0.0, 0.0, v.x,
            0.0, 1.0, 0.0, v.y,
            0.0, 0.0, 1.0, v.z,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a `Matrix4x4` that rotates around the x axis by the specified angle (in radians).
    pub fn rotate_x(angle: f32) -> Matrix4x4 {
        let (sin, cos) = angle.sin_cos();

        Matrix4x4::new([
            1.0, 0.0, 0.0, 0.0,
            0.0, cos, -sin, 0.0,
            0.0, sin, cos, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a `Matrix4x4` that rotates around the y axis by the specified angle (in radians).
    pub fn rotate_y(angle: f32) -> Matrix4x4 {
        let (sin, cos) = angle.sin_cos();

        Matrix4x4::new([
            cos, 0.0, sin, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sin, 0.0, cos, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a `Matrix4x4` that rotates around the z axis by the specified angle (in radians).
    pub fn rotate_z(angle: f32) -> Matrix4x4 {
        let (sin, cos) = angle.sin_cos();

        Matrix4x4::new([
            cos, -sin, 0.0, 0.0,
            sin, cos, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a `Matrix4x4` that rotates around the specified axis by the specified angle (in radians).
    pub fn rotate_axis(axis: Vector3, angle: f32) -> Matrix4x4 {
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

    /// Returns a `Matrix4x4` that scales by factor `sx` along the x dimension, factor `sy` along the y dimension and factor `sz` along the z dimension.
    pub fn scale(sx: f32, sy: f32, sz: f32) -> Matrix4x4 {
        Matrix4x4::new([
            sx, 0.0, 0.0, 0.0,
            0.0, sy, 0.0, 0.0,
            0.0, 0.0, sz, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a `Matrix4x4` that scales uniformly by factor `s` along all dimensions.
    pub fn scale_uniform(s: f32) -> Matrix4x4 {
        Matrix4x4::new([
            s, 0.0, 0.0, 0.0,
            0.0, s, 0.0, 0.0,
            0.0, 0.0, s, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Returns a `Matrix4x4` that performs the inverse of a look-at transform.
    ///
    /// # Parameters
    ///
    /// - `from` - the observer's location.
    /// - `target` - the location to "look at".
    /// - `up` - the "up" direction.
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

    /// Returns the element at row `row` and column `col`.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.m[row * 4 + col]
    }

    /// Returns a mutable reference to the element at row `row` and column `col`.
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f32 {
        &mut self.m[row * 4 + col]
    }

    /// Sets the element at row `row` and column `col` to the specified value.
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.m[row * 4 + col] = value;
    }

    /// Returns the row specified by the given index.
    pub fn row(&self, index: usize) -> [f32; 4] {
        let i = index * 4;
        [self.m[i], self.m[i + 1], self.m[i + 2], self.m[i + 3]]
    }

    /// Returns the column specified by the given index.
    pub fn column(&self, index: usize) -> [f32; 4] {
        [self.m[index], self.m[index + 4], self.m[index + 8], self.m[index + 12]]
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

    /// Computes the inverse of this matrix.
    ///
    /// If the matrix is singular, a `NonInvertibleMatrixError` is returned.
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
            Ok(&adjugate / det)
        } else {
            Err(NonInvertibleMatrixError)
        }
    }
}

impl Mul for &Matrix4x4 {
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

impl Mul<f32> for &Matrix4x4 {
    type Output = Matrix4x4;

    fn mul(self, s: f32) -> Matrix4x4 {
        Matrix4x4::new(array![|i| self.m[i] * s; 16])
    }
}

impl Mul<&Matrix4x4> for f32 {
    type Output = Matrix4x4;

    fn mul(self, m: &Matrix4x4) -> Matrix4x4 {
        m * self
    }
}

impl MulAssign<f32> for &mut Matrix4x4 {
    fn mul_assign(&mut self, s: f32) {
        for m in &mut self.m {
            *m *= s;
        }
    }
}

impl Div<f32> for &Matrix4x4 {
    type Output = Matrix4x4;

    fn div(self, s: f32) -> Matrix4x4 {
        Matrix4x4::new(array![|i| self.m[i] / s; 16])
    }
}

impl DivAssign<f32> for &mut Matrix4x4 {
    fn div_assign(&mut self, s: f32) {
        for m in &mut self.m {
            *m /= s;
        }
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

// ===== Transform3 ============================================================================================================================================

impl Transform3 {
    fn new(forward: Arc<Matrix4x4>, inverse: Arc<Matrix4x4>) -> Transform3 {
        Transform3 { forward, inverse }
    }

    /// Returns a `Transform3` which represents the identity transform.
    pub fn identity() -> Transform3 {
        let forward = Arc::new(Matrix4x4::identity());
        let inverse = forward.clone();
        Transform3::new(forward, inverse)
    }

    /// Returns a `Transform3` that translates along the vector `v`.
    pub fn translate(v: Vector3) -> Transform3 {
        Transform3::new(Arc::new(Matrix4x4::translate(v)), Arc::new(Matrix4x4::translate(-v)))
    }

    /// Returns a `Transform3` that rotates around the x axis by the specified angle (in radians).
    pub fn rotate_x(angle: f32) -> Transform3 {
        let forward = Matrix4x4::rotate_x(angle);
        let inverse = forward.transpose();
        Transform3::new(Arc::new(forward), Arc::new(inverse))
    }

    /// Returns a `Transform3` that rotates around the y axis by the specified angle (in radians).
    pub fn rotate_y(angle: f32) -> Transform3 {
        let forward = Matrix4x4::rotate_y(angle);
        let inverse = forward.transpose();
        Transform3::new(Arc::new(forward), Arc::new(inverse))
    }

    /// Returns a `Transform3` that rotates around the z axis by the specified angle (in radians).
    pub fn rotate_z(angle: f32) -> Transform3 {
        let forward = Matrix4x4::rotate_z(angle);
        let inverse = forward.transpose();
        Transform3::new(Arc::new(forward), Arc::new(inverse))
    }

    /// Returns a `Transform3` that rotates around the specified axis by the specified angle (in radians).
    pub fn rotate_axis(axis: Vector3, angle: f32) -> Transform3 {
        let forward = Matrix4x4::rotate_axis(axis, angle);
        let inverse = forward.transpose();
        Transform3::new(Arc::new(forward), Arc::new(inverse))
    }

    /// Returns a `Transform3` that scales by factor `sx` along the x dimension, factor `sy` along the y dimension and factor `sz` along the z dimension.
    pub fn scale(sx: f32, sy: f32, sz: f32) -> Transform3 {
        Transform3::new(Arc::new(Matrix4x4::scale(sx, sy, sz)), Arc::new(Matrix4x4::scale(1.0 / sx, 1.0 / sy, 1.0 / sz)))
    }

    /// Returns a `Transform3` that scales uniformly by factor `s` along all dimensions.
    pub fn scale_uniform(s: f32) -> Transform3 {
        Transform3::new(Arc::new(Matrix4x4::scale_uniform(s)), Arc::new(Matrix4x4::scale_uniform(1.0 / s)))
    }

    /// Returns a `Transform3` that performs a look-at transform.
    ///
    /// # Parameters
    ///
    /// - `from` - the observer's location.
    /// - `target` - the location to "look at".
    /// - `up` - the "up" direction.
    pub fn look_at(from: Point3, target: Point3, up: Vector3) -> Result<Transform3, NonInvertibleMatrixError> {
        let inverse = Matrix4x4::inverse_look_at(from, target, up);
        let forward = inverse.inverse()?;
        Ok(Transform3::new(Arc::new(forward), Arc::new(inverse)))
    }

    /// Creates a `Transform3` from a matrix.
    ///
    /// The inverse of the matrix is computed. If the matrix is singular, a `NonInvertibleMatrixError` is returned.
    pub fn from_matrix(forward: Matrix4x4) -> Result<Transform3, NonInvertibleMatrixError> {
        let inverse = forward.inverse()?;
        Ok(Transform3::new(Arc::new(forward), Arc::new(inverse)))
    }

    /// Combines this transform with another transform.
    ///
    /// The result is a transform that first applies this transform, followed by the specified transform.
    pub fn and_then(&self, transform: &Transform3) -> Transform3 {
        Transform3::new(Arc::new(&*transform.forward * &*self.forward), Arc::new(&*self.inverse * &*transform.inverse))
    }

    /// Returns the inverse of this transform.
    ///
    /// This is a cheap operation; no matrix inverse needs to be computed.
    pub fn inverse(&self) -> Transform3 {
        Transform3::new(self.inverse.clone(), self.forward.clone())
    }
}

// ===== ApplyTransform3 =======================================================================================================================================

impl ApplyTransform3<Point3> for Transform3 {
    type Output = Point3;

    fn transform(&self, p: Point3) -> Point3 {
        &*self.forward * p
    }
}

impl ApplyTransform3<Vector3> for Transform3 {
    type Output = Vector3;

    fn transform(&self, v: Vector3) -> Vector3 {
        &*self.forward * v
    }
}

impl ApplyTransform3<Normal3> for Transform3 {
    type Output = Normal3;

    fn transform(&self, n: Normal3) -> Normal3 {
        // Normals are transformed by the transpose of the inverse
        (Vector3::from(n) * &*self.inverse).into()
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

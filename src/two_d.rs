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

/// Dimension in 2D space.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Dimension2 { X, Y }

/// Point in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

/// Vector in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

/// Normal vector in 2D space.
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct Normal2 {
    pub x: f32,
    pub y: f32,
}

/// Transformation matrix for transformations in 2D space.
#[derive(Clone, PartialEq, Debug)]
#[repr(C, align(16))]
pub struct Matrix3x3 {
    m: [f32; 9]
}

/// Transformation in 2D space; holds shared pointers to a transformation matrix and its inverse.
#[derive(Clone, PartialEq, Debug)]
pub struct Transform2 {
    pub forward: Arc<Matrix3x3>,
    pub inverse: Arc<Matrix3x3>,
}

/// Implement this trait for `Transform2` with a specific type to be able to transform values of that type with a `Transform2`.
pub trait ApplyTransform2<T> {
    type Output;

    fn transform(&self, input: T) -> Self::Output;
}

// ===== Point2 ================================================================================================================================================

impl Point2 {
    /// Creates a new `Point2` with x and y coordinate values.
    pub const fn new(x: f32, y: f32) -> Point2 {
        Point2 { x, y }
    }

    /// Returns a `Point2` which represents the origin: x = 0 and y = 0.
    pub const fn origin() -> Point2 {
        Point2::new(0.0, 0.0)
    }

    /// Computes the distance between two points.
    ///
    /// If you only need to compare relative distances, consider using [`distance_squared`](#method.distance_squared) instead, which avoids a costly
    /// square root computation.
    pub fn distance(self, p: Point2) -> f32 {
        (self - p).length()
    }

    /// Computes the square of the distance between two points.
    ///
    /// Use this method instead of [`distance`](#method.distance) when possible, to avoid a costly square root computation.
    pub fn distance_squared(self, p: Point2) -> f32 {
        (self - p).length_squared()
    }

    /// Returns the element-wise minimum between this point and the point `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Point2;
    ///
    /// let p1 = Point2::new(2.0, 1.0);
    /// let p2 = Point2::new(4.0, -1.5);
    /// assert_eq!(p1.min(p2), Point2::new(2.0, -1.5));
    /// ```
    pub fn min(self, p: Point2) -> Point2 {
        Point2::new(f32::min(self.x, p.x), f32::min(self.y, p.y))
    }

    /// Returns the element-wise maximum between this point and the point `p`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Point2;
    ///
    /// let p1 = Point2::new(2.0, 1.0);
    /// let p2 = Point2::new(4.0, -1.5);
    /// assert_eq!(p1.max(p2), Point2::new(4.0, 1.0));
    /// ```
    pub fn max(self, p: Point2) -> Point2 {
        Point2::new(f32::max(self.x, p.x), f32::max(self.y, p.y))
    }

    /// Returns the dimension with the minimum extent of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension2, Point2};
    ///
    /// let p = Point2::new(2.0, -3.0);
    /// assert_eq!(p.min_dimension(), Dimension2::X);
    /// ```
    pub fn min_dimension(self) -> Dimension2 {
        if self.x.abs() <= self.y.abs() { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the maximum extent of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension2, Point2};
    ///
    /// let p = Point2::new(2.0, -3.0);
    /// assert_eq!(p.max_dimension(), Dimension2::Y);
    /// ```
    pub fn max_dimension(self) -> Dimension2 {
        if self.x.abs() > self.y.abs() { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the element-wise floor of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Point2;
    ///
    /// let p = Point2::new(1.5, -2.25);
    /// assert_eq!(p.floor(), Point2::new(1.0, -3.0));
    /// ```
    pub fn floor(self) -> Point2 {
        Point2::new(self.x.floor(), self.y.floor())
    }

    /// Returns the element-wise ceiling of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Point2;
    ///
    /// let p = Point2::new(1.5, -2.25);
    /// assert_eq!(p.ceil(), Point2::new(2.0, -2.0));
    /// ```
    pub fn ceil(self) -> Point2 {
        Point2::new(self.x.ceil(), self.y.ceil())
    }

    /// Returns the element-wise absolute value of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Point2;
    ///
    /// let p = Point2::new(3.5, -2.0);
    /// assert_eq!(p.abs(), Point2::new(3.5, 2.0));
    /// ```
    pub fn abs(self) -> Point2 {
        Point2::new(self.x.abs(), self.y.abs())
    }

    /// Returns a point with a permutation of the elements of this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension2, Point2};
    ///
    /// let p = Point2::new(1.0, 2.0);
    /// assert_eq!(p.permute(Dimension2::Y, Dimension2::X), Point2::new(2.0, 1.0));
    /// ```
    pub fn permute(self, dim_x: Dimension2, dim_y: Dimension2) -> Point2 {
        Point2::new(self[dim_x], self[dim_y])
    }
}

impl Index<Dimension2> for Point2 {
    type Output = f32;

    fn index(&self, index: Dimension2) -> &f32 {
        match index {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl IndexMut<Dimension2> for Point2 {
    fn index_mut(&mut self, index: Dimension2) -> &mut f32 {
        match index {
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

impl Mul<f32> for Point2 {
    type Output = Point2;

    fn mul(self, s: f32) -> Point2 {
        Point2::new(self.x * s, self.y * s)
    }
}

impl Mul<Point2> for f32 {
    type Output = Point2;

    fn mul(self, p: Point2) -> Point2 {
        p * self
    }
}

impl MulAssign<f32> for Point2 {
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
    }
}

impl Div<f32> for Point2 {
    type Output = Point2;

    fn div(self, s: f32) -> Point2 {
        Point2::new(self.x / s, self.y / s)
    }
}

impl DivAssign<f32> for Point2 {
    fn div_assign(&mut self, s: f32) {
        self.x /= s;
        self.y /= s;
    }
}

impl From<Vector2> for Point2 {
    fn from(v: Vector2) -> Point2 {
        Point2::new(v.x, v.y)
    }
}

// ===== Vector2 ===============================================================================================================================================

impl Vector2 {
    /// Creates a new `Vector2` with x and y coordinate values.
    pub const fn new(x: f32, y: f32) -> Vector2 {
        Vector2 { x, y }
    }

    /// Returns a `Vector2` which represents the zero vector: x = 0 and y = 0.
    pub const fn zero() -> Vector2 {
        Vector2::new(0.0, 0.0)
    }

    /// Returns a `Vector2` which represents the x axis: x = 1 and y = 0.
    pub const fn x_axis() -> Vector2 {
        Vector2::new(1.0, 0.0)
    }

    /// Returns a `Vector2` which represents the y axis: x = 0 and y = 1.
    pub const fn y_axis() -> Vector2 {
        Vector2::new(0.0, 1.0)
    }

    /// Returns a `Vector2` which represents the axis specified by the dimension `dim`.
    pub fn axis(dim: Dimension2) -> Vector2 {
        match dim {
            Dimension2::X => Vector2::x_axis(),
            Dimension2::Y => Vector2::y_axis(),
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
        self.x * self.x + self.y * self.y
    }

    /// Returns a new vector which points in the same direction as this vector and which has length 1.
    pub fn normalize(self) -> Vector2 {
        self / self.length()
    }

    /// Computes the dot product between this vector and the vector `v`.
    pub fn dot(self, v: Vector2) -> f32 {
        self.x * v.x + self.y * v.y
    }

    /// Returns the element-wise minimum between this vector and the vector `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Vector2;
    ///
    /// let v1 = Vector2::new(2.0, 1.0);
    /// let v2 = Vector2::new(4.0, -1.5);
    /// assert_eq!(v1.min(v2), Vector2::new(2.0, -1.5));
    /// ```
    pub fn min(self, v: Vector2) -> Vector2 {
        Vector2::new(f32::min(self.x, v.x), f32::min(self.y, v.y))
    }

    /// Returns the element-wise maximum between this vector and the vector `v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Vector2;
    ///
    /// let v1 = Vector2::new(2.0, 1.0);
    /// let v2 = Vector2::new(4.0, -1.5);
    /// assert_eq!(v1.max(v2), Vector2::new(4.0, 1.0));
    /// ```
    pub fn max(self, v: Vector2) -> Vector2 {
        Vector2::new(f32::max(self.x, v.x), f32::max(self.y, v.y))
    }

    /// Returns the dimension with the minimum extent of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension2, Vector2};
    ///
    /// let v = Vector2::new(2.0, -3.0);
    /// assert_eq!(v.min_dimension(), Dimension2::X);
    /// ```
    pub fn min_dimension(self) -> Dimension2 {
        if self.x.abs() <= self.y.abs() { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the dimension with the maximum extent of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension2, Vector2};
    ///
    /// let v = Vector2::new(2.0, -3.0);
    /// assert_eq!(v.max_dimension(), Dimension2::Y);
    /// ```
    pub fn max_dimension(self) -> Dimension2 {
        if self.x.abs() > self.y.abs() { Dimension2::X } else { Dimension2::Y }
    }

    /// Returns the element-wise floor of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Vector2;
    ///
    /// let v = Vector2::new(1.5, -2.25);
    /// assert_eq!(v.floor(), Vector2::new(1.0, -3.0));
    /// ```
    pub fn floor(self) -> Vector2 {
        Vector2::new(self.x.floor(), self.y.floor())
    }

    /// Returns the element-wise ceiling of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Vector2;
    ///
    /// let v = Vector2::new(1.5, -2.25);
    /// assert_eq!(v.ceil(), Vector2::new(2.0, -2.0));
    /// ```
    pub fn ceil(self) -> Vector2 {
        Vector2::new(self.x.ceil(), self.y.ceil())
    }

    /// Returns the element-wise absolute value of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::Vector2;
    ///
    /// let v = Vector2::new(3.5, -2.0);
    /// assert_eq!(v.abs(), Vector2::new(3.5, 2.0));
    /// ```
    pub fn abs(self) -> Vector2 {
        Vector2::new(self.x.abs(), self.y.abs())
    }

    /// Returns a vector with a permutation of the elements of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use vecmath::{Dimension2, Vector2};
    ///
    /// let v = Vector2::new(1.0, 2.0);
    /// assert_eq!(v.permute(Dimension2::Y, Dimension2::X), Vector2::new(2.0, 1.0));
    /// ```
    pub fn permute(self, dim_x: Dimension2, dim_y: Dimension2) -> Vector2 {
        Vector2::new(self[dim_x], self[dim_y])
    }
}

impl Index<Dimension2> for Vector2 {
    type Output = f32;

    fn index(&self, index: Dimension2) -> &f32 {
        match index {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl IndexMut<Dimension2> for Vector2 {
    fn index_mut(&mut self, index: Dimension2) -> &mut f32 {
        match index {
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

impl Mul<f32> for Vector2 {
    type Output = Vector2;

    fn mul(self, s: f32) -> Vector2 {
        Vector2::new(self.x * s, self.y * s)
    }
}

impl Mul<Vector2> for f32 {
    type Output = Vector2;

    fn mul(self, v: Vector2) -> Self::Output {
        v * self
    }
}

impl MulAssign<f32> for Vector2 {
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
    }
}

impl Div<f32> for Vector2 {
    type Output = Vector2;

    fn div(self, s: f32) -> Self::Output {
        Vector2::new(self.x / s, self.y / s)
    }
}

impl DivAssign<f32> for Vector2 {
    fn div_assign(&mut self, s: f32) {
        self.x /= s;
        self.y /= s;
    }
}

impl From<Point2> for Vector2 {
    fn from(p: Point2) -> Self {
        Vector2::new(p.x, p.y)
    }
}

impl From<Normal2> for Vector2 {
    fn from(n: Normal2) -> Self {
        Vector2::new(n.x, n.y)
    }
}

// ===== Normal2 ===============================================================================================================================================

impl Normal2 {
    /// Creates a new `Normal2` with x and y coordinate values.
    pub const fn new(x: f32, y: f32) -> Normal2 {
        Normal2 { x, y }
    }

    /// Returns a `Normal2` which represents the zero vector: x = 0 and y = 0.
    pub const fn zero() -> Normal2 {
        Normal2::new(0.0, 0.0)
    }

    /// Returns a `Normal2` which represents the x axis: x = 1 and y = 0.
    pub const fn x_axis() -> Normal2 {
        Normal2::new(1.0, 0.0)
    }

    /// Returns a `Normal2` which represents the y axis: x = 0 and y = 1.
    pub const fn y_axis() -> Normal2 {
        Normal2::new(0.0, 1.0)
    }

    /// Returns a `Normal2` which represents the axis specified by the dimension `dim`.
    pub fn axis(dim: Dimension2) -> Normal2 {
        match dim {
            Dimension2::X => Normal2::x_axis(),
            Dimension2::Y => Normal2::y_axis(),
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
        self.x * self.x + self.y * self.y
    }

    /// Returns a new normal which points in the same direction as this normal and which has length 1.
    pub fn normalize(self) -> Normal2 {
        self / self.length()
    }

    /// Computes the dot product between this normal and the vector `v`.
    pub fn dot(self, v: Vector2) -> f32 {
        self.x * v.x + self.y * v.y
    }
}

impl Index<Dimension2> for Normal2 {
    type Output = f32;

    fn index(&self, index: Dimension2) -> &f32 {
        match index {
            Dimension2::X => &self.x,
            Dimension2::Y => &self.y,
        }
    }
}

impl IndexMut<Dimension2> for Normal2 {
    fn index_mut(&mut self, index: Dimension2) -> &mut f32 {
        match index {
            Dimension2::X => &mut self.x,
            Dimension2::Y => &mut self.y,
        }
    }
}

impl Add<Normal2> for Normal2 {
    type Output = Normal2;

    fn add(self, n: Normal2) -> Normal2 {
        Normal2::new(self.x + n.x, self.y + n.y)
    }
}

impl AddAssign<Normal2> for Normal2 {
    fn add_assign(&mut self, n: Normal2) {
        self.x += n.x;
        self.y += n.y;
    }
}

impl Sub<Normal2> for Normal2 {
    type Output = Normal2;

    fn sub(self, n: Normal2) -> Normal2 {
        Normal2::new(self.x - n.x, self.y - n.y)
    }
}

impl SubAssign<Normal2> for Normal2 {
    fn sub_assign(&mut self, n: Normal2) {
        self.x -= n.x;
        self.y -= n.y;
    }
}

impl Add<Vector2> for Normal2 {
    type Output = Normal2;

    fn add(self, v: Vector2) -> Normal2 {
        Normal2::new(self.x + v.x, self.y + v.y)
    }
}

impl AddAssign<Vector2> for Normal2 {
    fn add_assign(&mut self, v: Vector2) {
        self.x += v.x;
        self.y += v.y;
    }
}

impl Sub<Vector2> for Normal2 {
    type Output = Normal2;

    fn sub(self, v: Vector2) -> Normal2 {
        Normal2::new(self.x - v.x, self.y - v.y)
    }
}

impl SubAssign<Vector2> for Normal2 {
    fn sub_assign(&mut self, v: Vector2) {
        self.x -= v.x;
        self.y -= v.y;
    }
}

impl Neg for Normal2 {
    type Output = Normal2;

    fn neg(self) -> Normal2 {
        Normal2::new(-self.x, -self.y)
    }
}

impl Mul<f32> for Normal2 {
    type Output = Normal2;

    fn mul(self, s: f32) -> Normal2 {
        Normal2::new(self.x * s, self.y * s)
    }
}

impl Mul<Normal2> for f32 {
    type Output = Normal2;

    fn mul(self, n: Normal2) -> Self::Output {
        n * self
    }
}

impl MulAssign<f32> for Normal2 {
    fn mul_assign(&mut self, s: f32) {
        self.x *= s;
        self.y *= s;
    }
}

impl Div<f32> for Normal2 {
    type Output = Normal2;

    fn div(self, s: f32) -> Self::Output {
        Normal2::new(self.x / s, self.y / s)
    }
}

impl DivAssign<f32> for Normal2 {
    fn div_assign(&mut self, s: f32) {
        self.x /= s;
        self.y /= s;
    }
}

impl From<Vector2> for Normal2 {
    fn from(v: Vector2) -> Self {
        Normal2::new(v.x, v.y)
    }
}

// ===== Matrix3x3 =============================================================================================================================================

impl Matrix3x3 {
    /// Creates a new `Matrix3x3` with the given elements.
    ///
    /// Elements in a matrix are stored in row-major order.
    pub fn new(m: [f32; 9]) -> Matrix3x3 {
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

    /// Returns a `Matrix3x3` that translates along the vector `v`.
    pub fn translate(v: Vector2) -> Matrix3x3 {
        Matrix3x3::new([
            1.0, 0.0, v.x,
            0.0, 1.0, v.y,
            0.0, 0.0, 1.0,
        ])
    }

    /// Returns a `Matrix3x3` that rotates around the origin by the specified angle (in radians).
    pub fn rotate(angle: f32) -> Matrix3x3 {
        let (sin, cos) = angle.sin_cos();

        Matrix3x3::new([
            cos, -sin, 0.0,
            sin, cos, 0.0,
            0.0, 0.0, 1.0,
        ])
    }

    /// Returns a `Matrix3x3` that scales by factor `sx` along the x dimension and factor `sy` along the y dimension.
    pub fn scale(sx: f32, sy: f32) -> Matrix3x3 {
        Matrix3x3::new([
            sx, 0.0, 0.0,
            0.0, sy, 0.0,
            0.0, 0.0, 1.0,
        ])
    }

    /// Returns a `Matrix3x3` that scales uniformly by factor `s` along all dimensions.
    pub fn scale_uniform(s: f32) -> Matrix3x3 {
        Matrix3x3::new([
            s, 0.0, 0.0,
            0.0, s, 0.0,
            0.0, 0.0, 1.0,
        ])
    }

    /// Returns the element at row `row` and column `col`.
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.m[row * 3 + col]
    }

    /// Returns a mutable reference to the element at row `row` and column `col`.
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f32 {
        &mut self.m[row * 3 + col]
    }

    /// Sets the element at row `row` and column `col` to the specified value.
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.m[row * 3 + col] = value;
    }

    /// Returns the row specified by the given index.
    pub fn row(&self, index: usize) -> [f32; 3] {
        let i = index * 3;
        [self.m[i], self.m[i + 1], self.m[i + 2]]
    }

    /// Returns the column specified by the given index.
    pub fn column(&self, index: usize) -> [f32; 3] {
        [self.m[index], self.m[index + 3], self.m[index + 6]]
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(&self) -> Matrix3x3 {
        Matrix3x3::new([
            self.m[0], self.m[3], self.m[6],
            self.m[1], self.m[4], self.m[7],
            self.m[2], self.m[5], self.m[8],
        ])
    }

    /// Computes the inverse of this matrix.
    ///
    /// If the matrix is singular, a `NonInvertibleMatrixError` is returned.
    pub fn inverse(&self) -> Result<Matrix3x3, NonInvertibleMatrixError> {
        let det = self.m[0] * self.m[4] * self.m[8] + self.m[1] * self.m[5] * self.m[6] + self.m[2] * self.m[3] * self.m[7]
            - self.m[2] * self.m[4] * self.m[6] - self.m[1] * self.m[3] * self.m[8] - self.m[0] * self.m[5] * self.m[7];

        if det != 0.0 {
            Ok(Matrix3x3::new([
                (self.m[4] * self.m[8] - self.m[5] * self.m[7]) / det,
                (self.m[2] * self.m[7] - self.m[1] * self.m[8]) / det,
                (self.m[1] * self.m[5] - self.m[2] * self.m[4]) / det,
                (self.m[5] * self.m[6] - self.m[3] * self.m[8]) / det,
                (self.m[0] * self.m[8] - self.m[2] * self.m[6]) / det,
                (self.m[2] * self.m[3] - self.m[0] * self.m[5]) / det,
                (self.m[3] * self.m[7] - self.m[4] * self.m[6]) / det,
                (self.m[1] * self.m[6] - self.m[0] * self.m[7]) / det,
                (self.m[0] * self.m[4] - self.m[1] * self.m[3]) / det,
            ]))
        } else {
            Err(NonInvertibleMatrixError)
        }
    }
}

impl Mul for &Matrix3x3 {
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

impl Mul<f32> for &Matrix3x3 {
    type Output = Matrix3x3;

    fn mul(self, s: f32) -> Matrix3x3 {
        Matrix3x3::new(array![|i| self.m[i] * s; 9])
    }
}

impl Mul<&Matrix3x3> for f32 {
    type Output = Matrix3x3;

    fn mul(self, m: &Matrix3x3) -> Matrix3x3 {
        m * self
    }
}

impl MulAssign<f32> for &mut Matrix3x3 {
    fn mul_assign(&mut self, s: f32) {
        for m in &mut self.m {
            *m *= s;
        }
    }
}

impl Div<f32> for &Matrix3x3 {
    type Output = Matrix3x3;

    fn div(self, s: f32) -> Matrix3x3 {
        Matrix3x3::new(array![|i| self.m[i] / s; 9])
    }
}

impl DivAssign<f32> for &mut Matrix3x3 {
    fn div_assign(&mut self, s: f32) {
        for m in &mut self.m {
            *m /= s;
        }
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

// ===== Transform2 ============================================================================================================================================

impl Transform2 {
    fn new(forward: Arc<Matrix3x3>, inverse: Arc<Matrix3x3>) -> Transform2 {
        Transform2 { forward, inverse }
    }

    /// Returns a `Transform2` which represents the identity transform.
    pub fn identity() -> Transform2 {
        let forward = Arc::new(Matrix3x3::identity());
        let inverse = forward.clone();
        Transform2::new(forward, inverse)
    }

    /// Returns a `Transform2` that translates along the vector `v`.
    pub fn translate(v: Vector2) -> Transform2 {
        Transform2::new(Arc::new(Matrix3x3::translate(v)), Arc::new(Matrix3x3::translate(-v)))
    }

    /// Returns a `Transform2` that rotates around the origin by the specified angle (in radians).
    pub fn rotate(angle: f32) -> Transform2 {
        let forward = Matrix3x3::rotate(angle);
        let inverse = forward.transpose();
        Transform2::new(Arc::new(forward), Arc::new(inverse))
    }

    /// Returns a `Transform2` that scales by factor `sx` along the x dimension and factor `sy` along the y dimension.
    pub fn scale(sx: f32, sy: f32) -> Transform2 {
        Transform2::new(Arc::new(Matrix3x3::scale(sx, sy)), Arc::new(Matrix3x3::scale(1.0 / sx, 1.0 / sy)))
    }

    /// Returns a `Transform2` that scales uniformly by factor `s` along all dimensions.
    pub fn scale_uniform(s: f32) -> Transform2 {
        Transform2::new(Arc::new(Matrix3x3::scale_uniform(s)), Arc::new(Matrix3x3::scale_uniform(1.0 / s)))
    }

    /// Creates a `Transform2` from a matrix.
    ///
    /// The inverse of the matrix is computed. If the matrix is singular, a `NonInvertibleMatrixError` is returned.
    pub fn from_matrix(forward: Matrix3x3) -> Result<Transform2, NonInvertibleMatrixError> {
        let inverse = forward.inverse()?;
        Ok(Transform2::new(Arc::new(forward), Arc::new(inverse)))
    }

    /// Combines this transform with another transform.
    ///
    /// The result is a transform that first applies this transform, followed by the specified transform.
    pub fn and_then(&self, transform: &Transform2) -> Transform2 {
        Transform2::new(Arc::new(&*transform.forward * &*self.forward), Arc::new(&*self.inverse * &*transform.inverse))
    }

    /// Returns the inverse of this transform.
    ///
    /// This is a cheap operation; no matrix inverse needs to be computed.
    pub fn inverse(&self) -> Transform2 {
        Transform2::new(self.inverse.clone(), self.forward.clone())
    }
}

// ===== ApplyTransform2 =======================================================================================================================================

impl ApplyTransform2<Point2> for Transform2 {
    type Output = Point2;

    fn transform(&self, p: Point2) -> Point2 {
        &*self.forward * p
    }
}

impl ApplyTransform2<Vector2> for Transform2 {
    type Output = Vector2;

    fn transform(&self, v: Vector2) -> Vector2 {
        &*self.forward * v
    }
}

impl ApplyTransform2<Normal2> for Transform2 {
    type Output = Normal2;

    fn transform(&self, n: Normal2) -> Normal2 {
        // Normals are transformed by the transpose of the inverse
        (Vector2::from(n) * &*self.inverse).into()
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

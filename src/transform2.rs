// Copyright 2024 Jesper de Jong
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
use std::ops::{Div, DivAssign, Mul, MulAssign};

use crate::{BoundingBox2, max, min, NonInvertibleMatrixError, Point2, Ray2, Scalar, Transform, Vector2};

/// Matrix with 3 rows and 3 columns for transformations in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix3x3<S: Scalar> {
    m: [S; 9],
}

/// Alias for `Matrix3x3<f32>`.
pub type Matrix3x3f = Matrix3x3<f32>;

/// Alias for `Matrix3x3<f64>`.
pub type Matrix3x3d = Matrix3x3<f64>;

/// Transform for transformations in 2D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Transform2<S: Scalar> {
    pub forward: Matrix3x3<S>,
    pub inverse: Matrix3x3<S>,
}

/// Alias for `Transform2<f32>`.
pub type Transform2f = Transform2<f32>;

/// Alias for `Transform2<f64>`.
pub type Transform2d = Transform2<f64>;

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
    pub fn get(&self, row: u32, col: u32) -> S {
        self.m[Self::linear_index(row, col)]
    }

    /// Returns a mutable reference to an element at a row and column of the matrix.
    #[inline]
    pub fn get_mut(&mut self, row: u32, col: u32) -> &mut S {
        &mut self.m[Self::linear_index(row, col)]
    }

    /// Sets the value of an element at a row and column of the matrix.
    #[inline]
    pub fn set(&mut self, row: u32, col: u32, value: S) {
        self.m[Self::linear_index(row, col)] = value;
    }

    #[inline]
    fn linear_index(row: u32, col: u32) -> usize {
        debug_assert!(row < 3, "Invalid row index: {}", row);
        debug_assert!(col < 3, "Invalid column index: {}", col);
        (row * 3 + col) as usize
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

impl<S: Scalar> Mul<S> for &Matrix3x3<S> {
    type Output = Matrix3x3<S>;

    #[inline]
    fn mul(self, s: S) -> Matrix3x3<S> {
        Matrix3x3::new(array::from_fn(|i| self.m[i] * s))
    }
}

impl<S: Scalar> MulAssign<S> for &mut Matrix3x3<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        for m in &mut self.m { *m *= s; }
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

    fn mul(self, rhs: &Matrix3x3<S>) -> Matrix3x3<S> {
        Matrix3x3::new([
            self.m[0] * rhs.m[0] + self.m[1] * rhs.m[3] + self.m[2] * rhs.m[6],
            self.m[0] * rhs.m[1] + self.m[1] * rhs.m[4] + self.m[2] * rhs.m[7],
            self.m[0] * rhs.m[2] + self.m[1] * rhs.m[5] + self.m[2] * rhs.m[8],
            self.m[3] * rhs.m[0] + self.m[4] * rhs.m[3] + self.m[5] * rhs.m[6],
            self.m[3] * rhs.m[1] + self.m[4] * rhs.m[4] + self.m[5] * rhs.m[7],
            self.m[3] * rhs.m[2] + self.m[4] * rhs.m[5] + self.m[5] * rhs.m[8],
            self.m[6] * rhs.m[0] + self.m[7] * rhs.m[3] + self.m[8] * rhs.m[6],
            self.m[6] * rhs.m[1] + self.m[7] * rhs.m[4] + self.m[8] * rhs.m[7],
            self.m[6] * rhs.m[2] + self.m[7] * rhs.m[5] + self.m[8] * rhs.m[8],
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
    pub fn and_then(&self, next: &Transform2<S>) -> Transform2<S> {
        let forward = &next.forward * &self.forward;
        let inverse = &self.inverse * &next.inverse;
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

impl<S: Scalar> Transform<Point2<S>> for Transform2<S> {
    type Output = Point2<S>;

    /// Transforms a point.
    #[inline]
    fn transform(&self, p: Point2<S>) -> Point2<S> {
        &self.forward * p
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

impl<S: Scalar> Transform<&Ray2<S>> for Transform2<S> {
    type Output = Ray2<S>;

    /// Transforms a ray.
    #[inline]
    fn transform(&self, ray: &Ray2<S>) -> Ray2<S> {
        Ray2::new(self.transform(ray.origin), self.transform(ray.direction))
    }
}

impl<S: Scalar> Transform<&BoundingBox2<S>> for Transform2<S> {
    type Output = BoundingBox2<S>;

    /// Transforms a bounding box.
    fn transform(&self, bb: &BoundingBox2<S>) -> BoundingBox2<S> {
        let o = self.transform(bb.min);
        let d = self.transform(bb.diagonal());

        let (mut min_corner, mut max_corner) = (o, o);
        for i in 1..=3 {
            let mut corner = o;
            if i & 0b01 != 0 { corner.x += d.x; }
            if i & 0b10 != 0 { corner.y += d.y; }

            min_corner = min(min_corner, corner);
            max_corner = max(max_corner, corner);
        }

        BoundingBox2::new(min_corner, max_corner)
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

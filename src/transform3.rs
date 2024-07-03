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

use crate::{BoundingBox3, CrossProduct, max, min, NonInvertibleMatrixError, Point3, Ray3, Scalar, Transform, Vector3};

/// Matrix with 4 rows and 4 columns for transformations in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix4x4<S: Scalar> {
    m: [S; 16],
}

/// Alias for `Matrix4x4<f32>`.
pub type Matrix4x4f = Matrix4x4<f32>;

/// Alias for `Matrix4x4<f64>`.
pub type Matrix4x4d = Matrix4x4<f64>;

/// Transform for transformations in 3D space.
#[derive(Clone, PartialEq, Debug)]
pub struct Transform3<S: Scalar> {
    pub forward: Matrix4x4<S>,
    pub inverse: Matrix4x4<S>,
}

/// Alias for `Transform3<f32>`.
pub type Transform3f = Transform3<f32>;

/// Alias for `Transform3<f64>`.
pub type Transform3d = Transform3<f64>;

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

    /// Returns a rotation matrix which rotates around the Z axis.
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
        debug_assert!(row < 4, "Invalid row index: {}", row);
        debug_assert!(col < 4, "Invalid column index: {}", col);
        (row * 4 + col) as usize
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
            Ok(&adjugate / det)
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

impl<S: Scalar> MulAssign<S> for &mut Matrix4x4<S> {
    #[inline]
    fn mul_assign(&mut self, s: S) {
        for m in &mut self.m { *m *= s; }
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
    pub fn and_then(&self, next: &Transform3<S>) -> Transform3<S> {
        let forward = &next.forward * &self.forward;
        let inverse = &self.inverse * &next.inverse;
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

impl<S: Scalar> Transform<Point3<S>> for Transform3<S> {
    type Output = Point3<S>;

    /// Transforms a point.
    #[inline]
    fn transform(&self, p: Point3<S>) -> Point3<S> {
        &self.forward * p
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

impl<S: Scalar> Transform<&Ray3<S>> for Transform3<S> {
    type Output = Ray3<S>;

    /// Transforms a ray.
    #[inline]
    fn transform(&self, ray: &Ray3<S>) -> Ray3<S> {
        Ray3::new(self.transform(ray.origin), self.transform(ray.direction))
    }
}

impl<S: Scalar> Transform<&BoundingBox3<S>> for Transform3<S> {
    type Output = BoundingBox3<S>;

    /// Transforms a bounding box.
    fn transform(&self, bb: &BoundingBox3<S>) -> BoundingBox3<S> {
        let o = self.transform(bb.min);
        let d = self.transform(bb.diagonal());

        let (mut min_corner, mut max_corner) = (o, o);
        for i in 1..=7 {
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

impl<S: Scalar> TryFrom<Matrix4x4<S>> for Transform3<S> {
    type Error = NonInvertibleMatrixError;

    #[inline]
    fn try_from(forward: Matrix4x4<S>) -> Result<Transform3<S>, NonInvertibleMatrixError> {
        let inverse = forward.inverse()?;
        Ok(Transform3::new(forward, inverse))
    }
}

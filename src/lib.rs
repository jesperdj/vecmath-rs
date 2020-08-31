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

//! Vector math for 2D and 3D graphics applications in Rust.
//!
//! # Main components and design decisions
//!
//! There are both 2D and 3D `Point`, `Vector` and `Normal` structs, which represent a point, vector and normal in 2D or 3D space.
//!
//! There are 3x3 and 4x4 matrices for transforming 2D and 3D points, vectors and normals.
//!
//! A `Transform` combines two matrices to represent a transform and its inverse. Keeping a transformation matrix and its inverse together in a `Transform`
//! makes it easy to get the inverse transformation without requiring matrix inversions, which can be computationally costly and prone to numeric precision
//! problems.
//!
//! All structs have `f32` components. I could have chosen to abstract away the component type using a type alias or by using generics, but this would have
//! made the implementation much more complex and isn't needed for most graphics applications. I also expect that restricting the component type to `f32`
//! will make it easier to optimize the implementation with SIMD intrinsics.
//!
//! ## Points and vectors
//!
//! There are separate structs for points and vectors, because the semantics for points and vectors are different.
//!
//! A point is a point in Euclidean space, relative to an origin. A vector consists of an "arrow" which is not relative to an origin. The difference between
//! points and vectors has consequences when points and vectors are transformed in homogeneous coordinates: for a point, the fourth element is considered
//! to be always 1, while for a vector, it is considered to be always 0. This means that for example a translation does have an effect on a point, but not
//! on a vector.
//!
//! ## Normals
//!
//! There is a separate struct for normal vectors (to be used, for example, for surface normals on a 3D surface), because the way vectors and normals are
//! transformed, is different. To keep the transformed normal perpendicular to the surface, it must be transformed by the transpose of the inverse.

pub use three_d::*;
pub use two_d::*;

mod two_d;
mod three_d;

/// Returned when a matrix cannot be inverted.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct NonInvertibleMatrixError;

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

use crate::Scalar;
use std::cmp::Ordering;

/// Trait for types for which a length can be computed.
pub trait Length<S: Scalar>: Copy {
    /// Computes and returns the length of a value.
    fn length(self) -> S;
}

/// Trait for types for which lengths can be compared.
pub trait RelativeLength<S: Scalar>: Length<S> {
    /// Compares the length of this value with the length of `other` and returns an `Ordering`.
    ///
    /// # Example
    /// ```
    /// use std::cmp::Ordering;
    /// use vecmath::{RelativeLength, Vector2};
    ///
    /// let v1 = Vector2::new(3.0, 4.0);
    /// let v2 = Vector2::new(-2.0, 3.5);
    ///
    /// match v1.cmp_length(v2) {
    ///     Ordering::Less => println!("v1 is shorter than v2"),
    ///     Ordering::Equal => println!("v1 has the same length as v2"),
    ///     Ordering::Greater => println!("v1 is longer than v2"),
    /// }
    /// ```
    fn cmp_length(self, other: Self) -> Ordering;

    /// Returns `true` if this value is shorter than `other`, `false` otherwise.
    fn is_shorter_than(self, other: Self) -> bool {
        self.cmp_length(other) == Ordering::Less
    }

    /// Returns `true` if this value is longer than `other`, `false` otherwise.
    fn is_longer_than(self, other: Self) -> bool {
        self.cmp_length(other) == Ordering::Greater
    }
}

/// Compares the length of two values and returns the shortest one.
///
/// If `first` and `second` are equally long, `first` is returned.
pub fn shortest<T: RelativeLength<S>, S: Scalar>(first: T, second: T) -> T {
    if first.is_longer_than(second) { second } else { first }
}

/// Compares the length of two values and returns the longest one.
///
/// If `first` and `second` are equally long, `first` is returned.
pub fn longest<T: RelativeLength<S>, S: Scalar>(first: T, second: T) -> T {
    if first.is_shorter_than(second) { second } else { first }
}

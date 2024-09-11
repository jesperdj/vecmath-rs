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

/// Trait for types for which a distance between values can be computed.
pub trait Distance<S: Scalar>: Copy {
    /// Computes and returns the distance between two values.
    fn distance(self, other: Self) -> S;
}

/// Trait for types for which distances between values can be compared.
pub trait RelativeDistance<S: Scalar>: Distance<S> {
    /// Compares the distance between this value and `first` and this value and `second` and returns an `Ordering`.
    ///
    /// # Example
    /// ```
    /// use std::cmp::Ordering;
    /// use vecmath::{Point2, RelativeDistance};
    ///
    /// let p1 = Point2::new(1.0, 2.0);
    /// let p2 = Point2::new(-1.0, 4.0);
    /// let p3 = Point2::new(4.0, 3.0);
    ///
    /// match p1.cmp_distance(p2, p3) {
    ///     Ordering::Less => println!("p2 is closer to p1 than p3"),
    ///     Ordering::Equal => println!("p2 and p3 are equally distant from p1"),
    ///     Ordering::Greater => println!("p2 is farther from p1 than p3"),
    /// }
    /// ```
    fn cmp_distance(self, first: Self, second: Self) -> Ordering;

    /// Compares the distance between this value and `first` and this value and `second` and returns the closest one.
    ///
    /// If `first` and `second` are equally far, `first` is returned.
    fn closest(self, first: Self, second: Self) -> Self {
        if self.cmp_distance(first, second) == Ordering::Greater { second } else { first }
    }

    /// Compares the distance between this value and `first` and this value and `second` and returns the farthest one.
    ///
    /// If `first` and `second` are equally far, `first` is returned.
    fn farthest(self, first: Self, second: Self) -> Self {
        if self.cmp_distance(first, second) == Ordering::Less { second } else { first }
    }
}

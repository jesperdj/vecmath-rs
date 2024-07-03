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

/// Trait for types for which a distance between values can be computed.
pub trait Distance: Copy {
    /// The output type which expresses the distance between values.
    type Output;

    /// Computes and returns the distance between two values.
    fn distance(self, other: Self) -> Self::Output;
}

/// Computes and returns the distance between two values.
#[inline]
pub fn distance<T: Distance>(value: T, other: T) -> T::Output {
    value.distance(other)
}

/// Trait for types for which distances between values can be compared.
pub trait RelativeDistance: Copy {
    /// Returns `true` if `a` is closer to this value than `b`.
    fn is_closer_to(self, a: Self, b: Self) -> bool;

    /// Returns `true` if `a` is farther from this value than `b`.
    fn is_farther_from(self, a: Self, b: Self) -> bool;

    /// Checks which of the values `a` and `b` is closer to this value and returns the closest one.
    fn closest(self, a: Self, b: Self) -> Self;

    /// Checks which of the values `a` and `b` is farther from this value and returns the farthest one.
    fn farthest(self, a: Self, b: Self) -> Self;
}

/// Checks which of the values `a` and `b` is closer to `value` and returns the closest one.
#[inline]
pub fn closest<T: RelativeDistance>(value: T, a: T, b: T) -> T {
    value.closest(a, b)
}

/// Checks which of the values `a` and `b` is farther from `value` and returns the farthest one.
#[inline]
pub fn farthest<T: RelativeDistance>(value: T, a: T, b: T) -> T {
    value.farthest(a, b)
}

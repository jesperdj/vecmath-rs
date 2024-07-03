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

/// Trait for types for which a length can be computed.
pub trait Length: Copy {
    /// The output type which expresses the length of a value.
    type Output;

    /// Computes and returns the length of a value.
    fn length(self) -> Self::Output;
}

/// Computes and returns the length of a value.
#[inline]
pub fn length<T: Length>(value: T) -> T::Output {
    value.length()
}

/// Trait for types for which lengths can be compared.
pub trait RelativeLength: Copy {
    /// Returns `true` if this value is shorter than the other value.
    fn is_shorter_than(self, other: Self) -> bool;

    /// Returns `true` if this value is longer than the other value.
    fn is_longer_than(self, other: Self) -> bool;

    /// Returns the shortest of two values.
    fn shortest(self, other: Self) -> Self;

    /// Returns the longest of two values.
    fn longest(self, other: Self) -> Self;
}

/// Returns the shortest of two values.
#[inline]
pub fn shortest<T: RelativeLength>(value: T, other: T) -> T {
    value.shortest(other)
}

/// Returns the longest of two values.
#[inline]
pub fn longest<T: RelativeLength>(value: T, other: T) -> T {
    value.longest(other)
}

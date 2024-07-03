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

use num_traits::{Float, NumAssign};

use crate::{Distance, Length, MinMax, RelativeDistance, RelativeLength};

/// Scalar value.
pub trait Scalar: Float + NumAssign + ScalarConst {}

/// Extra constants that are not defined in `num_traits::float::FloatConst`.
pub trait ScalarConst {
    /// Returns the representation of `0.5` of the implementing type.
    fn half() -> Self;

    /// Returns the representation of `2.0` of the implementing type.
    fn two() -> Self;
}

// ===== Scalar ================================================================================================================================================

impl Scalar for f32 {}

impl Scalar for f64 {}

impl ScalarConst for f32 {
    #[inline(always)]
    fn half() -> f32 {
        0.5f32
    }

    #[inline(always)]
    fn two() -> f32 {
        2.0f32
    }
}

impl ScalarConst for f64 {
    #[inline(always)]
    fn half() -> f64 {
        0.5f64
    }

    #[inline(always)]
    fn two() -> f64 {
        2.0f64
    }
}

impl<S: Scalar> MinMax for S {
    #[inline]
    fn min(self, s: S) -> S {
        S::min(self, s)
    }

    #[inline]
    fn max(self, s: S) -> S {
        S::max(self, s)
    }
}

impl<S: Scalar> Length for S {
    type Output = S;

    #[inline]
    fn length(self) -> S {
        self.abs()
    }
}

impl<S: Scalar> RelativeLength for S {
    #[inline]
    fn is_shorter_than(self, s: S) -> bool {
        self.length() < s.length()
    }

    #[inline]
    fn is_longer_than(self, s: S) -> bool {
        self.length() > s.length()
    }

    #[inline]
    fn shortest(self, s: S) -> Self {
        if self.is_shorter_than(s) { self } else { s }
    }

    #[inline]
    fn longest(self, s: S) -> Self {
        if self.is_longer_than(s) { self } else { s }
    }
}

impl<S: Scalar> Distance for S {
    type Output = S;

    #[inline]
    fn distance(self, s: S) -> S {
        (s - self).length()
    }
}

impl<S: Scalar> RelativeDistance for S {
    #[inline]
    fn is_closer_to(self, s1: S, s2: S) -> bool {
        self.distance(s1) < self.distance(s2)
    }

    #[inline]
    fn is_farther_from(self, s1: S, s2: S) -> bool {
        self.distance(s1) > self.distance(s2)
    }

    #[inline]
    fn closest(self, s1: S, s2: S) -> S {
        if self.is_closer_to(s1, s2) { s1 } else { s2 }
    }

    #[inline]
    fn farthest(self, s1: S, s2: S) -> S {
        if self.is_farther_from(s1, s2) { s1 } else { s2 }
    }
}

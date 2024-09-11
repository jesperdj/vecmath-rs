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
use num_traits::{ConstZero, Zero};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

#[cfg(feature = "rand")]
use rand::{distributions::Standard, prelude::Distribution, Rng};

/// An angle, specified in either radians or degrees.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub enum Angle<S: Scalar> {
    /// An angle in radians.
    Radians(S),

    /// An angle in degrees.
    Degrees(S),
}

// ===== Angle =================================================================================================================================================

impl<S: Scalar> Angle<S> {
    /// Returns the value of this angle in radians.
    pub fn radians(self) -> S {
        match self {
            Angle::Radians(rad) => rad,
            Angle::Degrees(deg) => deg.to_radians(),
        }
    }

    /// Returns the value of this angle in degrees.
    pub fn degrees(self) -> S {
        match self {
            Angle::Radians(rad) => rad.to_degrees(),
            Angle::Degrees(deg) => deg,
        }
    }

    /// Returns this angle as an instance of the `Radians` variant.
    pub fn as_radians(self) -> Angle<S> {
        Angle::Radians(self.radians())
    }

    /// Returns this angle as an instance of the `Degrees` variant.
    pub fn as_degrees(self) -> Angle<S> {
        Angle::Degrees(self.degrees())
    }
}

impl<S: Scalar> Zero for Angle<S> {
    /// Returns an angle of zero radians.
    fn zero() -> Angle<S> {
        Angle::ZERO
    }

    /// Returns `true` if this angle is zero (either radians or degrees), `false` otherwise.
    fn is_zero(&self) -> bool {
        match self {
            Angle::Radians(rad) => rad.is_zero(),
            Angle::Degrees(deg) => deg.is_zero(),
        }
    }
}

impl<S: Scalar> ConstZero for Angle<S> {
    /// A constant angle of zero radians.
    const ZERO: Angle<S> = Angle::Radians(S::ZERO);
}

impl<S: Scalar> Add for Angle<S> {
    type Output = Angle<S>;

    /// Adds two angles.
    ///
    /// The result will be in the units (radians or degrees) of the left hand side.
    fn add(self, other: Angle<S>) -> Angle<S> {
        match self {
            Angle::Radians(rad) => Angle::Radians(rad + other.radians()),
            Angle::Degrees(deg) => Angle::Degrees(deg + other.degrees()),
        }
    }
}

impl<S: Scalar> AddAssign for Angle<S> {
    /// Adds an angle to this angle.
    fn add_assign(&mut self, other: Angle<S>) {
        match self {
            Angle::Radians(rad) => *rad += other.radians(),
            Angle::Degrees(deg) => *deg += other.degrees(),
        }
    }
}

impl<S: Scalar> Sub for Angle<S> {
    type Output = Angle<S>;

    /// Subtracts two angles.
    ///
    /// The result will be in the units (radians or degrees) of the left hand side.
    fn sub(self, other: Angle<S>) -> Angle<S> {
        match self {
            Angle::Radians(rad) => Angle::Radians(rad - other.radians()),
            Angle::Degrees(deg) => Angle::Degrees(deg - other.degrees()),
        }
    }
}

impl<S: Scalar> SubAssign for Angle<S> {
    /// Subtracts an angle from this angle.
    fn sub_assign(&mut self, other: Angle<S>) {
        match self {
            Angle::Radians(rad) => *rad -= other.radians(),
            Angle::Degrees(deg) => *deg -= other.degrees(),
        }
    }
}

impl<S: Scalar> Neg for Angle<S> {
    type Output = Angle<S>;

    /// Negates this angle.
    fn neg(self) -> Angle<S> {
        match self {
            Angle::Radians(rad) => Angle::Radians(-rad),
            Angle::Degrees(deg) => Angle::Degrees(-deg),
        }
    }
}

impl<S: Scalar> Mul<S> for Angle<S> {
    type Output = Angle<S>;

    /// Multiplies this angle with a scalar value.
    fn mul(self, value: S) -> Angle<S> {
        match self {
            Angle::Radians(rad) => Angle::Radians(rad * value),
            Angle::Degrees(deg) => Angle::Degrees(deg * value),
        }
    }
}

impl<S: Scalar> MulAssign<S> for Angle<S> {
    /// Multiplies this angle with a scalar value.
    fn mul_assign(&mut self, value: S) {
        match self {
            Angle::Radians(rad) => *rad *= value,
            Angle::Degrees(deg) => *deg *= value,
        }
    }
}

impl<S: Scalar> Div<S> for Angle<S> {
    type Output = Angle<S>;

    /// Divides this angle by a scalar value.
    fn div(self, value: S) -> Angle<S> {
        match self {
            Angle::Radians(rad) => Angle::Radians(rad / value),
            Angle::Degrees(deg) => Angle::Degrees(deg / value),
        }
    }
}

impl<S: Scalar> DivAssign<S> for Angle<S> {
    /// Divides this angle by a scalar value.
    fn div_assign(&mut self, value: S) {
        match self {
            Angle::Radians(rad) => *rad /= value,
            Angle::Degrees(deg) => *deg /= value,
        }
    }
}

impl<S: Scalar> Rem<S> for Angle<S> {
    type Output = Angle<S>;

    /// Computes the remainder (the `%` operation) on an angle with a scalar value.
    fn rem(self, value: S) -> Angle<S> {
        match self {
            Angle::Radians(rad) => Angle::Radians(rad % value),
            Angle::Degrees(deg) => Angle::Degrees(deg % value),
        }
    }
}

impl<S: Scalar> RemAssign<S> for Angle<S> {
    /// Computes the remainder (the `%` operation) on an angle with a scalar value.
    fn rem_assign(&mut self, value: S) {
        match self {
            Angle::Radians(rad) => *rad %= value,
            Angle::Degrees(deg) => *deg %= value,
        }
    }
}

#[cfg(feature = "rand")]
impl<S: Scalar> Distribution<Angle<S>> for Standard
where
    Standard: Distribution<S>,
{
    /// Generates a random angle between 0 and 2π radians (uniformly distributed).
    ///
    /// # Example
    /// ```
    /// use vecmath::Angle;
    /// use rand::random;
    ///
    /// let angle: Angle<f64> = random();
    /// println!("{:?}", angle);
    /// ```
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Angle<S> {
        Angle::Radians(rng.gen() * S::TAU())
    }
}

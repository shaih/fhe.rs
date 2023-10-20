#![warn(missing_docs, unused_imports)]
//! Polynomials in R_q\[x\] where the q is a single-precision prime modulus

mod ops;

use sha2::{Digest, Sha256};

use crate::{ntt::NttOperator, Error, Result};
use crate::zq::Modulus;

use fhe_util::sample_vec_cbd;
use itertools::Itertools;
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};

/// Possible representations of the underlying polynomial.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub enum Representation {
    /// This is the list of coefficients ci, such that the polynomial is c0 + c1
    /// * x + ... + c_(degree - 1) * x^(degree - 1)
    #[default]
    PowerBasis,
    /// This is the NTT representation of the PowerBasis representation.
    Ntt,
    /// This is a "Shoup" representation of the Ntt representation used for
    /// faster multiplication.
    NttShoup,
}

/// Parameters of a polynomial modulo some q
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Zqcontext {
    /// The polynomial degree
    pub degree: usize,

    /// The integer modulus
    pub q: Box<Modulus>,

    /// operator to performs NTT and iNTT
    pub ops: Box<NttOperator>,
    // Note: q, ops are in boxes just to make initialization easier

    // the numbers 0..degree-1, all bit-reversed
    bitrev: Vec<usize>
}

impl Zqcontext {
    /// Creates a Zqcontext for a modulus and a polynomial degree.
    ///
    /// The modulus must be a primes < 2^62, supporting NTT of size `degree`.
    pub fn new(modulus: u64, degree: usize) -> Result<Self> {
        if !degree.is_power_of_two() || degree < 8 {
            Err(Error::Default(
                "The degree must be a power of two, and >= 8".to_string()))
        } else {
            let q = Modulus::new(modulus)?;
            let ops = NttOperator::new(&q, degree)
                .ok_or(Error::Default(
                    "Impossible to construct a Ntt operator".to_string()))?;
            Ok( Self {
                degree,
                q: Box::<Modulus>::new(q),
                ops: Box::<NttOperator>::new(ops),
                bitrev: (0..degree)
                    .map(|j| j.reverse_bits() >> (degree.leading_zeros() + 1))
                    .collect_vec()
            })
        }
    }

    /// Read-only access to the single-precision modulus
	pub fn modulus(&self) -> u64 {
		self.q.modulus()
	}
}


/// Helper for computing substitution F(X) -> F(X^k) mod (X^d+1)
#[derive(Debug, PartialEq, Eq)]
pub struct SubstitutionExponent {
    /// The value of the exponent (the integer k in F(X^k))
    pub exponent: usize,

    // indexes in bit-reverse, ordered as k/2,3k/2,5k/2,... (mod d)
    power_bitrev: Vec<usize>, 
}

impl SubstitutionExponent {
    /// Creates a substitution element for computing F(X) -> F(X^k).
    /// Returns an error if the exponent k is even (since it isn't
    /// supported in NTT representation)
    pub fn new(exponent: usize, degree: usize) -> Result<Self> {
        let exponent = exponent % (2 * degree);
        if exponent & 1 == 0 {
            return Err(Error::Default(
                "The exponent k for F(X)->F(X^k) should be odd".to_string(),
            ));
        }
        // Compute power_bitrev, a list of indexes in bit-reversed representation
        // Setting x = (k-1)/2, The indexes are ordered as
        //             rev(x), rev(x+k mod d), rev(x+2k mod d), ...
        // Example: with degree=8 and exponent=3, power_bitrev will be
        // (rev(1),rev(4),rev(7),rev(2),rev(5),rev(0),rev(3),rev(6))
        // = (4, 1, 7, 2, 5, 0, 6, 3)

        let mut power = (exponent - 1) / 2;
        let mut power_bitrev = Vec::<usize>::with_capacity(degree);

        // reduction modulo 'degree' can be done just by masking.
        let mask = degree - 1;
        for _ in 0..degree {
            power_bitrev.push(power.reverse_bits() >> (degree.leading_zeros() + 1));
            power = (power + exponent) & mask;
        }
        Ok(Self {
            // ctx: ctx.clone(),
            exponent,
            power_bitrev,
        })
    }
}

/// Struct that holds a polynomial fmodulo a single-precision number q.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZqX {
    ctx: Arc<Zqcontext>,              // parameters and pre-computed tables
    representation: Representation,   // coeff, ntt, or shoup-ntt
    has_lazy_coefficients: bool,   // see below   
    allow_variable_time_computations: bool,
    coefficients: Vec<u64>,        // either coefficients or ntt coefficient
    coefficients_shoup: Option<Vec<u64>>, // floor(2^64*coefficients[..]/q)
}
/* has_lazy_coefficients: when multiplying poly1 in Ntt representation by
 * a poly2 in Ntt Shoup representation, the procedure stills work even if
 * the coefficients of poly1 are in [0, 4*q) instead of [0, q). So when
 * computing the NTT representation of poly1, there is no need to fully
 * reduce the coefficients to the range [0,q), it is sufficeint to reduce
 * them lazily to the range [0,4q). 
 */

impl AsRef<ZqX> for ZqX {
    fn as_ref(&self) -> &ZqX {
        self
    }
}

impl AsMut<ZqX> for ZqX {
    fn as_mut(&mut self) -> &mut ZqX {
        self
    }
}

impl ZqX {
    /// Creates a polynomial holding the constant 0.
    pub fn zero(ctx: &Arc<Zqcontext>, representation: Representation) -> Self {
        Self {
            ctx: ctx.clone(),
            representation: representation.clone(),
            allow_variable_time_computations: false,
            has_lazy_coefficients: false,
            coefficients: vec!(0_u64; ctx.degree),
            coefficients_shoup: if representation == Representation::NttShoup {
                Some(vec!(0_u64; ctx.degree))
            } else {
                None
            },
        }
    }

    /// Enable variable time computations when this polynomial is involved.
    ///
    /// # Safety
    ///
    /// By default, this is marked as unsafe, but is usually safe when only
    /// public data is processed.
    pub unsafe fn allow_variable_time_computations(&mut self) {
        self.allow_variable_time_computations = true
    }

    /// Disable variable time computations when this polynomial is involved.
    pub fn disallow_variable_time_computations(&mut self) {
        self.allow_variable_time_computations = false
    }

    /// Current representation of the polynomial.
    pub const fn representation(&self) -> &Representation {
        &self.representation
    }

    /// Change the representation of the underlying polynomial.
    pub fn change_representation(&mut self, to: Representation) {
        match self.representation {
            Representation::PowerBasis => {
                match to {
                    Representation::Ntt => self.ntt_forward(),
                    Representation::NttShoup => {
                        self.ntt_forward();
                        self.compute_coefficients_shoup();
                    }
                    Representation::PowerBasis => {} // no-op
                }
            }
            Representation::Ntt => {
                match to {
                    Representation::PowerBasis => self.ntt_backward(),
                    Representation::NttShoup => self.compute_coefficients_shoup(),
                    Representation::Ntt => {} // no-op
                }
            }
            Representation::NttShoup => {
                if to != Representation::NttShoup {
                    // We are not sure whether this polynomial was sensitive or not,
                    // so for security, we zeroize the Shoup coefficients.
                    self.coefficients_shoup.as_mut().unwrap()
                        .zeroize();
                    self.coefficients_shoup = None
                }
                match to {
                    Representation::PowerBasis => self.ntt_backward(),
                    Representation::Ntt => {}      // no-op
                    Representation::NttShoup => {} // no-op
                }
            }
        }

        self.representation = to;
    }

    /// Compute the Shoup representation = floor(2^64 * coefficients / q)
    fn compute_coefficients_shoup(&mut self) {
        self.coefficients_shoup  = Some(self.ctx.q.shoup_vec(&self.coefficients));
    }

    /// Override the internal representation to a given representation.
    ///
    /// # Safety
    ///
    /// Prefer the `change_representation` function to safely modify the
    /// polynomial representation. If the `to` representation is NttShoup, the
    /// coefficients are still computed correctly to avoid being in an unstable
    /// state. Similarly, if we override a representation which was NttShoup, we
    /// zeroize the existing Shoup coefficients.
    pub unsafe fn override_representation(&mut self, to: Representation) {
        if to == Representation::NttShoup {
            self.compute_coefficients_shoup()
        } else if self.coefficients_shoup.is_some() {
            self.coefficients_shoup.as_mut().unwrap().zeroize();
            self.coefficients_shoup = None
        }
        self.representation = to;
    }

    /// Generate a random polynomial.
    pub fn random<R: RngCore + CryptoRng>(
        ctx: &Arc<Zqcontext>,
        representation: Representation,
        rng: &mut R,
    ) -> Self {
        let mut p = Self {
            ctx: ctx.clone(),
            representation: representation.clone(),
            allow_variable_time_computations: false,
            has_lazy_coefficients: false,
            coefficients: ctx.q.random_vec(ctx.degree, rng),
            coefficients_shoup: None
        };
        if p.representation == Representation::NttShoup {
            p.compute_coefficients_shoup();
        }
        p
    }

    /// Generate a random polynomial deterministically from a seed.
    pub fn random_from_seed(
        ctx: &Arc<Zqcontext>,
        representation: Representation,
        seed: <ChaCha8Rng as SeedableRng>::Seed,
    ) -> Self {
        // Hash the seed into a ChaCha8Rng seed.
        let mut hasher = Sha256::new();
        hasher.update(seed);
        let mut prng =
            ChaCha8Rng::from_seed(<ChaCha8Rng as SeedableRng>::Seed::from(hasher.finalize()));
            /*
            let mut p = Self {
                ctx: ctx.clone(),
                representation: representation.clone(),
                allow_variable_time_computations: false,
                has_lazy_coefficients: false,
                coefficients: ctx.q.random_vec(ctx.degree, &mut prng),
                coefficients_shoup: None
            };
            if p.representation == Representation::NttShoup {
                p.compute_coefficients_shoup();
            }
            p
            */
            Self::random(ctx, representation, &mut prng)
    }

    /// Generate a small poly and convert to the specified representation
    ///
    /// Returns an error if the variance is not in [1, ..., 16].
    pub fn small<T: RngCore + CryptoRng>(
        ctx: &Arc<Zqcontext>,
        representation: Representation,
        variance: usize,
        rng: &mut T,
    ) -> Result<Self> {
        if !(1..=16).contains(&variance) {
            Err(Error::Default(
                "The variance should be an integer between 1 and 16".to_string(),
            ))
        } else {
            let coeffs = Zeroizing::new(
                sample_vec_cbd(ctx.degree, variance, rng)
                    .map_err(|e| Error::Default(e.to_string()))?,
            );
            let mut p = try_convert_from(
                coeffs.as_slice(),
                ctx,
                false,
                Representation::PowerBasis,
            )?;
            if representation != Representation::PowerBasis {
                p.change_representation(representation);
            }
            Ok(p)
        }
    }

    /// Access to the polynomial coefficients.
    pub fn coefficients(&self) -> &Vec<u64> {
        &self.coefficients
    }

    /// Computes the forward Ntt on the coefficients
    fn ntt_forward(&mut self) {
        let ops = &*self.ctx.ops;
        if self.allow_variable_time_computations {
            unsafe { ops.forward_vt(self.coefficients.as_mut_ptr()) };
        } else {
            ops.forward(self.coefficients.as_mut_slice());
        }
    }

    /// Computes the backward Ntt on the coefficients
    fn ntt_backward(&mut self) {
        let ops = &*self.ctx.ops;
        if self.allow_variable_time_computations {
            unsafe { ops.backward_vt(self.coefficients.as_mut_ptr()) };
        } else {
            ops.backward(self.coefficients.as_mut_slice());
        }
    }

    /// Automorphism: Convert F(X) to F(X^k) mod (X^d +1).
    /// In PowerBasis representation, k can be any integer < 2*degree.
    /// In Ntt/NttShoup representation, k can be any odd integer < 2*degree.
    // SHAI: Why does k have to be odd for Ntt representation?
    pub fn substitute(&self, k: &SubstitutionExponent) -> Result<ZqX> {
        let mut poly = ZqX::zero(&self.ctx, self.representation.clone());
        if self.allow_variable_time_computations {
            unsafe { poly.allow_variable_time_computations() }
        }
        // Permute the coefficient vector
        match self.representation {
            Representation::Ntt => {
                for (i,j) in self.ctx.bitrev.iter().zip(k.power_bitrev.iter()) {
                    poly.coefficients[*i] = self.coefficients[*j];
                }
            }
            // Permute both coefficients and shoup-representation together
            Representation::NttShoup => {
                let this_shoup = self.coefficients_shoup.as_ref().unwrap();
                let other_shoup = poly.coefficients_shoup.as_mut().unwrap();
                for (i,j) in self.ctx.bitrev.iter().zip(k.power_bitrev.iter()) {
                    poly.coefficients[*i] = self.coefficients[*j];
                    other_shoup[*i] = this_shoup[*j];
                }
            }
            /* Substitution algorithm for F(X) -> F(X^k) mod X^d +1 in the power basis,
             * when d is a power of two and 0 < k < 2d:
             * Start from an all-zero output array c'[....].
             * Then go over the input coefficients in order. Add or subtract each input
             * coefficient c[i] from the corresponding output coefficient c'[i*j mod d]:
             *             c'[i*j mod d] = c'[i*j mod d] (+ or -) c[i].
             * c[i] is added when i*j mod d has even numbers of wraparounds (i.e., if
             * (i*j) div d is even), and is subtracted otherwise.
             */
            Representation::PowerBasis => {
                let mut add = true;
                let invert = k.exponent >= self.ctx.degree;
                let mut iprime = 0usize;

                // degree is a power of 2, so modular addition can be done using masking
                let mask = self.ctx.degree -1;

                // Using "old-style" loop, to make it clearer what's going on
                for i in 0..self.ctx.degree {
                    if add {
                        poly.coefficients[iprime] = self.ctx.q.add(
                            poly.coefficients[iprime], self.coefficients[i]);
                    }
                    else {
                        poly.coefficients[iprime] = self.ctx.q.sub(
                            poly.coefficients[iprime], self.coefficients[i]);
                    }
                    // The following assumes that 0 < k.exponent < 2*degree
                    let next = (iprime + k.exponent) & mask; // add modulo d
                    // Check for one or two wraparounds
                    add ^= (next <= iprime) ^ invert;
                    iprime = next;
                }
            }
        }
        Ok(poly)
    }

    /// Create a polynomial which can only be multiplied by a polynomial
    /// in NttShoup representation. All other operations may panic.
    ///
    /// # Safety
    /// This creates a polynomial that allows variable time operations.
    pub unsafe fn create_constant_ntt_polynomial_with_lazy_coefficients_and_variable_time(
        power_basis_coefficients: &[u64],
        ctx: &Arc<Zqcontext>,
    ) -> Self {
        let mut poly = Self {
            ctx: ctx.clone(),
            representation: Representation::Ntt,
            allow_variable_time_computations: true,
            coefficients: power_basis_coefficients.to_vec(),
            coefficients_shoup: None,
            has_lazy_coefficients: true,
        };
        ctx.q.lazy_reduce_vec(&mut poly.coefficients);   // reduce to [0,2q)
        ctx.ops.forward_vt_lazy(poly.coefficients.as_mut_ptr()); // Lazy NTT
        poly
    }

    /// Returns the Zqcontext of the underlying polynomial
    pub fn ctx(&self) -> &Arc<Zqcontext> {
        &self.ctx
    }

    /// Multiplies a polynomial in PowerBasis representation by x^(-power).
    pub fn multiply_inverse_power_of_x(&mut self, power: usize) -> Result<()> {
        if self.representation != Representation::PowerBasis {
            return Err(Error::IncorrectRepresentation(
                self.representation.clone(),
                Representation::PowerBasis,
            ));
        }
        let shift = ((self.ctx.degree << 1) - power) % (self.ctx.degree << 1);
        let mask = self.ctx.degree - 1;
        let original_coefficients = self.coefficients.clone();
        for k in 0..self.ctx.degree {
            let index = shift + k;
            if index & self.ctx.degree == 0 {
                self.coefficients[index & mask] = original_coefficients[k];
            } else {
                self.coefficients[index & mask] = self.ctx.q.neg(original_coefficients[k]);
            }
        }
        Ok(())
    }
    
}

fn try_convert_from<'a>(
    v: &'a [i64],
    ctx: &Arc<Zqcontext>,
    variable_time: bool,
    representation: Representation) -> Result<ZqX> {
    
    if representation != Representation::PowerBasis {
        Err(Error::Default(
            "Converting signed integer require to import in PowerBasis representation"
                .to_string(),
        ))
    } else if v.len() <= ctx.degree {
        let mut out = ZqX::zero(ctx, Representation::PowerBasis);
        if variable_time {
            unsafe { out.allow_variable_time_computations() }
        }
        if variable_time {
            unsafe { out.coefficients[..v.len()].copy_from_slice(&ctx.q.reduce_vec_i64_vt(v)) }
        } else {
            out.coefficients[..v.len()].copy_from_slice(Zeroizing::new(ctx.q.reduce_vec_i64(v)).as_ref());
        };
        Ok(out)
    } else {
        Err(Error::Default("In PowerBasis representation with signed integers, only `degree` coefficients can be specified".to_string()))
    }
}

impl Zeroize for ZqX {
    fn zeroize(&mut self) {
        self.coefficients.zeroize();
        if let Some(s) = self.coefficients_shoup.as_mut() {
            s.zeroize();
        }
    }
}

impl From<&ZqX> for Vec<u64> {
    fn from(p: &ZqX) -> Self {
        let mut coefs = p.coefficients.clone();
        if p.has_lazy_coefficients {
            let q = &*p.ctx.q;
            q.reduce_vec(coefs.as_mut_slice());
        }
        coefs
    }
}

impl From<&ZqX> for Vec<i64> {
    fn from(p: &ZqX) -> Self {
        let coefs: Vec<u64> = p.into();
        // FIXME: maybe use a non-VT implementation?
        unsafe { p.ctx.q.center_vec_vt(coefs.as_slice()) }
    }
}


#[cfg(test)]
mod tests {
    use super::Representation;
    use crate::zqx::{Zqcontext, try_convert_from, SubstitutionExponent, ZqX};
    //use fhe_util::variance;
    //use itertools::Itertools;
    //use num_traits::Zero;
    use rand::{thread_rng, Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    //use core::slice::SlicePattern;
    use std::{error::Error, sync::Arc};

    // Moduli to be used in tests.
    const MODULI: &[u64; 5] = &[
        1153,
        4611686018326724609,
        4611686018309947393,
        4611686018232352769,
        4611686018171535361,
    ];

    #[test]
    fn poly_zero() -> Result<(), Box<dyn Error>> {
        // let reference = &[
        //     BigUint::zero(),
        //     BigUint::zero(),
        //     BigUint::zero(),
        //     BigUint::zero(),
        //     BigUint::zero(),
        //     BigUint::zero(),
        //     BigUint::zero(),
        //     BigUint::zero(),
        // ];

        for modulus in MODULI {
            let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
            let p = ZqX::zero(&ctx, Representation::PowerBasis);
            let q = ZqX::zero(&ctx, Representation::Ntt);
            assert_ne!(p, q);
            assert_eq!(&p.coefficients, &[0; 8]);
            assert_eq!(&q.coefficients, &[0; 8]);
        }
        Ok(())
    }

    #[test]
    fn ctx() -> Result<(), Box<dyn Error>> {
        for modulus in MODULI {
            let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
            let p = ZqX::zero(&ctx, Representation::PowerBasis);
            assert_eq!(p.ctx(), &ctx);
        }
        Ok(())
    }

    #[test]
    fn random() -> Result<(), Box<dyn Error>> {
        //let mut rng = thread_rng();
        for _ in 0..100 {
            let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
            thread_rng().fill(&mut seed);

            let mut seed2 = <ChaCha8Rng as SeedableRng>::Seed::default();
            thread_rng().fill(&mut seed2);

            for modulus in MODULI {
                let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
                let p = ZqX::random_from_seed(&ctx, Representation::Ntt, seed);
                let q = ZqX::random_from_seed(&ctx, Representation::Ntt, seed);
                let r = ZqX::random_from_seed(&ctx, Representation::Ntt, seed2);
                assert_eq!(p, q);
                assert_ne!(p, r);
            }
        }
        Ok(())
    }

    #[test]
    fn coefficients() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for _ in 0..50 {
            for modulus in MODULI {
                let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
                let p = ZqX::random(&ctx, Representation::Ntt, &mut rng);
                let p_coefficients = Vec::<u64>::from(&p);
                assert_eq!(p_coefficients, *p.coefficients())
            }
        }
        Ok(())
    }

    #[test]
    fn modulus() -> Result<(), Box<dyn Error>> {
        for modulus in MODULI {
            let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
            assert_eq!(ctx.modulus(), *modulus)
        }
        Ok(())
    }

    #[test]
    fn allow_variable_time_computations() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for modulus in MODULI {
            let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
            let mut p = ZqX::random(&ctx, Representation::default(), &mut rng);
            assert!(!p.allow_variable_time_computations);

            unsafe { p.allow_variable_time_computations() }
            assert!(p.allow_variable_time_computations);

            let q = p.clone();
            assert!(q.allow_variable_time_computations);

            p.disallow_variable_time_computations();
            assert!(!p.allow_variable_time_computations);

            // Allowing variable time propagates.
            let mut p = ZqX::random(&ctx, Representation::Ntt, &mut rng);
            unsafe { p.allow_variable_time_computations() }
            let mut q = ZqX::random(&ctx, Representation::Ntt, &mut rng);

            assert!(!q.allow_variable_time_computations);
            q *= &p;
            assert!(q.allow_variable_time_computations);

            q.disallow_variable_time_computations();
            q += &p;
            assert!(q.allow_variable_time_computations);

            q.disallow_variable_time_computations();
            q -= &p;
            assert!(q.allow_variable_time_computations);

            q = -&p;
            assert!(q.allow_variable_time_computations);
        }
        Ok(())
    }


    #[test]
    fn change_representation() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for modulus in MODULI {
            let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);

            let mut p = ZqX::random(&ctx, Representation::default(), &mut rng);
            assert_eq!(p.representation, Representation::default());
            assert_eq!(p.representation(), &Representation::default());

            p.change_representation(Representation::PowerBasis);
            assert_eq!(p.representation, Representation::PowerBasis);
            assert_eq!(p.representation(), &Representation::PowerBasis);
            assert!(p.coefficients_shoup.is_none());

            let p_power = p.clone(); // Fixed at power basis

            p.change_representation(Representation::Ntt);
            assert_eq!(p.representation, Representation::Ntt);
            assert_eq!(p.representation(), &Representation::Ntt);
            // NOTE: This could fail with negligible probability, if p is zero
            assert_ne!(p.coefficients, p_power.coefficients);
            assert!(p.coefficients_shoup.is_none());

            let p_ntt = p.clone(); // Fixed at NTT representation

            p.change_representation(Representation::NttShoup);
            assert_eq!(p.representation, Representation::NttShoup);
            assert_eq!(p.representation(), &Representation::NttShoup);
            assert_ne!(p.coefficients, p_power.coefficients);
            assert!(p.coefficients_shoup.is_some());

            let p_ntt_shoup = p.clone(); // Fixed at Shoup representation

            p.change_representation(Representation::PowerBasis);
            assert_eq!(p, p_power);

            p.change_representation(Representation::NttShoup);
            assert_eq!(p, p_ntt_shoup);

            p.change_representation(Representation::Ntt);
            assert_eq!(p, p_ntt);

            p.change_representation(Representation::PowerBasis);
            assert_eq!(p, p_power);
        }
        Ok(())
    }

    #[test]
    fn override_representation() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for modulus in MODULI {
            let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
            let mut p = ZqX::random(&ctx, Representation::PowerBasis, &mut rng);
            assert_eq!(p.representation(), &p.representation);
            let q = p.clone();

            unsafe { p.override_representation(Representation::Ntt) }
            assert_eq!(p.representation, Representation::Ntt);
            assert_eq!(p.representation(), &p.representation);
            // This unsafe low-level routine doesn't change the coefficients
            assert_eq!(p.coefficients, q.coefficients);
            assert!(p.coefficients_shoup.is_none());

            unsafe { p.override_representation(Representation::NttShoup) }
            assert_eq!(p.representation, Representation::NttShoup);
            assert_eq!(p.representation(), &p.representation);
            assert_eq!(p.coefficients, q.coefficients);
            assert!(p.coefficients_shoup.is_some());

            unsafe { p.override_representation(Representation::PowerBasis) }
            assert_eq!(p, q);

            unsafe { p.override_representation(Representation::NttShoup) }
            assert!(p.coefficients_shoup.is_some());

            unsafe { p.override_representation(Representation::Ntt) }
            assert!(p.coefficients_shoup.is_none());
        }
        Ok(())
    }

    #[test]
    fn small() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for modulus in MODULI {
            let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);

            // Try a few unsupported variance setting and check that you get the right error
            for v in [0_usize, 17] {
                let e = ZqX::small(&ctx, Representation::PowerBasis, v, &mut rng);
                assert!(e.is_err());
                assert_eq!(
                    e.unwrap_err().to_string(),
                    "The variance should be an integer between 1 and 16"
                );
            }
            // Try the valid variance settings, check that the coefficients are small
            for i in 1..=16 {
                let p = ZqX::small(&ctx, Representation::PowerBasis, i, &mut rng)?;
                let coefficients = p.coefficients().as_slice();
                let v = unsafe { ctx.q.center_vec_vt(coefficients) };

                assert!(v.iter().map(|vi| vi.abs()).max().unwrap() <= 2 * i as i64);
            }
            // Check that you get some variations, even for the small-variance setting
            // NOTE: This can fail with a very small probability
            let e1 = ZqX::small(&ctx, Representation::PowerBasis, 1, &mut rng)?;
            let e2 = ZqX::small(&ctx, Representation::PowerBasis, 1, &mut rng)?;
            let e3 = ZqX::small(&ctx, Representation::PowerBasis, 1, &mut rng)?;
            let e4 = ZqX::small(&ctx, Representation::PowerBasis, 1, &mut rng)?;

            assert!(e1 != e2 || e2 != e3 || e3 != e4);
        }
        Ok(())
    }

    #[test]
    fn substitute() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        let degree = 8;
        for modulus in MODULI {
            let ctx = Arc::new(Zqcontext::new(*modulus, degree)?);
            let p = ZqX::random(&ctx, Representation::PowerBasis, &mut rng);
            let mut p_ntt = p.clone();
            p_ntt.change_representation(Representation::Ntt);
            let mut p_ntt_shoup = p.clone();
            p_ntt_shoup.change_representation(Representation::NttShoup);
            let p_coeffs = Vec::<u64>::from(&p);

            // Substitution by even numbers, should fail
            assert!(SubstitutionExponent::new(0, degree).is_err());
            assert!(SubstitutionExponent::new(2, degree).is_err());
            assert!(SubstitutionExponent::new(16, degree).is_err());

            // Substitution by 1 leaves the polynomials unchanged
            assert_eq!(p, p.substitute(&SubstitutionExponent::new(1, degree)?)?);
            assert_eq!(
                p_ntt,
                p_ntt.substitute(&SubstitutionExponent::new(1, degree)?)?
            );
            assert_eq!(
                p_ntt_shoup,
                p_ntt_shoup.substitute(&SubstitutionExponent::new(1, degree)?)?
            );

            // Substitution by 3
            let mut q = p.substitute(&SubstitutionExponent::new(3, degree)?)?;
            let mut v = vec![0u64; 8];
            for i in 0..8 {
                v[(3 * i) % 8] = if ((3 * i) / 8) & 1 == 1 && p_coeffs[i] > 0 {
                    *modulus - p_coeffs[i]
                } else {
                    p_coeffs[i]
                };
            }
            assert_eq!(&Vec::<u64>::from(&q), &v);

            let q_ntt = p_ntt.substitute(&SubstitutionExponent::new(3, degree)?)?;
            q.change_representation(Representation::Ntt);
            assert_eq!(q, q_ntt);

            let q_ntt_shoup = p_ntt_shoup.substitute(&SubstitutionExponent::new(3, degree)?)?;
            q.change_representation(Representation::NttShoup);
            assert_eq!(q, q_ntt_shoup);

            // 11 = 3^(-1) % 16
            assert_eq!(
                p,
                p.substitute(&SubstitutionExponent::new(3, degree)?)?
                    .substitute(&SubstitutionExponent::new(11, degree)?)?
            );
            assert_eq!(
                p_ntt,
                p_ntt
                    .substitute(&SubstitutionExponent::new(3, degree)?)?
                    .substitute(&SubstitutionExponent::new(11, degree)?)?
            );
            assert_eq!(
                p_ntt_shoup,
                p_ntt_shoup
                    .substitute(&SubstitutionExponent::new(3, degree)?)?
                    .substitute(&SubstitutionExponent::new(11, degree)?)?
            );
        }

        Ok(())
    }

    #[test]
    fn multiply_inverse_power_of_x() -> Result<(), Box<dyn Error>> {
        for modulus in MODULI {
            let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
            let q = &*ctx.q;
            let coeffs: [i64; 8] = [1,0,-1,-2,-1,0,1,2];
            let mut e = try_convert_from(&coeffs, &ctx, false, Representation::PowerBasis)?;

            assert!(e.multiply_inverse_power_of_x(1).is_ok()); // times X^{-1}
            let expected:[i64;8] = [0,-1,-2,-1,0,1,2,-1];
            let obtained = unsafe {q.center_vec_vt(&e.coefficients.as_slice())};
            assert_eq!(expected, obtained.as_slice());

            assert!(e.multiply_inverse_power_of_x(5).is_ok()); // times X^{-5}
            let expected:[i64;8] = [1,2,-1,0,1,2,1,0];
            let obtained = unsafe {q.center_vec_vt(&e.coefficients.as_slice())};
            assert_eq!(expected, obtained.as_slice());
        }
        Ok(())
    }
}

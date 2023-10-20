//! Implementation of operations over single-precision polynomials.

use super::ZqX;
use crate::{Error, Result, zqx::Representation};
//use itertools::{Itertools};
use std::{
    cmp::min,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

// Implementing self += other
impl AddAssign<&ZqX> for ZqX {
    fn add_assign(&mut self, p: &ZqX) {
        // Coefficients must be fully reduced, no lazy reduction
        // SHAI: Why is that?
        assert!(!self.has_lazy_coefficients && !p.has_lazy_coefficients);
        assert_ne!(
            self.representation,
            Representation::NttShoup,
            "Cannot add to a polynomial in NttShoup representation"
        ); // SHAI: Why is that?
        assert_eq!(
            self.representation, p.representation,
            "Incompatible representations"
        );
        // SHAI: not being able to operate on polynomials in different
        // representations calls to question the desicion to have them as
        // the same struct (as opposed to a different struct for each form)

        // Check that these polynomials are defined relative to the same
        // integer and polynomial moduli
        debug_assert_eq!(self.ctx, p.ctx, "Incompatible contexts");

        // allow-variable-time is sticky, the result allows variable time
        // if either of the arguments do
        self.allow_variable_time_computations |= p.allow_variable_time_computations;

        // Call the lower-level addition routine on the coefficient vectors
        if self.allow_variable_time_computations {
            unsafe{ self.ctx.q.add_vec_vt(self.coefficients.as_mut_slice(), &p.coefficients); }
        } else {
            self.ctx.q.add_vec(self.coefficients.as_mut_slice(), &p.coefficients);
        }
    }
}

// Implementing binary addition, self + other
impl Add<&ZqX> for &ZqX {
    type Output = ZqX;
    fn add(self, p: &ZqX) -> ZqX {
        let mut q = self.clone();
        q += p;
        q
    }
}

// Add with move, returning the result
// SHAI: When is this used?
impl Add for ZqX {
    type Output = ZqX;
    fn add(self, mut p: ZqX) -> ZqX {
        p += &self;
        p
    }
}

// Implementing self -= other
impl SubAssign<&ZqX> for ZqX {
    fn sub_assign(&mut self, p: &ZqX) {
        // Coefficients must be fully reduced, no lazy reduction
        assert!(!self.has_lazy_coefficients && !p.has_lazy_coefficients);
        assert_ne!(
            self.representation,
            Representation::NttShoup,
            "Cannot subtract from a polynomial in NttShoup representation"
        );
        assert_eq!(
            self.representation, p.representation,
            "Incompatible representations"
        );

        // Check that these polynomials are defined relative to the same
        // integer and polynomial moduli
        debug_assert_eq!(self.ctx, p.ctx, "Incompatible contexts");

        // allow-variable-time is sticky, the result allows variable time
        // if either of the arguments do
        self.allow_variable_time_computations |= p.allow_variable_time_computations;

        // Call the lower-level addition routine on the coefficient vectors
        if self.allow_variable_time_computations {
            unsafe{ self.ctx.q.sub_vec_vt(self.coefficients.as_mut_slice(), &p.coefficients); }
        } else {
            self.ctx.q.sub_vec(self.coefficients.as_mut_slice(), &p.coefficients);
        }
    }
}

// Implementing binary subtraction, self - other
impl Sub<&ZqX> for &ZqX {
    type Output = ZqX;
    fn sub(self, p: &ZqX) -> ZqX {
        let mut q = self.clone();
        q -= p;
        q
    }
}

// Implementing self *= other
impl MulAssign<&ZqX> for ZqX {
    fn mul_assign(&mut self, p: &ZqX) {
        // other must be fully reduced, no lazy reduction
        assert!(!p.has_lazy_coefficients);
        // self cannot be in NttShoup representation (SHAI: why?)
        assert_ne!(
            self.representation,
            Representation::NttShoup,
            "Cannot multiply to a polynomial in NttShoup representation"
        );
        if self.has_lazy_coefficients && self.representation == Representation::Ntt {
            assert!(
				p.representation == Representation::NttShoup,
				"Can only multiply a polynomial with lazy coefficients by an NttShoup representation."
			);
        } else {
            assert_eq!(
                self.representation,
                Representation::Ntt,
                "Multiplication requires an Ntt representation."
            );
            // SHAI: Why not convert to NTT representation? This calls into
            // question the design of having all these representations in
            // the same struct, as opposed to one struct for each
        }

        // Check that these polynomials are defined relative to the same
        // integer and polynomial moduli
        debug_assert_eq!(self.ctx, p.ctx, "Incompatible contexts");

        // allow-variable-time is sticky, the result allows variable time
        // if either of the arguments do
        self.allow_variable_time_computations |= p.allow_variable_time_computations;

        match p.representation {
            // The "simple" procedure, when both are in NTT representation
            Representation::Ntt => {
                if self.allow_variable_time_computations {
                    unsafe { self.ctx.q.mul_vec_vt(self.coefficients.as_mut_slice(), &p.coefficients); }
                } else {
                    self.ctx.q.mul_vec(self.coefficients.as_mut_slice(), &p.coefficients);
                }
            }

            // Optimized implementation when other is in NttShoup representation
            Representation::NttShoup => {
                // Convert from Option< Vec<u64> > to &[u64]
                let shoup_coeffs = &p.coefficients_shoup.as_ref().unwrap()[..];
                if self.allow_variable_time_computations {
                    unsafe { self.ctx.q.mul_shoup_vec_vt(
                            self.coefficients.as_mut_slice(), &p.coefficients, shoup_coeffs); }
                } else {
                    self.ctx.q.mul_shoup_vec(
                        self.coefficients.as_mut_slice(), &p.coefficients, shoup_coeffs);
                }
                self.has_lazy_coefficients = false
            }
            _ => {
                panic!("Multiplication requires a multipliand in Ntt or NttShoup representation.")
            }
        }
    }
}

// Implementing self *= scalar
impl MulAssign<u64> for ZqX {
    fn mul_assign(&mut self, p: u64) {
        // self must be fully reduced, no lazy reduction (SHAI: Why?)
        assert!(!self.has_lazy_coefficients);

        // Ensure that p is fully reduced
        let pp = p % self.ctx.modulus();

        // Multiply the coefficient vector by p, modulo q
        if self.allow_variable_time_computations {
            unsafe { self.ctx.q.scalar_mul_vec_vt(self.coefficients.as_mut_slice(), pp); }
        } else {
            self.ctx.q.scalar_mul_vec(self.coefficients.as_mut_slice(), pp);
        }

        // If needed, re-calculate the NttShoup coefficients
        if self.representation == Representation::NttShoup {
            unsafe {self.override_representation(Representation::NttShoup);}
        }
    }
}

// Multiplication with move, return the result
// FIXME: Something is very odd about the logic below
impl Mul<&ZqX> for &ZqX {
    type Output = ZqX;
    fn mul(self, p: &ZqX) -> ZqX {
        match self.representation {
            Representation::NttShoup => {
                // TODO: To test, and do the same thing for add, sub, and neg
                let mut q = p.clone();
                if q.representation == Representation::NttShoup {
                    unsafe { q.override_representation(Representation::Ntt) }
                }
                q *= self;
                q
            }
            _ => {
                let mut q = self.clone();
                q *= p;
                q
            }
        }
    }
}

impl Mul<u64> for &ZqX {
    type Output = ZqX;
    fn mul(self, p: u64) -> ZqX {
        let mut q = self.clone();
        q *= p;
        q
    }
}

impl Mul<&ZqX> for u64 {
    type Output = ZqX;
    fn mul(self, p: &ZqX) -> ZqX {
        p * self
    }
}

// Implementing -p
impl Neg for &ZqX {
    type Output = ZqX;

    fn neg(self) -> ZqX {
        assert!(!self.has_lazy_coefficients);
        let mut out = self.clone();
        if out.allow_variable_time_computations {
            unsafe { out.ctx.q.neg_vec_vt(out.coefficients.as_mut_slice()); }
        } else {
            out.ctx.q.neg_vec(out.coefficients.as_mut_slice());
        }
        out
    }
}

impl Neg for ZqX {
    type Output = ZqX;

    fn neg(mut self) -> ZqX {
        assert!(!self.has_lazy_coefficients);
        if self.allow_variable_time_computations {
            unsafe { self.ctx.q.neg_vec_vt(self.coefficients.as_mut_slice()); }
        } else {
            self.ctx.q.neg_vec(self.coefficients.as_mut_slice());
        }
        self
    }
}

/// Computes the Fused-Mul-Add operation `out[i] += x[i] * y[i]`
unsafe fn fma(out: &mut [u128], x: &[u64], y: &[u64]) {
    let n = out.len();
    assert_eq!(x.len(), n);
    assert_eq!(y.len(), n);

    macro_rules! fma_at {
        ($idx:expr) => {
            *out.get_unchecked_mut($idx) +=
                (*x.get_unchecked($idx) as u128) * (*y.get_unchecked($idx) as u128);
        };
    }

    let r = n / 16;
    for i in 0..r {
        fma_at!(16 * i);
        fma_at!(16 * i + 1);
        fma_at!(16 * i + 2);
        fma_at!(16 * i + 3);
        fma_at!(16 * i + 4);
        fma_at!(16 * i + 5);
        fma_at!(16 * i + 6);
        fma_at!(16 * i + 7);
        fma_at!(16 * i + 8);
        fma_at!(16 * i + 9);
        fma_at!(16 * i + 10);
        fma_at!(16 * i + 11);
        fma_at!(16 * i + 12);
        fma_at!(16 * i + 13);
        fma_at!(16 * i + 14);
        fma_at!(16 * i + 15);
    }

    for i in 0..n % 16 {
        fma_at!(16 * r + i);
    }
}

/// Compute the dot product between two iterators of polynomials.
/// Returna an error if the iterator counts are 0, or if any of the polynomial
/// is not in Ntt or NttShoup representation.
pub fn dot_product<'a, 'b, I, J>(p: I, q: J) -> Result<ZqX>
where
    I: Iterator<Item = &'a ZqX> + Clone,
    J: Iterator<Item = &'b ZqX> + Clone,
{
    debug_assert!(!p
        .clone()
        .any(|pi| pi.representation == Representation::PowerBasis));
    debug_assert!(!q
        .clone()
        .any(|qi| qi.representation == Representation::PowerBasis));

    // FIXME: If any of the iterators in *not* empty then we can get from it
    // the context, so we could return the zero polynomial in that case.
    let mut count = min(p.clone().count(), q.clone().count()) as u64;
    if count == 0 {
        return Err(Error::Default("At least one iterator is empty".to_string()));
    }

    // FIXME: Why not use p.peekable().peek()?
    let p_first = p.clone().next().unwrap();

    // Reference to the Modulus object
    let m = p_first.ctx.q.as_ref();

    // The degree of the polynomials
    let degree = p_first.ctx.degree;

    // FIXME: that looks wrong, why only look at the flag of the 1st ZqX in p?
    let allow_vt: bool = p_first.allow_variable_time_computations;

    // Maximum number of products that can be accumulated: If q<2^{64-n}
    // then you can accumulate 2^{2n} products of numbers in Z_q before
    // you are risking overflow of the 128-bit accumultor. We never try
    // to accumulate more than 2^63 products at a time.
    let n_zeros = min(63, m.modulus().leading_zeros() * 2);
    let mut max_acc = 1_u64 << n_zeros;

    // Initialize the accumulator, as a low-level vector of 128-bit integers
    let mut acc = vec!(0_u128; degree);

    // The easy case: we can accumulate all the products in one go
    if count as u64 <= max_acc {
        for (pi, qi) in p.zip(q) { // iterate over both lists
            unsafe { fma(acc.as_mut_slice(), pi.coefficients().as_slice(), qi.coefficients().as_slice()); }
        }
    }
    // The hard case: accumulate max_acc products at a time
    else {
        let mut mut_p = p;
        let mut mut_q = q;
        // The loop below maintain invariant that count => max_acc
        loop { // as long as count > 0
            for _ in 0..max_acc {
                let p_coefs= mut_p.next().unwrap().coefficients().as_slice();
                let q_coefs= mut_q.next().unwrap().coefficients().as_slice();
                unsafe { fma(acc.as_mut_slice(), p_coefs, q_coefs); }
            }
            count -= max_acc; // How many are left to process
            if count == 0 {   // Nothing left to process
                break;
            }
            if count < max_acc {
                max_acc = count; // process only remaining count products in last iteration
            }

            // Reduce the accumulator modulo q before the next iteration
            if allow_vt {
                for x in acc.iter_mut() {
                    unsafe { *x = m.reduce_u128_vt(*x) as u128; }
                }
            } else {
                for x in acc.iter_mut() {
                    *x = m.reduce_u128(*x) as u128;
                }
            }
        }
    }

    // Finally, reduce the accumulator modulo m into a vector of u64's
    let mut coeffs = vec!(0_u64; degree);
    if allow_vt {
        for (x,y) in coeffs.iter_mut().zip(acc.iter()) {
            unsafe { *x = m.reduce_u128_vt(*y); }
        }
    } else {
        for (x,y) in coeffs.iter_mut().zip(acc.iter()) {
            *x = m.reduce_u128(*y);
        }
    }

    Ok(ZqX {
        ctx: p_first.ctx.clone(),
        representation: Representation::Ntt,
        allow_variable_time_computations: allow_vt,
        coefficients: coeffs,
        coefficients_shoup: None,
        has_lazy_coefficients: false,
    })
}


#[cfg(test)]
mod tests {
    use itertools::{izip, Itertools};
    use rand::thread_rng;

    use super::dot_product;
    use crate::zqx::{Zqcontext, ZqX, Representation};
    use std::{error::Error, sync::Arc};

    static MODULI: &[u64; 3] = &[1153, 4611686018326724609, 4611686018309947393];

    #[test]
    fn add() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for _ in 0..100 {
            for modulus in MODULI {
                let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
                let m = ctx.q.as_ref();

                let p = ZqX::random(&ctx, Representation::PowerBasis, &mut rng);
                let q = ZqX::random(&ctx, Representation::PowerBasis, &mut rng);
                let r = &p + &q;
                assert_eq!(r.representation, Representation::PowerBasis);
                let mut a = p.coefficients.clone();
                m.add_vec(a.as_mut_slice(), q.coefficients.as_slice());
                assert_eq!(r.coefficients, a);

                let p = ZqX::random(&ctx, Representation::Ntt, &mut rng);
                let q = ZqX::random(&ctx, Representation::Ntt, &mut rng);
                let r = &p + &q;
                let mut a = p.coefficients.clone();
                m.add_vec(a.as_mut_slice(), q.coefficients.as_slice());
                assert_eq!(r.coefficients, a);
            }
        }
        Ok(())
    }

    #[test]
    fn sub() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for _ in 0..100 {
            for modulus in MODULI {
                let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
                let m = ctx.q.as_ref();

                let p = ZqX::random(&ctx, Representation::PowerBasis, &mut rng);
                let q = ZqX::random(&ctx, Representation::PowerBasis, &mut rng);
                let r = &p - &q;
                assert_eq!(r.representation, Representation::PowerBasis);
                let mut a = p.coefficients.clone();
                m.sub_vec(a.as_mut_slice(), q.coefficients.as_slice());
                assert_eq!(r.coefficients, a);
            }
        }
        Ok(())
    }

    #[test]
    fn mul() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for _ in 0..100 {
            for modulus in MODULI {
                let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
                let m = ctx.q.as_ref();

                let p = ZqX::random(&ctx, Representation::Ntt, &mut rng);
                let q = ZqX::random(&ctx, Representation::Ntt, &mut rng);
                let r = &p * &q;
                assert_eq!(r.representation, Representation::Ntt);
                let mut a = p.coefficients.clone();
                m.mul_vec(a.as_mut_slice(), q.coefficients.as_slice());
                assert_eq!(r.coefficients, a);
            }
        }
        Ok(())
    }

    #[test]
    fn mul_shoup() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for _ in 0..100 {
            for modulus in MODULI {
                let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
                let m = ctx.q.as_ref();

                let p = ZqX::random(&ctx, Representation::Ntt, &mut rng);
                let q = ZqX::random(&ctx, Representation::NttShoup, &mut rng);
                let r = &p * &q;
                assert_eq!(r.representation, Representation::Ntt);
                let mut a = p.coefficients.clone();
                m.mul_vec(a.as_mut_slice(), q.coefficients.as_slice());
                assert_eq!(r.coefficients, a);
            }
        }
        Ok(())
    }

    #[test]
    fn neg() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for _ in 0..100 {
            for modulus in MODULI {
                let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);
                let m = ctx.q.as_ref();

                let p = ZqX::random(&ctx, Representation::PowerBasis, &mut rng);
                let r = -&p;
                assert_eq!(r.representation, Representation::PowerBasis);
                let mut a = p.coefficients.clone();
                m.neg_vec(a.as_mut_slice());
                assert_eq!(r.coefficients, a);

                let p = ZqX::random(&ctx, Representation::Ntt, &mut rng);
                let r = -&p;
                assert_eq!(r.representation, Representation::Ntt);
                let mut a = p.coefficients.clone();
                m.neg_vec(a.as_mut_slice());
                assert_eq!(r.coefficients, a);
            }
        }
        Ok(())
    }

    #[test]
    fn test_dot_product() -> Result<(), Box<dyn Error>> {
        let mut rng = thread_rng();
        for _ in 0..20 {
            for modulus in MODULI {
                let ctx = Arc::new(Zqcontext::new(*modulus, 8)?);

                for len in 1..50 {
                    let p = (0..len)
                        .map(|_| ZqX::random(&ctx, Representation::Ntt, &mut rng))
                        .collect_vec();
                    let q = (0..len)
                        .map(|_| ZqX::random(&ctx, Representation::Ntt, &mut rng))
                        .collect_vec();
                    let r = dot_product(p.iter(), q.iter())?;

                    let mut expected = ZqX::zero(&ctx, Representation::Ntt);
                    izip!(&p, &q).for_each(|(pi, qi)| expected += &(pi * qi));
                    assert_eq!(r, expected);
                }
            }
        }
        Ok(())
    }
}

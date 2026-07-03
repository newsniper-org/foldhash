//! The foldhash `secure` variant — a keyed 64-bit PRF (no S-box).
//!
//! Unlike [`fast`](crate::fast) and [`quality`](crate::quality), which are
//! *non-cryptographic* and only *minimally* DoS-resistant, this variant is a
//! heuristic **keyed pseudo-random function** designed so that, without the
//! key, its output is indistinguishable from random and key recovery costs
//! ~2^128. Its worst-case cryptographic targets match SipHash-1-3 (see the
//! honest scope below). It is intended for MAC-/token-like uses and for hash
//! maps that need genuine HashDoS resistance rather than the "minimal"
//! resistance of [`fast`].
//!
//! ## Construction (RD1, `rotfeistel(23, 31)`, R_b = 2, R_f = 4)
//!
//! A 128-bit state `(a, b)` absorbs the message in 16-byte little-endian blocks,
//! running `R_B` non-linear rounds per block; a length-domain block and `R_F`
//! finalization rounds follow, and the 128-bit state is folded to 64 bits
//! (`a ^ b`). Each round is `non-ARX` and table-free:
//!
//! ```text
//! round(a, b, rk0, rk1):
//!     a ^= rk0 ; b ^= rk1                 # key injection (XOR)
//!     (lo, hi) = widen(a, b)              # lo = low64(a*b), hi = high64(a*b)
//!     a = lo ^ mulhi(a, C)               # multiply-high confusion (no S-box)
//!     b = hi ^ mulhi(b, Cp)
//!     t = a ^ rotl(b, 23) ; b = b ^ rotl(t, 31) ; a = t   # rotfeistel diffusion
//! ```
//!
//! Round keys are derived on the fly from the 128-bit master key with a
//! multiply-free additive (Weyl) schedule, so the only non-linearity is the
//! integer multiply (`widen`/`mulhi`) — there is no lookup table and no
//! bitsliced S-box, giving cache-/data-timing immunity on architectures with a
//! constant-latency 64x64->128 multiply.
//!
//! ## Honest scope
//!
//! Security is **heuristic** (established by empirical cryptanalysis and a
//! conditional PRF reduction), *not* an unconditional proof — as is the case for
//! every practical keyed hash including SipHash/HMAC. The 64-bit output implies
//! a 2^32 collision birthday bound, identical to SipHash-64. This variant is
//! slower than [`fast`]/[`quality`] (it is a real per-block cryptographic
//! permutation-like round), hence [`HashInfo`]-style `SLOW`.
//!
//! ## Keying
//!
//! Security **requires an unpredictable key**. Use [`RandomState`] (enabled by
//! the `secure` crate feature, which draws a 128-bit key from the OS CSPRNG via
//! `getrandom`). [`FixedState`] uses a caller-supplied fixed key and is
//! deterministic — it is only appropriate when the key is itself secret, and is
//! trivially HashDoS-vulnerable if the key is public.

use core::hash::{BuildHasher, Hasher};

// Fixed odd confusion constants (both odd => units in Z/2^64).
const C: u64 = 0x2545F4914F6CDD1D;
const CP: u64 = 0x9E3779B97F4A7C15;
// Golden-ratio odd Weyl increment (multiply-free key-schedule step).
const GAMMA: u64 = 0x9E3779B97F4A7C15;
// SplitMix64 finalizer multipliers (one-time init-state derivation only).
const SM1: u64 = 0xBF58476D1CE4E5B9;
const SM2: u64 = 0x94D049BB133111EB;
// Domain separators for init state and the length lane.
const INIT_A_DOMAIN: u64 = 0x6A09E667F3BCC908;
const INIT_B_DOMAIN: u64 = 0xBB67AE8584CAA73B;
const LEN_DOMAIN: u64 = 0x3C6EF372FE94F82B;
// rotfeistel rotations (23 + 31 = 54 (mod 64) != 0 => genuine b-dependence).
const RF_R1: u32 = 23;
const RF_R2: u32 = 31;
// Weyl key-schedule rotations.
const KS_R0: u32 = 23;
const KS_R1: u32 = 41;
// Adopted round counts.
const R_B: usize = 2;
const R_F: usize = 4;
// Arbitrary fixed key for `FixedState::default()` (insecure unless kept secret).
const FIXED_KLO: u64 = 0x243F6A8885A308D3;
const FIXED_KHI: u64 = 0x13198A2E03707344;

// --------------------------------------------------------------------------- //
// multiply primitives — must be constant-time on the data path.
//
// The secure variant's cache-/data-timing immunity rests on the 64x64->128
// multiply (widen/mulhi) having *data-independent latency*. This holds on
// mainstream cores (x86-64 MUL/MULX, aarch64 MUL/UMULH, modern RISC-V
// MUL/MULHU). On cores with a data-dependent / early-terminating multiplier
// (some ARM Cortex-M, older ARM7, certain low-end embedded RISC-V/MIPS) -- and
// on 32-bit / wasm targets that emulate the widening product with narrower
// multiplies that may themselves be variable-time -- the native multiply leaks
// operand magnitude through timing.
//
// Enable the `ct-mul` feature on such targets: it replaces every data-path
// multiply with a data-oblivious software multiply (`mul_wide_ct`) -- no
// multiply instruction, no branch on data, no table; only shifts, AND and ADD
// over a fixed 64-step schedule -- which is **bit-exact** with the native
// product (so the frozen KAT is preserved) at a substantial speed cost. Without
// the feature the fast native multiply is used.
// --------------------------------------------------------------------------- //

#[cfg(not(feature = "ct-mul"))]
#[inline(always)]
fn mulhi(x: u64, k: u64) -> u64 {
    (((x as u128) * (k as u128)) >> 64) as u64
}
#[cfg(not(feature = "ct-mul"))]
#[inline(always)]
fn widen(a: u64, b: u64) -> (u64, u64) {
    let p = (a as u128) * (b as u128);
    (p as u64, (p >> 64) as u64)
}
#[cfg(not(feature = "ct-mul"))]
#[inline(always)]
fn wmul_lo(x: u64, k: u64) -> u64 {
    x.wrapping_mul(k)
}

/// Constant-time (data-oblivious) 64x64 -> 128 schoolbook multiply.
///
/// Every iteration processes one bit of `b` with a *branchless* masked add;
/// the shift amounts are the loop index (not data), and no hardware multiply is
/// used. `black_box` stops LLVM from reassociating the shift/add schedule back
/// into a hardware multiply, which would reintroduce the variable-latency
/// instruction this path exists to avoid. Bit-exact with `(a as u128)*(b as u128)`.
#[cfg(feature = "ct-mul")]
#[inline]
fn mul_wide_ct(a: u64, b: u64) -> (u64, u64) {
    let mut lo: u64 = 0;
    let mut hi: u64 = 0;
    let mut i: u32 = 0;
    while i < 64 {
        let bit = (b >> i) & 1;
        let mask = 0u64.wrapping_sub(bit); // 0x0.. or 0xf.. — data-oblivious select
        let add_lo = (a << i) & mask; // i is the loop index, not secret data
        let add_hi = (if i == 0 { 0 } else { a >> (64 - i) }) & mask;
        let (nlo, carry) = lo.overflowing_add(core::hint::black_box(add_lo));
        lo = nlo;
        hi = hi.wrapping_add(add_hi).wrapping_add(carry as u64);
        i += 1;
    }
    (lo, hi)
}
#[cfg(feature = "ct-mul")]
#[inline(always)]
fn widen(a: u64, b: u64) -> (u64, u64) {
    mul_wide_ct(a, b)
}
#[cfg(feature = "ct-mul")]
#[inline(always)]
fn mulhi(x: u64, k: u64) -> u64 {
    mul_wide_ct(x, k).1
}
#[cfg(feature = "ct-mul")]
#[inline(always)]
fn wmul_lo(x: u64, k: u64) -> u64 {
    mul_wide_ct(x, k).0
}

#[inline(always)]
fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(GAMMA);
    let mut z = x;
    z = wmul_lo(z ^ (z >> 30), SM1);
    z = wmul_lo(z ^ (z >> 27), SM2);
    z ^ (z >> 31)
}

#[inline(always)]
const fn round_keys(klo: u64, khi: u64, w: u64) -> (u64, u64) {
    let rk0 = klo ^ w.rotate_left(KS_R0);
    let rk1 = khi ^ (w ^ klo).rotate_left(KS_R1);
    (rk0, rk1)
}

#[inline(always)]
fn init_state(klo: u64, khi: u64) -> (u64, u64) {
    let a = splitmix64((klo ^ INIT_A_DOMAIN).wrapping_add(khi));
    let b = splitmix64((khi ^ INIT_B_DOMAIN).wrapping_add(klo));
    (a, b)
}

#[inline(always)]
fn round(a: u64, b: u64, rk0: u64, rk1: u64) -> (u64, u64) {
    let a = a ^ rk0;
    let b = b ^ rk1;
    let (lo, hi) = widen(a, b);
    let a = lo ^ mulhi(a, C);
    let b = hi ^ mulhi(b, CP);
    let t = a ^ b.rotate_left(RF_R1);
    let b2 = b ^ t.rotate_left(RF_R2);
    (t, b2)
}

/// A [`Hasher`] implementing the foldhash `secure` keyed PRF.
///
/// Create one via [`RandomState`], [`FixedState`], or directly with
/// [`SecureFoldHasher::with_key`]. The resident working state is the two 64-bit
/// lanes plus the 128-bit master key (32 bytes) plus a 16-byte block buffer;
/// round keys are derived on the fly (no materialized round-key array).
#[derive(Clone)]
pub struct SecureFoldHasher {
    a: u64,
    b: u64,
    klo: u64,
    khi: u64,
    w: u64,
    buf: [u8; 16],
    buf_len: u8,
    total_len: u64,
}

impl SecureFoldHasher {
    /// Initializes a hasher with the given 128-bit master key
    /// (`key = (khi << 64) | klo`).
    ///
    /// For HashDoS/PRF security the key must be unpredictable to the attacker;
    /// prefer [`RandomState`] which draws it from the OS CSPRNG.
    #[inline]
    pub fn with_key(klo: u64, khi: u64) -> Self {
        let (a, b) = init_state(klo, khi);
        Self {
            a,
            b,
            klo,
            khi,
            w: 0,
            buf: [0u8; 16],
            buf_len: 0,
            total_len: 0,
        }
    }

    #[inline(always)]
    fn absorb_block(&mut self, m0: u64, m1: u64) {
        let mut a = self.a ^ m0;
        let mut b = self.b ^ m1;
        let mut w = self.w;
        let mut i = 0;
        while i < R_B {
            w = w.wrapping_add(GAMMA);
            let (rk0, rk1) = round_keys(self.klo, self.khi, w);
            let (na, nb) = round(a, b, rk0, rk1);
            a = na;
            b = nb;
            i += 1;
        }
        self.a = a;
        self.b = b;
        self.w = w;
    }
}

impl Hasher for SecureFoldHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.total_len = self.total_len.wrapping_add(bytes.len() as u64);
        let mut input = bytes;

        // Complete a partially-filled buffer first.
        if self.buf_len > 0 {
            let have = self.buf_len as usize;
            let need = 16 - have;
            let take = need.min(input.len());
            self.buf[have..have + take].copy_from_slice(&input[..take]);
            self.buf_len += take as u8;
            input = &input[take..];
            if self.buf_len < 16 {
                return;
            }
            let m0 = u64::from_le_bytes(self.buf[0..8].try_into().unwrap());
            let m1 = u64::from_le_bytes(self.buf[8..16].try_into().unwrap());
            self.absorb_block(m0, m1);
            self.buf_len = 0;
        }

        // Absorb full 16-byte blocks directly from the input.
        while input.len() >= 16 {
            let m0 = u64::from_le_bytes(input[0..8].try_into().unwrap());
            let m1 = u64::from_le_bytes(input[8..16].try_into().unwrap());
            self.absorb_block(m0, m1);
            input = &input[16..];
        }

        // Stash the tail (0..15 bytes) for the next write / finish.
        if !input.is_empty() {
            self.buf[..input.len()].copy_from_slice(input);
            self.buf_len = input.len() as u8;
        }
    }

    #[inline(always)]
    fn write_u8(&mut self, i: u8) {
        self.write(&i.to_le_bytes());
    }
    #[inline(always)]
    fn write_u16(&mut self, i: u16) {
        self.write(&i.to_le_bytes());
    }
    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.write(&i.to_le_bytes());
    }
    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.write(&i.to_le_bytes());
    }
    #[inline(always)]
    fn write_u128(&mut self, i: u128) {
        self.write(&i.to_le_bytes());
    }
    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.write(&i.to_le_bytes());
    }

    #[cfg(feature = "nightly")]
    #[inline(always)]
    fn write_str(&mut self, s: &str) {
        self.write(s.as_bytes());
    }

    #[inline]
    fn finish(&self) -> u64 {
        // finish() must not mutate; compute on local copies of the state.
        let mut a = self.a;
        let mut b = self.b;
        let mut w = self.w;

        // Final partial block: zero-pad to 16 bytes (only if non-empty),
        // matching the one-shot construction's remainder handling.
        if self.buf_len > 0 {
            let mut blk = [0u8; 16];
            blk[..self.buf_len as usize].copy_from_slice(&self.buf[..self.buf_len as usize]);
            let m0 = u64::from_le_bytes(blk[0..8].try_into().unwrap());
            let m1 = u64::from_le_bytes(blk[8..16].try_into().unwrap());
            a ^= m0;
            b ^= m1;
            let mut i = 0;
            while i < R_B {
                w = w.wrapping_add(GAMMA);
                let (rk0, rk1) = round_keys(self.klo, self.khi, w);
                let (na, nb) = round(a, b, rk0, rk1);
                a = na;
                b = nb;
                i += 1;
            }
        }

        // Length domain absorption.
        let l = self.total_len;
        a ^= l;
        b ^= l ^ LEN_DOMAIN;

        // Finalize rounds.
        let mut i = 0;
        while i < R_F {
            w = w.wrapping_add(GAMMA);
            let (rk0, rk1) = round_keys(self.klo, self.khi, w);
            let (na, nb) = round(a, b, rk0, rk1);
            a = na;
            b = nb;
            i += 1;
        }

        a ^ b
    }
}

/// A [`BuildHasher`] for [`SecureFoldHasher`] using a fixed, caller-supplied key.
///
/// This is deterministic and therefore **not** HashDoS-resistant if the key is
/// public. Only use it when the key is itself a secret (e.g. a MAC key) or when
/// you explicitly need reproducibility and accept the loss of DoS resistance.
#[derive(Clone, Debug)]
pub struct FixedState {
    klo: u64,
    khi: u64,
}

impl FixedState {
    /// Creates a [`FixedState`] from a 128-bit key (`key = (khi << 64) | klo`).
    #[inline(always)]
    pub const fn with_keys(klo: u64, khi: u64) -> Self {
        Self { klo, khi }
    }
}

impl Default for FixedState {
    #[inline(always)]
    fn default() -> Self {
        Self {
            klo: FIXED_KLO,
            khi: FIXED_KHI,
        }
    }
}

impl BuildHasher for FixedState {
    type Hasher = SecureFoldHasher;
    #[inline(always)]
    fn build_hasher(&self) -> SecureFoldHasher {
        SecureFoldHasher::with_key(self.klo, self.khi)
    }
}

/// A [`BuildHasher`] for [`SecureFoldHasher`] with a per-instance key drawn from
/// the operating-system CSPRNG (via `getrandom`).
///
/// This is the recommended way to use the `secure` variant: each `RandomState`
/// holds an independent, unpredictable 128-bit key (16 bytes), giving genuine
/// HashDoS resistance. Requires the `secure` crate feature.
#[cfg(feature = "secure")]
#[derive(Clone, Debug)]
pub struct RandomState {
    klo: u64,
    khi: u64,
}

#[cfg(feature = "secure")]
impl RandomState {
    /// Generates a `RandomState` with a fresh 128-bit key from the OS CSPRNG.
    ///
    /// # Panics
    /// Panics if the operating system entropy source is unavailable.
    #[inline]
    pub fn new() -> Self {
        let mut kb = [0u8; 16];
        getrandom::getrandom(&mut kb).expect("foldhash::secure::RandomState: OS CSPRNG unavailable");
        Self {
            klo: u64::from_le_bytes(kb[0..8].try_into().unwrap()),
            khi: u64::from_le_bytes(kb[8..16].try_into().unwrap()),
        }
    }
}

#[cfg(feature = "secure")]
impl Default for RandomState {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "secure")]
impl BuildHasher for RandomState {
    type Hasher = SecureFoldHasher;
    #[inline(always)]
    fn build_hasher(&self) -> SecureFoldHasher {
        SecureFoldHasher::with_key(self.klo, self.khi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::hash::Hasher;

    fn h(hexmsg: &str, klo: u64, khi: u64) -> u64 {
        let bytes: std::vec::Vec<u8> = (0..hexmsg.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hexmsg[i..i + 2], 16).unwrap())
            .collect();
        let mut hasher = SecureFoldHasher::with_key(klo, khi);
        hasher.write(&bytes);
        hasher.finish()
    }

    // FROZEN KAT — bit-exact with foldhash-analysis/secure-design/ref_rev.py
    // hash(.., key, R_b=2, R_f=4, "RD1") and perf-rs-rev/src/lib.rs. Any change
    // to the construction that alters these values is a breaking change.
    const KLO1: u64 = 0xfedcba9876543210;
    const KHI1: u64 = 0x0123456789abcdef;
    const KLO2: u64 = 0x0123456789abcdef;
    const KHI2: u64 = 0xdeadbeefcafebabe;

    #[test]
    fn kat_frozen() {
        let k1: &[(&str, u64)] = &[
            ("", 0x4dcd3c2372e7d42e),
            ("00", 0x1c1143f6537428c6),
            ("01", 0xca836991a950a9bf),
            ("616263", 0x52eeec20004c3b75),
            ("0102030405060708", 0x0b3d9a67c4c25fd5),
            ("000102030405060708090a0b0c0d0e0f", 0xd90c37583f2f6109),
            ("000102030405060708090a0b0c0d0e0f10", 0x6feabdce30e7e428),
            (
                "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                0x91c301d063c0c97c,
            ),
            (
                "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
                0xa4119f25307662b3,
            ),
            (
                "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
                0x0632af966f2195fe,
            ),
            (
                "74686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f67",
                0x5acd229c81be14fc,
            ),
        ];
        for (m, exp) in k1 {
            assert_eq!(h(m, KLO1, KHI1), *exp, "KAT mismatch key1 msg={m}");
        }
        assert_eq!(h("", KLO2, KHI2), 0x452dc37070e745a9);
        assert_eq!(h("0001020304050607", KLO2, KHI2), 0xf285e69f1fd036ca);
        assert_eq!(
            h(
                "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                KLO2, KHI2
            ),
            0xf9767f2762ff0f8c
        );
    }

    #[test]
    fn length_domain_separation() {
        let e = {
            let mut x = SecureFoldHasher::with_key(KLO1, KHI1);
            x.write(b"");
            x.finish()
        };
        let z = {
            let mut x = SecureFoldHasher::with_key(KLO1, KHI1);
            x.write(b"\x00");
            x.finish()
        };
        assert_ne!(e, z);
    }

    #[test]
    fn streaming_equals_oneshot() {
        // Splitting a message across arbitrary write() boundaries must equal a
        // single write of the concatenation.
        let msg: std::vec::Vec<u8> = (0..200u32).map(|i| (i.wrapping_mul(37) ^ 0xA5) as u8).collect();
        let oneshot = {
            let mut x = SecureFoldHasher::with_key(KLO1, KHI1);
            x.write(&msg);
            x.finish()
        };
        for split in [1usize, 7, 8, 15, 16, 17, 31, 32, 48, 100, 128, 199] {
            let mut x = SecureFoldHasher::with_key(KLO1, KHI1);
            x.write(&msg[..split]);
            x.write(&msg[split..]);
            assert_eq!(x.finish(), oneshot, "streaming split at {split} differs");
        }
    }

    #[test]
    fn ct_mul_bit_exact_vs_native() {
        // The constant-time software multiply must be bit-exact to the native
        // widening product for ALL inputs — this is what lets the `ct-mul`
        // feature preserve the frozen KAT. Both algorithms are checked here
        // regardless of which feature is active.
        fn native(a: u64, b: u64) -> (u64, u64) {
            let p = (a as u128) * (b as u128);
            (p as u64, (p >> 64) as u64)
        }
        fn ct(a: u64, b: u64) -> (u64, u64) {
            let mut lo = 0u64;
            let mut hi = 0u64;
            let mut i = 0u32;
            while i < 64 {
                let bit = (b >> i) & 1;
                let mask = 0u64.wrapping_sub(bit);
                let add_lo = (a << i) & mask;
                let add_hi = (if i == 0 { 0 } else { a >> (64 - i) }) & mask;
                let (nlo, carry) = lo.overflowing_add(add_lo);
                lo = nlo;
                hi = hi.wrapping_add(add_hi).wrapping_add(carry as u64);
                i += 1;
            }
            (lo, hi)
        }
        for &(a, b) in &[
            (0u64, 0u64),
            (0, 1),
            (1, 0),
            (u64::MAX, u64::MAX),
            (u64::MAX, 1),
            (1, u64::MAX),
            (1u64 << 63, 1u64 << 63),
            (0xdeadbeefcafebabe, 0x0123456789abcdef),
        ] {
            assert_eq!(ct(a, b), native(a, b), "ct != native for ({a:#x},{b:#x})");
        }
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut nextr = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..200_000 {
            let a = nextr();
            let b = nextr();
            assert_eq!(ct(a, b), native(a, b));
        }
    }

    #[test]
    fn write_u64_matches_le_bytes() {
        let v = 0x0123456789abcdefu64;
        let a = {
            let mut x = SecureFoldHasher::with_key(KLO1, KHI1);
            x.write_u64(v);
            x.finish()
        };
        let b = {
            let mut x = SecureFoldHasher::with_key(KLO1, KHI1);
            x.write(&v.to_le_bytes());
            x.finish()
        };
        assert_eq!(a, b);
    }
}

#![feature(int_log)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fhers::bfv::{
	BfvParameters, BfvParametersBuilder, Encoding, EvaluationKeyBuilder, Plaintext, SecretKey,
};
use fhers_traits::{FheEncoder, FheEncrypter};
use itertools::Itertools;
use std::time::Duration;
use std::{error::Error, sync::Arc};

fn params() -> Result<Vec<Arc<BfvParameters>>, Box<dyn Error>> {
	let par_small = BfvParametersBuilder::new()
		.set_degree(4096)
		.set_plaintext_modulus(1153)
		.set_ciphertext_moduli_sizes(&[36, 37, 37])
		.build()?;
	let par_large = BfvParametersBuilder::new()
		.set_degree(16384)
		.set_plaintext_modulus(1153)
		.set_ciphertext_moduli_sizes(&[62; 7])
		.build()
		.unwrap();
	Ok(vec![Arc::new(par_small), Arc::new(par_large)])
}

pub fn bfv_benchmark(c: &mut Criterion) {
	let mut group = c.benchmark_group("bfv");
	group.sample_size(10);
	group.warm_up_time(Duration::from_secs(1));
	group.measurement_time(Duration::from_secs(1));

	for par in params().unwrap() {
		let sk = SecretKey::random(&par);
		let ek = EvaluationKeyBuilder::new(&sk)
			.enable_inner_sum()
			.unwrap()
			.enable_relinearization()
			.unwrap()
			.enable_column_rotation(1)
			.unwrap()
			.enable_expansion(par.degree().ilog2() as usize)
			.unwrap()
			.build()
			.unwrap();

		let pt1 =
			Plaintext::try_encode(&(1..16u64).collect_vec() as &[u64], Encoding::poly(), &par)
				.unwrap();
		let pt2 =
			Plaintext::try_encode(&(3..39u64).collect_vec() as &[u64], Encoding::poly(), &par)
				.unwrap();
		let mut c1 = sk.try_encrypt(&pt1).unwrap();
		let c2 = sk.try_encrypt(&pt2).unwrap();

		group.bench_function(
			BenchmarkId::new(
				"add",
				format!(
					"{}/{}",
					par.degree(),
					par.moduli_sizes().iter().sum::<usize>()
				),
			),
			|b| {
				b.iter(|| c1 = &c1 + &c2);
			},
		);

		group.bench_function(
			BenchmarkId::new(
				"add_assign",
				format!(
					"{}/{}",
					par.degree(),
					par.moduli_sizes().iter().sum::<usize>()
				),
			),
			|b| {
				b.iter(|| c1 += &c2);
			},
		);

		group.bench_function(
			BenchmarkId::new(
				"sub",
				format!(
					"{}/{}",
					par.degree(),
					par.moduli_sizes().iter().sum::<usize>()
				),
			),
			|b| {
				b.iter(|| c1 = &c1 - &c2);
			},
		);

		group.bench_function(
			BenchmarkId::new(
				"sub_assign",
				format!(
					"{}/{}",
					par.degree(),
					par.moduli_sizes().iter().sum::<usize>()
				),
			),
			|b| {
				b.iter(|| c1 -= &c2);
			},
		);

		group.bench_function(
			BenchmarkId::new(
				"neg",
				format!(
					"{}/{}",
					par.degree(),
					par.moduli_sizes().iter().sum::<usize>()
				),
			),
			|b| {
				b.iter(|| c1 = -&c2);
			},
		);

		let c3 = &c1 * &c1;
		group.bench_function(
			BenchmarkId::new(
				"relinearize",
				format!(
					"{}/{}",
					par.degree(),
					par.moduli_sizes().iter().sum::<usize>()
				),
			),
			|b| {
				b.iter(|| ek.relinearizes_new(&c3));
			},
		);

		group.bench_function(
			BenchmarkId::new(
				"rotate",
				format!(
					"{}/{}",
					par.degree(),
					par.moduli_sizes().iter().sum::<usize>()
				),
			),
			|b| {
				b.iter(|| c1 = ek.rotates_column_by(&c1, 1).unwrap());
			},
		);

		group.bench_function(
			BenchmarkId::new(
				"inner_sum",
				format!(
					"{}/{}",
					par.degree(),
					par.moduli_sizes().iter().sum::<usize>()
				),
			),
			|b| {
				b.iter(|| c1 = ek.computes_inner_sum(&c1).unwrap());
			},
		);

		for i in 1..par.degree().ilog2() + 1 {
			if par.degree() > 2048 && i > 4 {
				continue; // Skip slow benchmarks
			}
			group.bench_function(
				BenchmarkId::new(
					format!("expand_{}", i),
					format!(
						"{}/{}",
						par.degree(),
						par.moduli_sizes().iter().sum::<usize>()
					),
				),
				|b| {
					b.iter(|| ek.expands(&c1, i as usize).unwrap());
				},
			);
		}

		group.bench_function(
			BenchmarkId::new(
				"mul",
				format!(
					"{}/{}",
					par.degree(),
					par.moduli_sizes().iter().sum::<usize>()
				),
			),
			|b| {
				b.iter(|| &c1 * &c2);
			},
		);
	}

	group.finish();
}

criterion_group!(bfv, bfv_benchmark);
criterion_main!(bfv);
#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap


CASES = [
    {
        "name": "ip_bits8_proj8_seed7",
        "metric": "ip",
        "dim": 16,
        "bits": 8,
        "projections": 8,
        "seed": 7,
        "vector": [
            0.25, -1.5, 0.75, 2.0, -0.5, 1.25, -0.75, 0.125,
            0.33, -0.66, 1.0, -1.2, 0.8, -0.4, 1.5, -0.9,
        ],
        "query": [
            -0.3, 0.7, -1.1, 1.3, 0.9, -0.2, 0.4, -1.5,
            0.6, 0.1, -0.8, 1.1, -0.7, 0.5, -1.0, 0.2,
        ],
    },
    {
        "name": "cosine_bits8_proj8_seed7",
        "metric": "cosine",
        "dim": 16,
        "bits": 8,
        "projections": 8,
        "seed": 7,
        "vector": [
            0.25, -1.5, 0.75, 2.0, -0.5, 1.25, -0.75, 0.125,
            0.33, -0.66, 1.0, -1.2, 0.8, -0.4, 1.5, -0.9,
        ],
        "query": [
            -0.3, 0.7, -1.1, 1.3, 0.9, -0.2, 0.4, -1.5,
            0.6, 0.1, -0.8, 1.1, -0.7, 0.5, -1.0, 0.2,
        ],
    },
    {
        "name": "ip_bits16_proj16_seed11",
        "metric": "ip",
        "dim": 16,
        "bits": 16,
        "projections": 16,
        "seed": 11,
        "vector": [
            1.2, -0.8, 0.4, -1.6, 0.9, 0.3, -0.7, 1.1,
            -1.4, 0.2, 0.6, -0.9, 1.8, -1.1, 0.5, 0.75,
        ],
        "query": [
            -0.5, 1.4, -0.2, 0.8, -1.0, 0.65, 0.45, -0.35,
            1.25, -0.55, 0.95, -1.3, 0.15, 0.85, -0.6, 1.05,
        ],
    },
]


def run_oracle(oracle_repo: pathlib.Path) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="tq-oracle-") as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        (tmpdir / "Cargo.toml").write_text(
            textwrap.dedent(
                f"""
                [package]
                name = "tq_oracle_runner"
                version = "0.1.0"
                edition = "2021"

                [dependencies]
                serde = {{ version = "1", features = ["derive"] }}
                serde_json = "1"
                turbo-quant = {{ path = {json.dumps(str(oracle_repo))} }}
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        srcdir = tmpdir / "src"
        srcdir.mkdir()
        (tmpdir / "cases.json").write_text(json.dumps(CASES), encoding="utf-8")
        (srcdir / "main.rs").write_text(
            textwrap.dedent(
                """
                use serde::{Deserialize, Serialize};
                use std::{f32, fs};
                use turbo_quant::TurboQuantizer;

                #[derive(Debug, Deserialize)]
                struct InputCase {
                    name: String,
                    metric: String,
                    dim: usize,
                    bits: u8,
                    projections: usize,
                    seed: u64,
                    vector: Vec<f32>,
                    query: Vec<f32>,
                }

                #[derive(Debug, Serialize)]
                struct OutputCase {
                    name: String,
                    metric: String,
                    dim: usize,
                    bits: usize,
                    projections: usize,
                    seed: u64,
                    vector: Vec<f32>,
                    query: Vec<f32>,
                    radii: Vec<f32>,
                    angle_indices: Vec<u16>,
                    residual_signs: Vec<i8>,
                    inner_product_estimate: f32,
                    l2_distance_estimate: f32,
                    code_norm_sq: f32,
                    exact_inner_product: f32,
                    exact_l2_distance: f32,
                }

                fn normalize(values: &mut [f32]) {
                    let norm_sq: f32 = values.iter().map(|v| v * v).sum();
                    if norm_sq <= f32::EPSILON {
                        return;
                    }
                    let inv_norm = 1.0f32 / norm_sq.sqrt();
                    for value in values.iter_mut() {
                        *value *= inv_norm;
                    }
                }

                fn main() {
                    let raw = fs::read_to_string("cases.json").expect("read cases");
                    let inputs: Vec<InputCase> = serde_json::from_str(&raw).expect("parse cases");
                    let mut outputs = Vec::new();

                    for input in inputs {
                        let quantizer = TurboQuantizer::new(
                            input.dim,
                            input.bits,
                            input.projections,
                            input.seed,
                        )
                        .expect("new quantizer");

                        let mut vector = input.vector.clone();
                        let mut query = input.query.clone();
                        if input.metric == "cosine" {
                            normalize(&mut vector);
                            normalize(&mut query);
                        }

                        let code = quantizer.encode(&vector).expect("encode");
                        let inner_product_estimate = quantizer
                            .inner_product_estimate(&code, &query)
                            .expect("ip estimate");
                        let l2_distance_estimate = quantizer
                            .l2_distance_estimate(&code, &query)
                            .expect("l2 estimate");
                        let code_norm_sq: f32 = code.polar_code.radii.iter().map(|v| v * v).sum();
                        let exact_inner_product: f32 = vector
                            .iter()
                            .zip(query.iter())
                            .map(|(lhs, rhs)| lhs * rhs)
                            .sum();
                        let exact_l2_distance: f32 = vector
                            .iter()
                            .zip(query.iter())
                            .map(|(lhs, rhs)| {
                                let diff = lhs - rhs;
                                diff * diff
                            })
                            .sum();

                        outputs.push(OutputCase {
                            name: input.name,
                            metric: input.metric,
                            dim: input.dim,
                            bits: input.bits as usize,
                            projections: input.projections,
                            seed: input.seed,
                            vector,
                            query,
                            radii: code.polar_code.radii,
                            angle_indices: code.polar_code.angle_indices,
                            residual_signs: code.residual_sketch.signs,
                            inner_product_estimate,
                            l2_distance_estimate,
                            code_norm_sq,
                            exact_inner_product,
                            exact_l2_distance,
                        });
                    }

                    println!("{}", serde_json::to_string_pretty(&outputs).expect("serialize"));
                }
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            ["cargo", "run", "--quiet"],
            cwd=tmpdir,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)


def render_float_list(values: list[float], total: int) -> str:
    padded = list(values) + [0.0] * (total - len(values))
    rendered: list[str] = []
    for value in padded:
        text = f"{value:.9g}"
        if "e" not in text and "." not in text:
            text += ".0"
        rendered.append(f"{text}f")
    return ", ".join(rendered)


def render_int_list(values: list[int], total: int, suffix: str = "") -> str:
    padded = list(values) + [0] * (total - len(values))
    return ", ".join(f"{value}{suffix}" for value in padded)


def write_header(results: list[dict], output_path: pathlib.Path) -> None:
    max_dim = max(item["dim"] for item in results)
    max_pairs = max(item["dim"] // 2 for item in results)
    max_projections = max(item["projections"] for item in results)

    lines: list[str] = [
        "// Generated by tests/unit/generate_tq_golden_header.py",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace tq_golden_fixture {",
        "",
        f"inline constexpr std::size_t kMaxDim = {max_dim};",
        f"inline constexpr std::size_t kMaxPairs = {max_pairs};",
        f"inline constexpr std::size_t kMaxProjections = {max_projections};",
        "",
        "struct OracleCase {",
        "    const char *name;",
        "    const char *metric;",
        "    std::size_t dim;",
        "    std::size_t bits;",
        "    std::size_t projections;",
        "    std::size_t seed;",
        "    std::array<float, kMaxDim> vector;",
        "    std::array<float, kMaxDim> query;",
        "    std::array<float, kMaxPairs> radii;",
        "    std::array<std::uint16_t, kMaxPairs> angle_indices;",
        "    std::array<std::int8_t, kMaxProjections> residual_signs;",
        "    float inner_product_estimate;",
        "    float l2_distance_estimate;",
        "    float code_norm_sq;",
        "    float exact_inner_product;",
        "    float exact_l2_distance;",
        "};",
        "",
        f"inline constexpr std::array<OracleCase, {len(results)}> kCases = {{",
    ]

    for item in results:
        lines.extend(
            [
                "    OracleCase{",
                f"        \"{item['name']}\",",
                f"        \"{item['metric']}\",",
                f"        {item['dim']},",
                f"        {item['bits']},",
                f"        {item['projections']},",
                f"        {item['seed']},",
                f"        {{{render_float_list(item['vector'], max_dim)}}},",
                f"        {{{render_float_list(item['query'], max_dim)}}},",
                f"        {{{render_float_list(item['radii'], max_pairs)}}},",
                f"        {{{render_int_list(item['angle_indices'], max_pairs)}}},",
                f"        {{{render_int_list(item['residual_signs'], max_projections)}}},",
                f"        {item['inner_product_estimate']:.9g}f,",
                f"        {item['l2_distance_estimate']:.9g}f,",
                f"        {item['code_norm_sq']:.9g}f,",
                f"        {item['exact_inner_product']:.9g}f,",
                f"        {item['exact_l2_distance']:.9g}f,",
                "    },",
            ]
        )

    lines.extend(["};", "", "} // namespace tq_golden_fixture", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    default_oracle = pathlib.Path("/Users/jeremy.plichta/git/turbo-quant-ref")
    oracle_repo = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default_oracle
    output_path = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).with_name("tq_golden_fixture.h")

    if not oracle_repo.exists():
        raise SystemExit(f"oracle repo not found: {oracle_repo}")
    if shutil.which("cargo") is None:
        raise SystemExit("cargo not found in PATH")

    results = run_oracle(oracle_repo)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header(results, output_path)
    rel = os.path.relpath(output_path, repo_root)
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

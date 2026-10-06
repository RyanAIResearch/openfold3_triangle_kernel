import ast
import hashlib
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import tools.manage_attested_sm90_wheel as artifact_manager
from tools.manage_attested_sm90_wheel import ArtifactError, discover_bundle
from tools.production_training import prepare_data

REPOSITORY = Path(__file__).resolve().parents[2]


def _manifest() -> dict:
    return tomllib.loads((REPOSITORY / "pixi.toml").read_text())


def test_h100_pixi_environment_is_non_editable_and_exactly_pinned():
    manifest = _manifest()
    environments = manifest["environments"]
    assert "openfold3-h100" not in environments
    features = environments["openfold3-cuda12-intree"]["features"]
    assert "turbopairformer-h100" in features
    assert "openfold3-editable" not in features
    assert "openfold3-editable-full" not in features
    assert "tests" not in features
    development_features = environments["openfold3-cuda12-dev"]["features"]
    assert "turbopairformer-h100" not in development_features
    assert "openfold3-editable" in development_features
    assert "tests" in development_features
    production = manifest["feature"]["turbopairformer-h100"]
    assert production["platforms"] == ["linux-64-cuda12"]
    assert production["dependencies"]["python"] == "3.14.*"
    assert production["dependencies"]["pytorch-gpu"] == "==2.10.0"
    assert production["dependencies"]["pytorch-lightning"] == "==2.6.1"
    assert production["dependencies"]["cuda-version"] == "12.9.*"
    production_wheel_path = (
        artifact_manager.PREBUILT_ARTIFACT_DIR / artifact_manager.PREBUILT_WHEEL
    ).relative_to(REPOSITORY)
    assert production["pypi-dependencies"]["openfold3"] == {
        "path": production_wheel_path.as_posix(),
    }

    lock = yaml.safe_load((REPOSITORY / "pixi.lock").read_text())
    locked_environments = lock["environments"]
    assert "openfold3-h100" not in locked_environments
    locked = locked_environments["openfold3-cuda12-intree"]
    production_wheel = "./" + production_wheel_path.as_posix()
    locked_pypi = []
    for packages in locked["packages"].values():
        for package in packages:
            if "pypi" in package:
                locked_pypi.append(package["pypi"])
                assert package["pypi"].startswith(("https://", "http://")) or (
                    package["pypi"] == production_wheel
                )
    assert locked_pypi.count(production_wheel) == 1
    development_locked = locked_environments["openfold3-cuda12-dev"]
    assert any(
        package.get("pypi") == "./"
        for packages in development_locked["packages"].values()
        for package in packages
    )


def test_cuequivariance_comparator_is_exactly_pinned_to_0_10_0():
    manifest = _manifest()
    for feature, platform, ops_package in (
        ("cuequivariance-cuda12", "*-cuda12", "cuequivariance-ops-torch-cu12"),
        ("cuequivariance-cuda13", "*-cuda13", "cuequivariance-ops-torch-cu13"),
    ):
        dependencies = manifest["feature"][feature]["target"][platform][
            "pypi-dependencies"
        ]
        assert dependencies == {
            "cuequivariance": "==0.10.0",
            "cuequivariance-torch": "==0.10.0",
            ops_package: "==0.10.0",
        }

    project = tomllib.loads((REPOSITORY / "pyproject.toml").read_text())
    cueq_extra = project["project"]["optional-dependencies"]["cuequivariance"]
    assert "cuequivariance==0.10.0" in cueq_extra
    assert "cuequivariance-torch==0.10.0" in cueq_extra
    assert "cuequivariance-ops-torch-cu12==0.10.0" in cueq_extra


def test_h100_pixi_tasks_keep_training_in_official_launchers():
    tasks = _manifest()["feature"]["turbopairformer-h100"]["tasks"]
    assert set(tasks) == {
        "doctor-h100",
        "doctor-hopper",
        "download-pdb-subset",
        "generate-pdb-subset",
        "purify-pdb-subset",
        "prepare-training-data",
        "prepare-pdb-subset",
        "prepare-crop-benchmark-data",
        "setup-pdb-subset",
        "setup-turbopairformer-h100",
        "verify-aot",
        "validate-training-data",
        "select-training-backend",
        "train-single-gpu",
        "train-single-node",
        "train-multinode",
        "benchmark-single-gpu",
        "benchmark-single-node",
        "benchmark-multinode",
        "plot-backend-comparison",
    }
    commands = {name: value["cmd"] for name, value in tasks.items() if "cmd" in value}
    assert commands["doctor-hopper"][-1] == "doctor-hopper"
    assert commands["doctor-h100"][-1] == "doctor-h100"
    assert commands["train-single-gpu"][-1] == "run_openfold"
    assert commands["train-single-node"][-1] == "run_openfold"
    assert commands["train-multinode"][-1] == "torchrun"
    assert commands["benchmark-single-gpu"][-1] == "tools/run_backend_benchmark.sh"
    assert commands["benchmark-single-node"][-1] == "tools/run_backend_benchmark.sh"
    assert commands["benchmark-multinode"][-1] == "tools/run_backend_benchmark.sh"
    assert commands["prepare-pdb-subset"][-1] == "tools/prepare_pdb_subset.py"
    assert commands["prepare-crop-benchmark-data"][-1] == (
        "tools/prepare_crop_benchmark_data.py"
    )
    assert commands["validate-training-data"][-1] == ("tools/validate_training_data.py")
    assert commands["plot-backend-comparison"][-1] == (
        "tools/plot_backend_comparison.py"
    )
    assert commands["setup-turbopairformer-h100"][3] == "verify-prebuilt"
    assert commands["verify-aot"][3] == "verify-prebuilt"
    assert not any("build" in " ".join(command) for command in commands.values())
    for name in ("train-single-gpu", "train-single-node", "train-multinode"):
        assert "production_training" not in " ".join(commands[name])


def test_h100_user_docs_use_normal_pixi_commands():
    for relative in (
        "README.md",
        "docs/source/bare_metal_distributed_quickstart.md",
    ):
        text = (REPOSITORY / relative).read_text()
        assert "--as-is" not in text
        assert "--locked" not in text
        assert "openfold3-h100" not in text
        assert "--frozen" not in text
        assert "pixi install -e openfold3-cuda12-intree" in text
        assert "pixi run -e openfold3-cuda12-intree" in text


def test_multinode_quickstart_uses_existing_runner_and_shared_rw_run_dir():
    text = (REPOSITORY / "docs/source/bare_metal_distributed_quickstart.md").read_text()
    assert "mkdir -p /shared/openfold3-run" in text
    assert "test -r /shared/openfold3-run && test -w /shared/openfold3-run" in text
    assert 'export RUNNER_YAML="$PWD/datasets/train_pdb_subset.${BACKEND}.yaml"' in text
    assert '--runner-yaml "$RUNNER_YAML"' in text
    assert '--runner-yaml "/shared/train_pdb_subset.${BACKEND}.yaml"' not in text
    assert "real shared, read-write filesystem" in text


def test_bare_metal_benchmark_prepares_compact_subset_atomically():
    text = (REPOSITORY / "docs/source/bare_metal_distributed_quickstart.md").read_text()
    assert text.count("--train-size 512") == 1
    assert "prepare-pdb-subset" in text
    assert "validate-training-data" in text
    assert "samples with replacement" in text
    assert "not model-quality or convergence" in text
    assert "prepare-crop-benchmark-data" in text
    assert "--crop-sizes 384 640 768" in text
    assert "provenance.json" in text
    assert "--train-size 4000" not in text
    assert "--train-size 1000" not in text


def test_bare_metal_sections_are_copy_paste_ready_for_multi_gpu_users():
    text = (REPOSITORY / "docs/source/bare_metal_distributed_quickstart.md").read_text()
    assert text.count("mkdir -p ../openfold3-run") >= 4
    assert "export DEVICES=8" in text
    assert "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in text
    assert "benchmark-single-node" in text
    assert "backend-comparison-1gpu/crop${CROP}" in text
    assert "backend-comparison-2gpu/crop${CROP}" in text
    assert "backend-comparison-8gpu/crop${CROP}" in text
    assert text.count("BACKENDS=(aqlab triton cueq turbopairformer)") >= 3
    assert text.count('for BACKEND in "${BACKENDS[@]}"') >= 3
    assert text.count("not plotting") >= 3
    assert text.count("if (( MATRIX_FAILED ))") >= 3
    assert text.count("summary.json") >= 4
    assert text.count("--num-workers 10") >= 4
    assert text.count("--require-full-crop") >= 4
    assert "--baseline triton" in text
    assert "Changing the worker count can therefore change the concrete" in text
    assert "not a per-step fingerprint" in text
    assert "two-node H100 qualification" not in text
    assert "TriOnce" not in text
    assert text.count("--use-tf32") >= 2
    assert "cuEquivariance" in text
    assert "0.10.0" in text


def test_dirty_pixi_lock_error_explains_repository_update(monkeypatch):
    monkeypatch.setattr(
        artifact_manager,
        "_git",
        lambda *args: " M pixi.lock",
    )

    with pytest.raises(ArtifactError) as caught:
        artifact_manager.source_commit()

    message = str(caught.value)
    assert "Dirty paths:\n M pixi.lock" in message
    assert "git restore --source=HEAD -- pixi.lock" in message
    assert "git pull --ff-only origin main" in message


def test_development_consumers_do_not_select_the_h100_runtime():
    for relative in (
        ".github/workflows/ci-test.yml",
        ".github/workflows/integration-test.yml",
        "docker/Dockerfile.pixi",
        "docker/DOCKER.md",
        "docs/source/training.md",
    ):
        text = (REPOSITORY / relative).read_text()
        assert "openfold3-cuda12-dev" in text


def test_official_launcher_preflights_data_and_aot_before_changing_cwd():
    wrapper = (REPOSITORY / "tools/run_official_training.sh").read_text()
    assert "tools.manage_attested_sm90_wheel verify-prebuilt" in wrapper
    assert "tools/validate_training_data.py" in wrapper
    assert "PYTHONPATH PYTHONHOME LD_PRELOAD" in wrapper
    assert '"$prefix/bin/"*' in wrapper
    assert 'exec "$launcher_path" "$@"' in wrapper
    for forbidden in (
        "Popen",
        "workflow_status",
        "metrics.csv",
        "launch_manifest",
        "rdzv",
        "max-restarts",
    ):
        assert forbidden not in wrapper


def test_backend_benchmark_wrapper_uses_installed_runtime_outside_source():
    wrapper = (REPOSITORY / "tools/run_backend_benchmark.sh").read_text()
    assert "tools.manage_attested_sm90_wheel verify-prebuilt" in wrapper
    assert "tools/validate_training_data.py" in wrapper
    assert "PYTHONPATH PYTHONHOME LD_PRELOAD" in wrapper
    assert "OF3_RUN_CWD must be outside the source checkout" in wrapper
    assert (
        '"$prefix/bin/python" "$source_root/tools/benchmark_training_backend.py"'
        in wrapper
    )


def test_prebuilt_bundle_is_checked_in_and_pinned_by_digest():
    artifact_dir = artifact_manager.PREBUILT_ARTIFACT_DIR
    assert artifact_dir.is_dir()
    assert {path.name for path in artifact_dir.iterdir() if path.is_file()} == set(
        artifact_manager.PREBUILT_FILE_SHA256
    )
    for name, expected in artifact_manager.PREBUILT_FILE_SHA256.items():
        digest = hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()
        assert digest == expected

    bundle = artifact_manager.prebuilt_bundle()

    assert bundle.manifest["git"] == {
        "commit": artifact_manager.PREBUILT_COMMIT,
        "dirty": False,
    }
    assert (
        bundle.wheel_sha256 == artifact_manager.PREBUILT_FILE_SHA256[bundle.wheel.name]
    )


def test_installed_wheel_verification_surfaces_child_diagnostic(monkeypatch):
    bundle = artifact_manager.prebuilt_bundle()
    monkeypatch.setattr(
        artifact_manager.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="installed AOT identity differs from selected bundle",
        ),
    )

    with pytest.raises(
        ArtifactError,
        match="installed AOT identity differs from selected bundle",
    ):
        artifact_manager.verify_installed(bundle, artifact_manager.PREBUILT_COMMIT)


def test_prebuilt_wheel_keeps_diffusion_on_the_aqlab_path():
    bundle = artifact_manager.prebuilt_bundle()
    member = "openfold3/projects/of3_all_atom/model.py"
    with zipfile.ZipFile(bundle.wheel) as archive:
        module = ast.parse(archive.read(member), filename=member)

    model = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "OpenFold3"
    )

    def method(name: str) -> ast.FunctionDef:
        return next(
            node
            for node in model.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def self_calls(function: ast.FunctionDef, attribute: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == attribute
        ]

    def keywords(call: ast.Call) -> dict[str, ast.expr]:
        return {
            keyword.arg: keyword.value
            for keyword in call.keywords
            if keyword.arg is not None
        }

    rollout_calls = self_calls(method("_rollout"), "sample_diffusion")
    assert len(rollout_calls) == 1
    rollout_keywords = keywords(rollout_calls[0])
    assert "use_stable_triangle_attention" not in rollout_keywords
    high_precision = rollout_keywords["use_high_precision_attention"]
    assert isinstance(high_precision, ast.Constant) and high_precision.value is True

    training_calls = self_calls(method("_train_diffusion"), "diffusion_module")
    assert len(training_calls) == 1
    training_keywords = keywords(training_calls[0])
    for backend in (
        "use_deepspeed_evo_attention",
        "use_triton_triangle_kernels",
        "use_cueq_triangle_kernels",
        "use_stable_triangle_attention",
        "use_lma",
    ):
        assert backend not in training_keywords
    high_precision = training_keywords["use_high_precision_attention"]
    assert isinstance(high_precision, ast.Constant) and high_precision.value is True


def test_prebuilt_is_the_only_path_allowed_to_decouple_checkout_docs(monkeypatch):
    sha256_file = artifact_manager.sha256_file
    monkeypatch.setattr(
        artifact_manager,
        "sha256_file",
        lambda path: (
            "0" * 64 if path == REPOSITORY / "README.md" else sha256_file(path)
        ),
    )
    with pytest.raises(ArtifactError, match="packaged source receipt mismatch"):
        discover_bundle(
            artifact_manager.PREBUILT_ARTIFACT_DIR,
            expected_commit=artifact_manager.PREBUILT_COMMIT,
        )

    assert artifact_manager.prebuilt_bundle().wheel.is_file()


def test_prebuilt_bundle_rejects_a_pinned_file_digest_change(monkeypatch):
    monkeypatch.setitem(
        artifact_manager.PREBUILT_FILE_SHA256,
        "RECEIPT.json",
        "0" * 64,
    )

    with pytest.raises(ArtifactError, match="digest mismatch: RECEIPT.json"):
        artifact_manager.prebuilt_bundle()


def test_prepare_data_does_not_materialize_or_modify_a_runner_yaml(tmp_path):
    work_root = tmp_path / "work"
    cache = work_root / "pdb_training_set" / "dataset_caches"
    cache.mkdir(parents=True)
    (cache / "training_cache_with_templates.json").write_text("{}\n")
    (cache / "validation_cache_with_templates.json").write_text("{}\n")

    result = prepare_data(
        SimpleNamespace(
            work_root=str(work_root),
            data_uri="unused",
            skip_download=True,
        )
    )

    assert result["status"] == "PASS"
    assert result["runner_yaml_modified"] is False
    assert not (work_root / "run" / "base.yml").exists()


def test_artifact_discovery_rejects_ambiguous_wheels(tmp_path):
    (tmp_path / "openfold3-one.whl").write_bytes(b"one")
    (tmp_path / "openfold3-two.whl").write_bytes(b"two")
    with pytest.raises(ArtifactError, match="exactly one wheel"):
        discover_bundle(tmp_path)


def test_artifact_discovery_rejects_an_incomplete_bundle(tmp_path):
    (tmp_path / "openfold3-only.whl").write_bytes(b"wheel")
    with pytest.raises(ArtifactError, match="incomplete"):
        discover_bundle(tmp_path)


def test_aot_identity_treats_symbol_order_as_non_semantic():
    first = {
        "extensions": {
            "turbopairformer_forward": {
                "sha256": "binary",
                "symbols": ["forward", "pack", "shadow"],
            }
        }
    }
    reordered = {
        "extensions": {
            "turbopairformer_forward": {
                "sha256": "binary",
                "symbols": ["shadow", "forward", "pack"],
            }
        }
    }
    changed = {
        "extensions": {
            "turbopairformer_forward": {
                "sha256": "binary",
                "symbols": ["forward", "pack"],
            }
        }
    }

    assert artifact_manager._canonical_aot_identity(
        first
    ) == artifact_manager._canonical_aot_identity(reordered)
    assert artifact_manager._canonical_aot_identity(
        first
    ) != artifact_manager._canonical_aot_identity(changed)


def test_artifact_publish_accepts_a_precreated_empty_directory(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    paths = tuple(
        source / name
        for name in (
            "openfold3-test.whl",
            "openfold3-test.whl.sha256",
            "openfold3-test.whl.attestation.json",
            "aot_clean_install_smoke.json",
        )
    )
    for index, path in enumerate(paths):
        path.write_text(str(index))
    bundle = artifact_manager.ArtifactBundle(
        wheel=paths[0],
        checksum=paths[1],
        attestation=paths[2],
        smoke=paths[3],
        wheel_sha256="test",
        manifest={},
        receipt={},
        installed_distribution={},
    )
    destination = tmp_path / "artifacts"
    destination.mkdir()
    monkeypatch.setattr(
        artifact_manager,
        "discover_bundle",
        lambda *args, **kwargs: bundle,
    )

    artifact_manager._copy_bundle(bundle, destination, "commit")

    assert {path.name for path in destination.iterdir()} == {
        path.name for path in paths
    }

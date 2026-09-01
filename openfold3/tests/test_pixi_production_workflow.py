import hashlib
import tomllib
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
    features = manifest["environments"]["openfold3-h100"]["features"]
    assert "stablefold-h100" in features
    assert "openfold3-editable" not in features
    assert "openfold3-editable-full" not in features
    production = manifest["feature"]["stablefold-h100"]
    assert production["platforms"] == ["linux-64-cuda12"]
    assert production["dependencies"]["python"] == "3.14.*"
    assert production["dependencies"]["pytorch-gpu"] == "==2.10.0"
    assert production["dependencies"]["pytorch-lightning"] == "==2.6.1"
    assert production["dependencies"]["cuda-version"] == "12.9.*"

    lock = yaml.safe_load((REPOSITORY / "pixi.lock").read_text())
    locked = lock["environments"]["openfold3-h100"]
    for packages in locked["packages"].values():
        for package in packages:
            if "pypi" in package:
                assert package["pypi"].startswith(("https://", "http://"))


def test_h100_pixi_tasks_keep_training_in_official_launchers():
    tasks = _manifest()["feature"]["stablefold-h100"]["tasks"]
    assert set(tasks) == {
        "doctor-h100",
        "download-pdb-subset",
        "generate-pdb-subset",
        "prepare-training-data",
        "setup-pdb-subset",
        "setup-stablefold-h100",
        "verify-aot",
        "select-training-backend",
        "train-single-gpu",
        "train-single-node",
        "train-multinode",
    }
    commands = {name: value["cmd"] for name, value in tasks.items() if "cmd" in value}
    assert commands["train-single-gpu"][-1] == "run_openfold"
    assert commands["train-single-node"][-1] == "run_openfold"
    assert commands["train-multinode"][-1] == "torchrun"
    assert commands["setup-stablefold-h100"][3] == "install-prebuilt"
    assert commands["verify-aot"][3] == "verify-prebuilt"
    assert not any("build" in " ".join(command) for command in commands.values())
    for name in ("train-single-gpu", "train-single-node", "train-multinode"):
        assert "production_training" not in " ".join(commands[name])


def test_official_launcher_wrapper_only_preflights_aot_and_changes_cwd():
    wrapper = (REPOSITORY / "tools/run_official_training.sh").read_text()
    assert "tools.manage_attested_sm90_wheel verify-prebuilt" in wrapper
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


def test_prebuilt_is_the_only_path_allowed_to_decouple_checkout_docs():
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
            "stablefold_forward": {
                "sha256": "binary",
                "symbols": ["forward", "pack", "shadow"],
            }
        }
    }
    reordered = {
        "extensions": {
            "stablefold_forward": {
                "sha256": "binary",
                "symbols": ["shadow", "forward", "pack"],
            }
        }
    }
    changed = {
        "extensions": {
            "stablefold_forward": {
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

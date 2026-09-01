# Bare-metal distributed quickstart

1. **Prepare data:** On a fresh Linux H100 host, install Pixi, clone the repository, create a shared run directory, download the published preprocessed PDB training set, and copy the ready PDB-only runner YAML with the commands below.

   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   export PATH="$HOME/.pixi/bin:$PATH"
   git clone https://github.com/RyanAIResearch/openfold3_triangle_kernel.git /mnt/shared/openfold3/src
   cd /mnt/shared/openfold3/src
   pixi install -e openfold3-cuda12
   mkdir -p /mnt/shared/openfold3/run
   pixi run -e openfold3-cuda12 aws s3 sync \
     s3://openfold3-data/pdb_training_set/ \
     /mnt/shared/openfold3/pdb_training_set/ --no-sign-request
   cp examples/training_yamls/stablefold_distributed_pdb.yml \
     /mnt/shared/openfold3/run/training.yml
   ```

2. **Build kernels:** From the clean committed checkout, build all five `sm_90a` AOT kernels once, verify the emitted wheel bundle, and install that exact non-editable wheel once into the shared immutable Pixi prefix—or repeat only the install command on every node when prefixes are node-local—with the commands below.

   ```bash
   cd /mnt/shared/openfold3/src
   pixi run -e openfold3-cuda12 bash -lc \
     'export CUDA_HOME="$CONDA_PREFIX" CUDACXX="$CONDA_PREFIX/bin/nvcc"; tools/build_sm90_wheel.sh'
   (cd dist && sha256sum --check openfold3-*.whl.sha256)
   /mnt/shared/openfold3/src/.pixi/envs/openfold3-cuda12/bin/python -m pip \
     install --no-deps --force-reinstall dist/openfold3-*.whl
   ```

3. **Choose an optimization and launch training:** Select exactly one training backend—`aqlab` for native AQLab/OpenFold3 stock PyTorch, `cueq` for cuEquivariance, `stablefold` for StableFold Attention plus TriMul, or `deepspeed-evo` for the DeepSpeed Evo attention operator—then activate the locked Pixi environment and run the commands below on every node after replacing `MASTER_ADDR` with node 0's routable address and setting `NODE_RANK` to `0,1,...,NNODES-1`, noting that the operator choice is independent of Lightning DeepSpeedStrategy or ZeRO, that YAML `devices` and `num_nodes` must equal `NPROC_PER_NODE` and `NNODES`, and that this enforced production-candidate path still awaits real 2-GPU and 2-node hardware qualification.

   ```bash
   BACKEND=stablefold  # aqlab | cueq | stablefold | deepspeed-evo
   /mnt/shared/openfold3/src/.pixi/envs/openfold3-cuda12/bin/python \
     /mnt/shared/openfold3/src/tools/select_training_backend.py \
     /mnt/shared/openfold3/run/training.yml --backend "$BACKEND" \
     --output-dir "/mnt/shared/openfold3/runs/$BACKEND" \
     --devices-per-node 8 --num-nodes 2 --epoch-len 128000 --fresh
   eval "$("$HOME/.pixi/bin/pixi" shell-hook \
     --manifest-path /mnt/shared/openfold3/src/pixi.toml \
     --environment openfold3-cuda12 --as-is)"
   cd /mnt/shared/openfold3/run
   export NNODES=2
   export NPROC_PER_NODE=8
   export NODE_RANK=0                  # use 1 on node 1
   export MASTER_ADDR=10.0.0.10        # node 0's routable address
   export MASTER_PORT=29500
   export OPENFOLD3_TRIANGLE_ALLOW_JIT=0
   export OPENFOLD3_SR_BACKWARD=1
   export OPENFOLD3_SR_SHADOW=1
   export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
   /mnt/shared/openfold3/src/.pixi/envs/openfold3-cuda12/bin/torchrun \
     --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" \
     --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" \
     --master-port="$MASTER_PORT" --no-python \
     /mnt/shared/openfold3/src/.pixi/envs/openfold3-cuda12/bin/run_openfold \
     train --runner-yaml /mnt/shared/openfold3/run/training.yml --seed 42
   ```

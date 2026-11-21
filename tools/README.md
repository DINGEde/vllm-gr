# Repository tools

Run these tools from the repository root. For supported environments and test
selection, see the [validation guide](../docs/contributing/tools/README.md).

## Run tests

| Entry point | When to use it |
| --- | --- |
| [tools/run_test.sh](run_test.sh) | Run the local suite on a prepared CUDA or Ascend host. The runner selects a device, sets up the environment, and accepts `--wheel` to add a wheel smoke test. |
| [tools/run_act.sh](run_act.sh) | Run GitHub Actions locally on a CUDA host with Docker and `act`. See the [local workflow tester](../docs/contributing/tools/README.md#github-actions-tester). |
| [tools/pre_commit/run_tests.sh](pre_commit/run_tests.sh) | The `run-tests` hook runs the full pytest suite with slow tests. Setting `VLLM_GR_TEST_LEVEL` delegates to the CI level runner. |
| [tools/ci/run_test_level.sh](ci/run_test_level.sh) | Run the tests for one CI level, filtered by the selected platform. Use `bash tools/ci/run_test_level.sh pr` for L1 and L2. |
| [tools/ci/run_resource_level.sh](ci/run_resource_level.sh) | Run the device discovery and locking probe on an isolated runner with `VLLM_GR_ISOLATED_RUNNER=true`. |

## Build, benchmark and prepare devices

| Task | Entry points and requirements |
| --- | --- |
| Build and check a wheel | Use [scripts/build_wheel.sh](../scripts/build_wheel.sh) to build the distribution. Use the [wheel smoke test](packaging/run_wheel_smoke_test.sh) to rebuild the wheel and verify its installation outside the checkout. After a clean build, the [wheel contents check](packaging/check_wheel_contents.py) compares it with a wheel built from Git HEAD. See [packaging commands](../docs/contributing/tools/README.md#packaging). |
| Select or lock a device | Source [tools/devices/resource.sh](devices/resource.sh) to load the CUDA or Ascend helper functions. See [resource selection and lock semantics](../docs/contributing/tools/README.md#resource-helper). |
| Prepare Ascend operators | The NPU test runner calls [install_xllm_ops.sh](npu/install_xllm_ops.sh) to check for xllm-ops and build it when missing. Use the matching [Ascend runtime](../docs/getting_started/installation/ascend.md). |
| Build a constraint table | Convert a catalog to triples with [build_constraint_triples.py](constraints/build_constraint_triples.py), then compile an artifact for your model and tokenizer with [build_constraint_table.py](constraints/build_constraint_table.py). See the [OneRec quickstart](../docs/getting_started/quickstart.md). |
| Generate gRPC bindings | Run [generate_grpc_proto.py](codegen/generate_grpc_proto.py) with the pinned `grpcio-tools` from the development dependencies. Pass `--check` to verify the checked-in bindings. |
| Measure offline A/B performance | Follow the [A/B guide](AB_Tests/README.md) to configure the benchmark server, CUDA container, model and dataset, then run the paired benchmark. |
| Prepare a Pangu SID checkpoint | Use [prepare_pangu_sid.py](model_conversion/prepare_pangu_sid.py) to extend public openPangu weights for SID execution smoke tests. See the [Pangu guide](../docs/contributing/tools/pangu1b.md) for the pinned snapshot and CPU dependencies. |

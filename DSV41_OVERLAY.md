# Image-pinned V4.1 / V12 API integration worktree

This tree starts with the Python source extracted from 0xSero's pinned base image,
with the recipe's supplied `flash_mla_sm120.py` replacement. It is not a checkout
of current upstream SGLang, and its original Git revision is unresolved.

- Recipe: `0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000` at
  `45f538a18569420721e00e37353f9a7e1af7e5da`.
- Base image: `lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5`.
- Source layer: `sha256:52f078c87b74c74ff1f1a469469a62031ce00c710700b2f5c278ad791961ab08`.
- API behavior/tests donor: local V12 fork at
  `71c498b991ce143999132aafd951d16d6b119805`, relative to
  `2c03d62532c880d17ef356c24275d1f7117c1d69`.

The Engram adapter/import hooks and all binary dependencies remain supplied by
the pinned recipe/image. Keep the overlay limited to reviewed API/parser files;
do not replace kernels or dependency versions as part of this API port.

The initial import commit records provenance, not hardware qualification. Prepare
and run CPU regressions independently, then qualify on top of the reproduced
four-GPU baseline. Full deployment status and recipes are maintained in
`../rtx-pro-6000-bench/docker/deepseek-v41/`.

# MLPerf Inference Vision - extended for Object Detection - CmdGen

CmdGen maps high-level CmdGen commands to low-level CK commands.

## Example

A high-level command:

```
"ck gen cmdgen:benchmark.mlperf-inference-vision --sut=chai \
  --scenario=offline --mode=accuracy --dataset_size=50 \
  --model=yolo-v3-coco --library=tensorflow-v2.6.0-cpu"
```

gets mapped to:

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
    --env.CK_LOADGEN_SCENARIO=Offline \
    --env.CK_LOADGEN_MODE='--accuracy' \
    --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
    \
    --dep_add_tags.weights=yolo-v3-coco \
    --env.CK_MODEL_PROFILE=tf_yolo \
    \
    --env.CK_INFERENCE_ENGINE=tensorflow \
    --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu\
    --env.CUDA_VISIBLE_DEVICES=-1"
```

## Save experimental results into a host directory

The user should belong to the group `krai` on the host machine.
If it does not exist:

```
sudo groupadd krai
sudo usermod -aG krai $USER
```

### Create a new repository

```
ck add repo:ck-object-detection.$(hostname).$(id -un) --quiet && \
ck add ck-object-detection.$(hostname).$(id -un):experiment:dummy --common_func && \
ck rm  ck-object-detection.$(hostname).$(id -un):experiment:dummy --force
```

### Make its `experiment` directory writable by group `krai`

```
export CK_EXPERIMENT_DIR="$HOME/CK/ck-object-detection.$(hostname).$(id -un)/experiment"
sudo chgrp krai $CK_EXPERIMENT_DIR -R && sudo chmod g+w $CK_EXPERIMENT_DIR -R
```

### Run

#### Run a CmdGen command from a Docker command

```
export CK_IMAGE="krai/mlperf-inference-vision-with-ck.tensorrt:21.09-py3_tf-2.6.0"
export CK_EXPERIMENT_DIR="$HOME/CK/ck-object-detection.$(hostname).$(id -un)/experiment"
docker run --user=krai:kraig --group-add $(cut -d: -f3 < <(getent group krai)) \
--volume ${CK_EXPERIMENT_DIR}:/home/krai/CK_REPOS/local/experiment --rm ${CK_IMAGE} \
"ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--scenario=offline --mode=accuracy --dataset_size=50 --buffer_size=64 \
--model=yolo-v3-coco --library=tensorflow-v2.6.0-cpu --sut=chai"
```

#### Run a Docker command from a CmdGen command

```
export CK_IMAGE="krai/mlperf-inference-vision-with-ck.tensorrt:21.09-py3_tf-2.6.0"
export CK_EXPERIMENT_DIR="$HOME/CK/ck-object-detection.$(hostname).$(id -un)/experiment"
ck run cmdgen:benchmark.mlperf-inference-vision --verbose \
--docker --docker_image=${CK_IMAGE} --experiment_dir=${CK_EXPERIMENT_DIR} \
--scenario=offline --mode=accuracy --dataset_size=50 --buffer_size=64 \
--model=yolo-v3-coco --library=tensorflow-v2.6.0-cpu --sut=chai
```

---
---

## Mappings

### Mapping for `--scenario`

It will affect the following flags in the CK environment:

```
--env.CK_LOADGEN_SCENARIO=[SCENARIO]
```
|SCENARIO|
|---|
| `SingleStream` |
| `MultiStream` | 
| `Server` |
| `Offline` |

---
---

### Mapping for `--mode` and `--mode-param`

`mode` for specifying mode and `mode-param` for specifying count or qps. 
It will affect the following flags in the CK environment:
```
--env.CK_LOADGEN_MODE
--env.CK_LOADGEN_EXTRA_PARAMS
--env.CK_OPTIMIZE_GRAPH
```

| Accuracy Mode | Performance Mode |
| --- | ---|
|`--env.CK_LOADGEN_MODE='--accuracy'` <br> `--env.CK_LOADGEN_EXTRA_PARAMS='--count 200 --max-query-count 200'` | `--env.CK_LOADGEN_EXTRA_PARAMS='--count 200 --performance-sample-count 200 --qps 3'` <br> `--env.CK_OPTIMIZE_GRAPH='True'`|

---
---

### Mapping for `--model`

It will affect the following flags in the ck environment:
```
--dep_add_tags.weights=[MODEL_NAME]
--env.CK_MODEL_PROFILE=[MODEL_PROFILE]
--env.CK_INFERENCE_ENGINE=[INFERENCE_ENGINE] (as shown in README.md)
--env.CK_INFERENCE_ENGINE_BACKEND=[INFERENCE_ENGINE_BACKEND] (as shown in README.md)
--env.CUDA_VISIBLE_DEVICES=[DEVICE_NUMBER] (will be discussed in the next section)
```

See README.md for the table.

---
---

## Mapping for `--library`
It will affect the following flags in the ck environment:
```
--env.CK_INFERENCE_ENGINE=[INFERENCE_ENGINE]
--env.CK_INFERENCE_ENGINE_BACKEND=[INFERENCE_ENGINE_BACKEND]
--env.CUDA_VISIBLE_DEVICES=[DEVICE_IDS]
```

|INFERENCE_ENGINE|INFERENCE_ENGINE_BACKEND|DEVICE_IDS|
|---|---|---|
|`tensorflow` |`default-cpu` |`-1`|
|`tensorflow` |`default-gpu` |`0`|
|`tensorflow` |`openvino-cpu`|`-1`|
|`tensorflow` |`openvino-gpu` |`-1` for an Intel chip with an integrated GPU; `0` for an Intel GPU|

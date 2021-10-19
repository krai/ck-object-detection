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
export CK_EXPERIMENTS="$HOME/CK/ck-object-detection.$(hostname).$(id -un)/experiment"
sudo chgrp krai $CK_EXPERIMENTS -R && chmod g+ws $CK_EXPERIMENTS -R
```

### Run

```
export CK_IMAGE="krai/mlperf-inference-vision-with-ck.tensorrt:21.09-py3_tf-2.6.0"
```

#### Run a CmdGen command from a Docker command

```
docker run --user=krai:kraig --group-add $(cut -d: -f3 < <(getent group krai)) \
--volume ${CK_EXPERIMENTS}:/home/krai/CK_REPOS/local/experiment --rm ${CK_IMAGE} \
"ck run cmdgen:benchmark.mlperf-inference-vision \
--scenario=offline --mode=accuracy --dataset_size=50 --buffer_size=64 \
--model=yolo-v3-coco --library=tensorflow-v2.6.0-cpu --sut=chai"
```

#### Run a Docker command from a CmdGen command [work-in-progress]

```
ck run cmdgen:benchmark.mlperf-inference-vision \
--docker --docker_image=${CK_IMAGE} --experiments-dir=${CK_EXPERIMENTS} \
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

| MODEL_NAME | MODEL_PROFILE |
| --- | --- | 
|`rcnn-nas-lowproposals-coco`|`default_tf_object_det_zoo`| 
|`rcnn-resnet50-lowproposals-coco`| `default_tf_object_det_zoo`|  
|`rcnn-resnet101-lowproposals-coco`| `default_tf_object_det_zoo`| 
|`rcnn-inception-resnet-v2-lowproposals-coco`| `default_tf_object_det_zoo`| 
|`rcnn-inception-v2-coco`|`default_tf_object_det_zoo`| 
|`ssd-inception-v2-coco`|`default_tf_object_det_zoo`|
|`ssd_mobilenet_v1_coco`|`default_tf_object_det_zoo`| 
|`ssd_mobilenet_v1_quantized_coco`|`default_tf_object_det_zoo`| 
|`ssd-mobilenet-v1-fpn-sbp-coco`|`default_tf_object_det_zoo`| 
|`ssd-resnet50-v1-fpn-sbp-coco`|`default_tf_object_det_zoo`| 
|`ssdlite-mobilenet-v2-coco`|`default_tf_object_det_zoo`|
|`yolo-v3-coco`|`tf_yolo`| 

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

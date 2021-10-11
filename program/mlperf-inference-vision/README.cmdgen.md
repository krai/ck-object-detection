## Overview of cmdgen
Cmdgen:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run cmdgen:benchmark.mlperf-inference-vision \
    --scenario=offline --mode=accuracy --mode-param=50 \
    --model=yolo-v3-coco \
    --library=tensorflow-v2.6.0-cpu"
```
What it actually represents in ck environment:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
    --env.CK_LOADGEN_SCENARIO=Offline \
    --env.CK_LOADGEN_MODE='--accuracy' \
    --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
    \
    --dep_add_tags.weights=yolo-v3-coco \
    --env.CK_LOADGEN_REF_PROFILE=tf_yolo \
    --env.CK_METRIC_TYPE=COCO \
    --env.CK_INFERENCE_ENGINE=tensorflow \
    \
    --dep_add_tags.inference-engine-backend=vpip\
    --env.CUDA_VISIBLE_DEVICES=-1 \
    \
    --skip_print_timers"
```

---
---
## Mapping for `--scenario`
It will affect the following flags in the ck environment:
```
--env.CK_LOADGEN_SCENARIO=[SCENARIO]
```
|SCENARIO|
|---|
| `SingleStrem` |
| `MultiStream` | 
| `Server` |
| `Offline` |

---
---

## Mapping for `--mode` and `--mode-param`
`mode` for specifying mode and `mode-param` for specifying count or qps. 
It will affect the following flags in the ck environment:
```
--env.CK_LOADGEN_MODE
--env.CK_LOADGEN_EXTRA_PARAMS
--env.CK_OPTIMIZE_GRAPH
```

| Accuracy Mode | Performance Mode |
| --- | ---|
|`--env.CK_LOADGEN_MODE='--accuracy'` <br> `--env.CK_LOADGEN_EXTRA_PARAMS='--count 50'` | `--env.CK_LOADGEN_EXTRA_PARAMS='--qps 30'` <br> `--env.CK_OPTIMIZE_GRAPH='True'`|

---
---

## Mapping for `--model`
It will affect the following flags in the ck environment:
```
--dep_add_tags.weights=[MODEL_NAME]
--env.CK_LOADGEN_REF_PROFILE=[LOADGEN_PROFILE]
--env.CK_METRIC_TYPE=[DATA_TYPE]
--env.CK_LOADGEN_BACKEND=[FRAMEWORK]
```

| MODEL_NAME | LOADGEN_PROFILE | DATA_TYPE | FRAMEWORK |
| --- | --- | --- | --- |
|`rcnn-nas-lowproposals-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`rcnn-resnet50-lowproposals-coco`| `default_tf_object_det_zoo`|  `COCO` | `tensorflow`, `openvino-cpu`|
|`rcnn-resnet101-lowproposals-coco`| `default_tf_object_det_zoo`| `COCO` | `tensorflow`, `openvino-cpu`|
|`rcnn-inception-resnet-v2-lowproposals-coco`| `default_tf_object_det_zoo`| `COCO` | `tensorflow`, (`openvino-cpu` maybe very slow) |
|`rcnn-inception-v2-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow`, `vopenvino`|
|`ssd-inception-v2-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`ssd_mobilenet_v1_coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`ssd_mobilenet_v1_quantized_coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`ssd-mobilenet-v1-fpn-sbp-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow`|
|`ssd-resnet50-v1-fpn-sbp-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow`|
|`ssdlite-mobilenet-v2-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow`|
|`yolo-v3-coco`|`tf_yolo`| `COCO` | `tensorflow`, `openvino-cpu`|

Comments: Added `ssd_mobilenet_v1_quantized_coco`.

---
---

## Mapping for `--library`
It will affect the following flags in the ck environment:
```
--env.CK_LOADGEN_BACKEND=[INFERENCE]
--dep_add_tags.[INFERENCE] = [INFERENCE_DETAILS]
```

|INFERENCE|INFERENCE_DETAILS|
|---|---|
|`tensorflow` |`vpip` |
|`openvino-cpu` |`vopenvino`|
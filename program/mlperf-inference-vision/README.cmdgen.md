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
    \
    --env.CK_INFERENCE_ENGINE=tensorflow \
    --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu\
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
--env.CK_INFERENCE_ENGINE=[INFERENCE_ENGINE]
--env.CK_INFERENCE_ENGINE_BACKEND=[INFERENCE_ENGINE_BACKEND]
--env.CUDA_VISIBLE_DEVICES=[DEVICE_NUMBER] (will be discussed in the next section)
```

| MODEL_NAME | LOADGEN_PROFILE | INFERENCE_ENGINE:INFERENCE_ENGINE_BACKEND |
| --- | --- | --- |
|`rcnn-nas-lowproposals-coco`|`default_tf_object_det_zoo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu`, `tensorflow`:`openvino-cpu` |
|`rcnn-resnet50-lowproposals-coco`| `default_tf_object_det_zoo`|  `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu`, `tensorflow`:`openvino-cpu`|
|`rcnn-resnet101-lowproposals-coco`| `default_tf_object_det_zoo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu`, `tensorflow`:`openvino-cpu`|
|`rcnn-inception-resnet-v2-lowproposals-coco`| `default_tf_object_det_zoo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu` , `tensorflow`:`openvino-cpu` |
|`rcnn-inception-v2-coco`|`default_tf_object_det_zoo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu` , `tensorflow`:`openvino-cpu`|
|`ssd-inception-v2-coco`|`default_tf_object_det_zoo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu`  |
|`ssd_mobilenet_v1_coco`|`default_tf_object_det_zoo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu`  |
|`ssd_mobilenet_v1_quantized_coco`|`default_tf_object_det_zoo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu`  |
|`ssd-mobilenet-v1-fpn-sbp-coco`|`default_tf_object_det_zoo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu` |
|`ssd-resnet50-v1-fpn-sbp-coco`|`default_tf_object_det_zoo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu` |
|`ssdlite-mobilenet-v2-coco`|`default_tf_object_det_zoo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu` |
|`yolo-v3-coco`|`tf_yolo`| `tensorflow`:`default-cpu`,`tensorflow`:`default-gpu`, `tensorflow`:`openvino-cpu`|

---
---

## Mapping for `--library`
It will affect the following flags in the ck environment:
```
--env.CK_INFERENCE_ENGINE =[INFERENCE_ENGINE]
--env.CK_INFERENCE_ENGINE_BACKEND = [INFERENCE_ENGINE_BACKEND]
--env.CUDA_VISIBLE_DEVICES=[DEVICE_NUMBER]
```

|INFERENCE_ENGINE|INFERENCE_ENGINE_BACKEND|DEVICE_NUMBER|
|---|---|---|
|`tensorflow` |`default-cpu` |`-1`|
|`tensorflow` |`default-gpu` |`0`|
|`tensorflow` |`openvino-cpu`|`-1`|
|`tensorflow` |`openvino-gpu` |`-1` for Intel CPU with Intergrated GPU; `0` for Intel GPU|
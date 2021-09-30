## Overview of cmdgen
```
time docker run -it --rm ${CK_IMAGE} \
"ck run cmdgen:benchmark.mlperf-inference-vision \
--scenario=offline --mode=accuracy --mode_param=50 \
--model=ssd-mobilenet-v1-fpn --library=tensorflow-v2.6.0-cpu"
```


## Mapping for `--scenario`


## Mapping for `--model`
It will affect the following flags in the ck environment:
```
--dep_add_tags.weights=[MODEL_NAME] \
--env.CK_LOADGEN_REF_PROFILE=[LOADGEN_PROFILE] \
--env.CK_METRIC_TYPE=[DATA_TYPE] 
--env.CK_LOADGEN_BACKEND=[FRAMEWORK]
```

| MODEL_NAME | LOADGEN_PROFILE | DATA_TYPE | FRAMEWORK |
| --- | --- | --- | --- |
|`ssd-inception-v2-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`rcnn-inception-v2-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`ssdlite-mobilenet-v2-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`rcnn-resnet101-lowproposals-coco`| `default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`rcnn-inception-resnet-v2-lowproposals-coco`| `default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`ssd-mobilenet-v1-fpn-sbp-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`rcnn-resnet50-lowproposals-coco`| `default_tf_object_det_zoo`|  `COCO` | `tensorflow` |
|`ssd-resnet50-v1-fpn-sbp-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`rcnn-nas-lowproposals-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`rcnn-nas-coco`|`default_tf_object_det_zoo`| `COCO` | `tensorflow` |
|`yolo-v3-coco`|`tf_yolo`| `COCO` | `tensorflow` |


## Mapping for `--library`
Given that the `--env.CK_LOADGEN_BACKEND=tensorflow `
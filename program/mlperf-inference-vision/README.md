# TensorFlow object-detection program

# Build the environment:

## 1) Docker
Build the docker image and container with the build file from `ck-mlperf/docker/mlperf-inference-vision-with-ck.tensorrt/build.sh`. Also, set the image name:
```
export CK_IMAGE="krai/mlperf-inference-vision-with-ck.tensorrt:21.08-py3_tf-2.6.0"
```

### a) Just run the docker AND execute the intended ck command
Following the format below:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision ... "
```
Quick Example:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_LOADGEN_REF_PROFILE=tf_yolo \
  --env.CK_METRIC_TYPE=COCO \
  --env.CK_LOADGEN_BACKEND=tensorflow \
  --dep_add_tags.lib-tensorflow=vpip\
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --skip_print_timers"
```

### b) To run custom command and edit the environment
use the following to create a dummy container `ck`
```
docker run -td --runtime=nvidia --entrypoint /bin/bash --name ck [IMAGE ID]
```
Getting into the container
```
docker exec -it ck /bin/bash
```
Stopping the container
```
docker stop ck
```
Remove the container
```
docker rm ck
```

## 2) Locally

### Repositories

```bash
$ ck pull repo:ck-object-detection
$ ck pull repo:ck-tensorflow
```

### TensorFlow

Install from source:
```bash
$ ck install package:lib-tensorflow-1.10.1-src-{cpu,cuda}
```
or from a binary `x86_64` package:
```bash
$ ck install package:lib-tensorflow-1.10.1-{cpu,cuda}
```

Or you can choose from different available version of TensorFlow packages:
```bash
$ ck install package --tags=lib,tensorflow
```

### TensorFlow models
```bash
$ ck install ck-tensorflow:package:tensorflowmodel-api
```

Install one or more object detection model package:
```bash
$ ck install package --tags=tensorflowmodel,object-detection

 0) tensorflowmodel-object-detection-ssd-resnet50-v1-fpn-sbp-640x640-coco  Version 20170714  (09baac5e6f931db2)
 1) tensorflowmodel-object-detection-ssd-mobilenet-v1-coco  Version 20170714  (385831f88e61be8c)
```

### Datasets
```bash
$ ck install package --tags=dataset,object-detection
```

**NB:** If you have previously installed the `coco` dataset, you should probably renew them:
```bash
$ ck refresh env:{dataset-env-uoa}
```
where `dataset-env-uoa` is one of the env identifiers returned by:
```bash
$ ck show env --tags=dataset,coco
```

---
---

# Running 
## 0) General Form:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct 
  # Model_Specifications
  --dep_add_tags.weights=[MODEL_NAME] \
  --env.CK_LOADGEN_REF_PROFILE=[LOADGEN_PROFILE] \
  --env.CK_METRIC_TYPE=[DATA_TYPE] \

  # Backend_Specifications
  --env.CK_LOADGEN_BACKEND=tensorflow \
  --dep_add_tags.lib-tensorflow=vpip \

  # Scenario_Specifications
  --env.CK_LOADGEN_SCENARIO=Offline \

  # Mode_Specifications
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \

  # Specify CPU or GPU
  --env.CUDA_VISIBLE_DEVICES=-1 \
  
  # Others
  --skip_print_timers"
```


## 1) With Different Model
Change the `--dep_add_tags.weights`, `--env.CK_LOADGEN_REF_PROFILE`, `--env.CK_METRIC_TYPE` as listed in the table to run different model
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \ 
  --dep_add_tags.weights=[MODEL_NAME] \
  --env.CK_LOADGEN_REF_PROFILE=[LOADGEN_PROFILE] \
  --env.CK_METRIC_TYPE=[DATA_TYPE] \
  \
  --env.CK_LOADGEN_BACKEND=tensorflow \
  --dep_add_tags.lib-tensorflow=vpip \
  --env.CK_LOADGEN_SCENARIO=Offline \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --skip_print_timers \
  --env.CUDA_VISIBLE_DEVICES=-1"
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
|`yolo-v3-coco`|`tf_yolo`| `COCO` | `tensorflow` |

<!-- |`ssdlite-mobilenet-v2-kitti`| `default_tf_object_det_zoo`| `KITTI` |
|`rcnn-nas-lowproposals-kitti`|`default_tf_object_det_zoo`| `KITTI` | -->


## 2) With Different Mode:
Mode could be changed by the tags `--env.CK_LOADGEN_MODE`. (When specified is accuracy, when not specified is performance.) In accuracy and performance mode, we can specify the count for accuracy mode and the qps for performance mode with the tag `--env.CK_LOADGEN_EXTRA_PARAMS`

| Accuracy Mode | Performance Mode |
| --- | ---|
|`--env.CK_LOADGEN_MODE='--accuracy'` <br> `--env.CK_LOADGEN_EXTRA_PARAMS='--count 50'` | `--env.CK_LOADGEN_EXTRA_PARAMS='--qps 30'` <br> `--env.CK_OPTIMIZE_GRAPH='True'`|


Accuracy Mode Example:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_LOADGEN_REF_PROFILE=tf_yolo \
  --env.CK_METRIC_TYPE=COCO \
  --env.CK_LOADGEN_BACKEND=tensorflow \
  --dep_add_tags.lib-tensorflow=vpip\
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --skip_print_timers"
```

Performance Mode Example:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_LOADGEN_EXTRA_PARAMS='--qps 30' \
  --env.CK_OPTIMIZE_GRAPH='True' \
  \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_LOADGEN_REF_PROFILE=tf_yolo \
  --env.CK_METRIC_TYPE=COCO \
  --env.CK_LOADGEN_BACKEND=tensorflow \
  --dep_add_tags.lib-tensorflow=vpip\
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --skip_print_timers"
```

## 3) With Different Device:
| CPU | GPU |
| --- | ---|
|`--env.CUDA_VISIBLE_DEVICES=-1` | Don't need to specify |

CPU Example
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_LOADGEN_REF_PROFILE=tf_yolo \
  --env.CK_METRIC_TYPE=COCO \
  --env.CK_LOADGEN_BACKEND=tensorflow \
  --dep_add_tags.lib-tensorflow=vpip\
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

GPU Example
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_LOADGEN_REF_PROFILE=tf_yolo \
  --env.CK_METRIC_TYPE=COCO \
  --env.CK_LOADGEN_BACKEND=tensorflow \
  --dep_add_tags.lib-tensorflow=vpip\
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

## 3) With Different Scenario:

Change the tag `--env.CK_LOADGEN_SCENARIO` to specify it

|SCENARIO|
|---|
| `SingleStrem`, `MultiStream`, `Server`, `Offline` |

```
time docker run -it --rm ${CK_IMAGE} 
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_LOADGEN_SCENARIO=[SCENARIO] \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_LOADGEN_REF_PROFILE=tf_yolo \
  --env.CK_METRIC_TYPE=COCO \
  --env.CK_LOADGEN_BACKEND=tensorflow \
  --dep_add_tags.lib-tensorflow=vpip\
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --skip_print_timers"
```


## 4) With Different Backend
Avalibale backend: tensorflow, openvino-cpu
<!-- tflite, onnxruntime, pytorch, pytorch-native, tensorflowRT -->

Tensorflow example:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_LOADGEN_BACKEND=[BACKEND] \
  --dep_add_tags.lib-tensorflow=vpip\
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_LOADGEN_REF_PROFILE=tf_yolo \
  --env.CK_METRIC_TYPE=COCO \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --skip_print_timers"
```

Openvino-cpu example:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=rcnn-inception-v2-coco \
  --env.CK_LOADGEN_REF_PROFILE=default_tf_object_det_zoo \
  --env.CK_METRIC_TYPE=COCO \
  --env.CK_LOADGEN_BACKEND=openvino-cpu \
  --dep_add_tags.lib-tensorflow=vopenvino \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --skip_print_timers"
```

## 5) Whether to use `optimize_for_inference` lib to optimise the graph
Default option is `False`, can be explicitly configured with `--env.CK_OPTIMIZE_GRAPH`

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_OPTIMIZE_GRAPH='True' \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_LOADGEN_REF_PROFILE=tf_yolo \
  --env.CK_METRIC_TYPE=COCO \
  --env.CK_LOADGEN_BACKEND=tensorflow \
  --dep_add_tags.lib-tensorflow=vpip\
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --skip_print_timers"
```

<!-- ### Program parameters for ck-mlperf-tf-object-detection

#### `CK_BATCH_COUNT`

The number of batches to be processed.

Default: `1`

#### `CK_BATCH_SIZE`

The number of images in each batch

Default: `1`

#### `CK_ENV_TENSORFLOW_MODEL_FROZEN_GRAPH`

The path to the graph to run the inference

Default: set by CK

#### `CK_ENV_TENSORFLOW_MODEL_LABELMAP_FILE`

File with the model labelmap file

Default: set by CK

#### `CK_ENV_TENSORFLOW_MODEL_DATASET_TYPE`

Type of the dataset (coco,kitti,...) that is used for the inference

Default: set by CK

#### `CK_ENV_IMAGE_WIDTH` and `CK_ENV_IMAGE_HEIGHT`

The dimensions for the resize of the images, for the preprocessing

Default: set by CK, according to the selected model

#### `CK_ENV_DATASET_IMAGE_DIR`

Path to the directory with the images

Default: set by CK

#### `CK_ENV_DATASET_TYPE`

Type of dataset used for the program run

Default: set by CK

#### `CK_ENV_DATASET_ANNOTATIONS_PATH`

Path to the file with the annotations

Default: set by CK

#### `CK_PROFILE`

mlperf profile to select for the run

Default: default\_tf\_object\_det\_zoo

#### `CK_SCENARIO`

mlperf scenario of the run

Default: Offline

#### `CK_NUM_THREADS`

Number of threads used in mlperf

Default: `1`

#### `CK_TIME`

mlperf parameter time to scan in seconds

Default: `60`

#### `CK_QPS`

mlperf target qps estimate

Default: `100`

#### `CK_ACCURACY`

mlperf variable used to enable the accuracy pass

Default: 'YES'

#### `CK_CACHE`

mlperf variable used to enable the reuse of preprocessed numpy files. enable ONLY when processing the same model in more than 1 run

Default: `0`

#### `CK_QUERIES_SINGLE` `CK_QUERIES_MULTI` `CK_QUERIES_OFFLINE`

mlperf variables with the queries for the different scenarios

Defaults: `1024` `24576` `24576`

#### `CK_MAX_LATENCY`

mlperf variable with the max latency in the 99pct tile

Default: `0.1` -->
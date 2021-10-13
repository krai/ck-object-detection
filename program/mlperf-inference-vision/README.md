# MLPerf Inference Vision - extended for Object Detection

This Collective Knowledge workflow is based on the [official MLPerf Inference Vision application](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) extended for diverse Object Detection models, as found e.g. in the [TF1 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and the [TF2 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

The table below shows currently supported models, frameworks ("inference engines") and library/device combinations ("inference engine backends").

| `MODEL_NAME`                                | `INFERENCE_ENGINE`  | `INFERENCE_ENGINE_BACKEND` |
| ---                                         | ---                 | --- |
|`rcnn-nas-lowproposals-coco`                 | `tensorflow`        | `default-cpu`,`default-gpu`, `openvino-cpu` |
|`rcnn-resnet50-lowproposals-coco`            | `tensorflow`        | `default-cpu`,`default-gpu`, `openvino-cpu` |
|`rcnn-resnet101-lowproposals-coco`           | `tensorflow`        | `default-cpu`,`default-gpu`, `openvino-cpu` |
|`rcnn-inception-resnet-v2-lowproposals-coco` | `tensorflow`        | `default-cpu`,`default-gpu`, `openvino-cpu` |
|`rcnn-inception-v2-coco`                     | `tensorflow`        | `default-cpu`,`default-gpu`, `openvino-cpu` |
|`ssd-inception-v2-coco`                      | `tensorflow`        | `default-cpu`,`default-gpu` |
|`ssd_mobilenet_v1_coco`                      | `tensorflow`        | `default-cpu`,`default-gpu` |
|`ssd_mobilenet_v1_quantized_coco`            | `tensorflow`        | `default-cpu`,`default-gpu` |
|`ssd-mobilenet-v1-fpn-sbp-coco`              | `tensorflow`        | `default-cpu`,`default-gpu` |
|`ssd-resnet50-v1-fpn-sbp-coco`               | `tensorflow`        | `default-cpu`,`default-gpu` |
|`ssdlite-mobilenet-v2-coco`                  | `tensorflow`        | `default-cpu`,`default-gpu` |
|`yolo-v3-coco` |                             | `tensorflow`        | `default-cpu`,`default-gpu`, `openvino-cpu` |


# Build the environment:

## 1) Docker

Build the Docker image by running the build script `ck-mlperf/docker/mlperf-inference-vision-with-ck.tensorrt/build.sh`.

Also, set the image name:

```
export CK_IMAGE="krai/mlperf-inference-vision-with-ck.tensorrt:21.08-py3_tf-2.6.0"
```

### a) Just run the docker AND execute the intended CK command

Following the format below:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision ... "
```

Example:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

### b) To run a custom command and edit the environment

#### Create a container called `ck`

##### Without GPU support

```
docker run -td --entrypoint /bin/bash --name ck ${CK_IMAGE}
```

##### With GPU support
```
docker run -td --runtime=nvidia --entrypoint /bin/bash --name ck ${CK_IMAGE}
```

#### Start the container

```
docker exec -it ck /bin/bash
```

#### Stop the container
```
docker stop ck
```

#### Remove the container
```
docker rm ck
```

## 2) Locally

### Repositories

```bash
$ ck pull repo:ck-object-detection --url=https://github.com/krai/ck-object-detection
$ ck pull repo:ck-tensorflow --url=https://github.com/krai/ck-tensorflow
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
$ ck install package --tags=object-detection,model,tf,tensorflow,tf1-zoo
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

# Running
## 0) General Form:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct
  # Model_Specifications
  --dep_add_tags.weights=[MODEL_NAME] \
  --env.CK_MODEL_PROFILE=[MODEL_PROFILE] \
  --env.CK_METRIC_TYPE=[DATA_TYPE] \

  # Backend_Specifications
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  # Pass in relevant devices: CPU/GPU
  --env.CUDA_VISIBLE_DEVICES=-1 \

  # Scenario_Specifications
  --env.CK_LOADGEN_SCENARIO=Offline \

  # Mode_Specifications
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \

  # Others
  --skip_print_timers"
```


## 1) Specify a Model

Specify both `--dep_add_tags.weights=[MODEL_NAME]` and `--env.CK_MODEL_PROFILE=[MODEL_PROFILE]`.


### Supported `MODEL_NAME`/`MODEL_PROFILE` combinations

| `MODEL_NAME`                               | `MODEL_PROFILE`             |
| ------------------------------------------ | --------------------------- |
|`rcnn-nas-lowproposals-coco`                | `default_tf_object_det_zoo` |
|`rcnn-resnet50-lowproposals-coco`           | `default_tf_object_det_zoo` |
|`rcnn-resnet101-lowproposals-coco`          | `default_tf_object_det_zoo` |
|`rcnn-inception-resnet-v2-lowproposals-coco`| `default_tf_object_det_zoo` |
|`rcnn-inception-v2-coco`                    | `default_tf_object_det_zoo` |
|`ssd-inception-v2-coco`                     | `default_tf_object_det_zoo` |
|`ssd_mobilenet_v1_coco`                     | `default_tf_object_det_zoo` |
|`ssd_mobilenet_v1_quantized_coco`           | `default_tf_object_det_zoo` |
|`ssd-mobilenet-v1-fpn-sbp-coco`             | `default_tf_object_det_zoo` |
|`ssd-resnet50-v1-fpn-sbp-coco`              | `default_tf_object_det_zoo` |
|`ssdlite-mobilenet-v2-coco`                 | `default_tf_object_det_zoo` |
|`yolo-v3-coco`                              | `tf_yolo`                   |


## 2) Specify a Mode

The LoadGen mode can be selected by the environment variable `--env.CK_LOADGEN_MODE`. (When the mode is specified, it is `AccuracyOnly`; otherwise, it is `PerformanceOnly`.)

### Accuracy

For the Accuracy mode, you can specify the number of samples to process e.g. `--env.CK_LOADGEN_EXTRA_PARAMS='--count 50'`.

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

### Performance

For the performance mode, you should specify the expected QPS e.g. `--env.CK_LOADGEN_EXTRA_PARAMS='--qps=30'`. We also recommended to specify `--env.CK_OPTIMIZE_GRAPH='True'`.

Example:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_LOADGEN_EXTRA_PARAMS='--qps 30' \
  --env.CK_OPTIMIZE_GRAPH='True' \
  \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

## 3) Specify a Scenario

You can specify the scenario with the `--env.CK_LOADGEN_SCENARIO` environment variable.

|SCENARIO|
|---|
| `SingleStream`, `MultiStream`, `Server`, `Offline` |

```
time docker run -it --rm ${CK_IMAGE}
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_LOADGEN_SCENARIO=[SCENARIO] \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu\
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --skip_print_timers"
```

## 4) Whether to use `optimize_for_inference` lib to optimize the graph

Use the environment variable `--env.CK_OPTIMIZE_GRAPH` to configure whether to optimize the model graph for execution (default: `False`).

We recommended it to be set to `False` when running in the Accuracy mode, and to `True` when running in the Performance mode.

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_OPTIMIZE_GRAPH='True' \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --dep_add_tags.inference-engine-backend=default-cpu\
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```


## 5) Select an Inference Engine and an Inference Engine Backend

### Supported `INFERENCE_ENGINE`/`INFERENCE_ENGINE_BACKEND`/`CUDA_VISIBLE_DEVICES` combinations

| `INFERENCE_ENGINE` | `INFERENCE_ENGINE_BACKEND`  | `CUDA_VISIBLE_DEVICES`       |
| ------------------ | --------------------------- | ---------------------------- | 
| `tensorflow`       | `default-cpu`               | `-1`                         | 
| `tensorflow`       | `default-gpu`               | `<device_id>`                | 
| `tensorflow`       | `openvino-cpu`              | `-1`                         | 
| `tensorflow`       | `openvino-gpu` (not tested) | `-1` (integrated Intel GPU)  |
| `tensorflow`       | `openvino-gpu` (not tested) | `0` (discreet Intel GPU)     |

### Examples

#### `tensorflow/default-cpu/-1`
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

#### `tensorflow/default-gpu/0`
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-gpu \
  --env.CUDA_VISIBLE_DEVICES=0 \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

#### `tensorflow/openvino-cpu/-1`

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=openvino-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=rcnn-inception-v2-coco \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

#### `tensorflow/openvino-gpu/-1` (not tested)

If the machine has an Intel chip with an integrated GPU, set `--env.CUDA_VISIBLE_DEVICES=-1`:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=openvino-gpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=rcnn-inception-v2-coco \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

#### `tensorflow/openvino-gpu/0` (not tested)

If the machine has a discreet Intel GPU, set `--env.CUDA_VISIBLE_DEVICES=0`:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=openvino-gpu \
  --env.CUDA_VISIBLE_DEVICES=0 \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=rcnn-inception-v2-coco \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

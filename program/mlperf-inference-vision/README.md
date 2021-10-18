# MLPerf Inference Vision - extended for Object Detection

This Collective Knowledge workflow is based on the [official MLPerf Inference Vision application](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) extended for diverse Object Detection models, as found e.g. in the [TF1 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and the [TF2 Object Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

The table below shows currently supported models, frameworks ("inference engines") and library/device combinations ("inference engine backends").

| `MODEL_NAME`                                 | `INFERENCE_ENGINE`  | `INFERENCE_ENGINE_BACKEND`                 |
| -------------------------------------------- | ------------------- | ------------------------------------------ |
| `rcnn-nas-lowproposals-coco`                 | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-resnet50-lowproposals-coco`            | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-resnet101-lowproposals-coco`           | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-inception-resnet-v2-lowproposals-coco` | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `rcnn-inception-v2-coco`                     | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |
| `ssd-inception-v2-coco`                      | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_mobilenet_v1_coco`                      | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd_mobilenet_v1_quantized_coco`            | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd-mobilenet-v1-fpn-sbp-coco`              | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssd-resnet50-v1-fpn-sbp-coco`               | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `ssdlite-mobilenet-v2-coco`                  | `tensorflow`        | `default-cpu`,`default-gpu`                |
| `yolo-v3-coco`                               | `tensorflow`        | `default-cpu`,`default-gpu`,`openvino-cpu` |


# Build the environment with Docker

Build the Docker image:

```
$ export CK_IMAGE_NAME=mlperf-inference-vision-with-ck.tensorrt SDK_VER=21.09-py3 TF_VER=2.6.0
$ $(ck find docker:${CK_IMAGE_NAME})/build.sh
...
Successfully built 362d3cd6ddd5
Successfully tagged krai/mlperf-inference-vision-with-ck.tensorrt:21.09-py3_tf-2.6.0

real    0m0.099s
user    0m0.024s
sys     0m0.005s
```

Set an environment variable for the built image and validate:

```
$ export CK_IMAGE="krai/${CK_IMAGE_NAME}:${SDK_VER}_tf-${TF_VER}"
$ docker image ls ${CK_IMAGE}
REPOSITORY                                      TAG                  IMAGE ID       CREATED         SIZE
krai/mlperf-inference-vision-with-ck.tensorrt   21.09-py3_tf-2.6.0   362d3cd6ddd5   8 minutes ago   16.6GB
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
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --dep_add_tags.weights=yolo-v3-coco \
  \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
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

<!-- ## 2) Locally

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
``` -->

---

# Running
## 0) General Form:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
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
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50'"
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

For the Accuracy mode, you can specify the number of samples to process e.g. `--env.CK_LOADGEN_EXTRA_PARAMS='--count 96'`.

```
time docker run -it --rm ${CK_IMAGE}
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 96' \
  \
  --dep_add_tags.weights=ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream"
```

### Performance

For the performance mode, you should specify the buffer size and the expected QPS e.g. `--env.CK_LOADGEN_EXTRA_PARAMS='--count 256 --qps=30'`. We also recommended to specify `--env.CK_OPTIMIZE_GRAPH='True'`.

Example:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 256 --qps 30' \
  --env.CK_OPTIMIZE_GRAPH='True' \
  \
  --dep_add_tags.weights=ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream"
```

## 3) Specify a Scenario

You can specify the scenario with the `--env.CK_LOADGEN_SCENARIO` environment variable.

|SCENARIO|
|---|
| `SingleStream`, `MultiStream`, `Server`, `Offline` |

```
time docker run -it --rm ${CK_IMAGE}
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_LOADGEN_SCENARIO=[SCENARIO] \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 96' \
  --dep_add_tags.weights=ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```

### Batch Size in Offline Mode

The batch size is 1 by default. You can experiment with `CK_BATCH_SIZE` in the `Offline` scenario:

Using the batch size of 32 under the `Accuracy` mode and `Offline` scenario:
```
time docker run -it --rm ${CK_IMAGE}
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_BATCH_SIZE=32 \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 96' \
  --env.CK_LOADGEN_SCENARIO=Offline \
  \
  --dep_add_tags.weights=ssd_mobilenet_v1_coco \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1"
```

Using the batch size of 32 under the `Performance` mode and `Offline` scenario:
```
time docker run -it --rm ${CK_IMAGE}
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
--env.CK_BATCH_SIZE=32 \
--env.CK_LOADGEN_EXTRA_PARAMS='--count 256 --qps 4' \
--env.CK_LOADGEN_SCENARIO=Offline \
\
--dep_add_tags.weights=ssd_mobilenet_v1_coco \
--env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
--env.CK_INFERENCE_ENGINE=tensorflow \
--env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
--env.CUDA_VISIBLE_DEVICES=-1"
```


## 4) Graph Optimization

Use the environment variable `--env.CK_OPTIMIZE_GRAPH` to configure whether to optimize the model graph for execution (default: `False`).

We recommended it to be set to `False` when running in the Accuracy mode, and to `True` when running in the Performance mode.

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_OPTIMIZE_GRAPH='True' \
  \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --dep_add_tags.weights=yolo-v3-coco \
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --dep_add_tags.inference-engine-backend=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream"
```


## 5) Select an Engine/Backend/Device

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
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --dep_add_tags.weights=rcnn-inception-v2-coco"
```

#### `tensorflow/default-gpu/0`
```
time docker run --runtime=nvidia -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-gpu \
  --env.CUDA_VISIBLE_DEVICES=0 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --dep_add_tags.weights=rcnn-inception-v2-coco"
```

#### `tensorflow/openvino-cpu/-1`

```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=openvino-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --dep_add_tags.weights=rcnn-inception-v2-coco"
```

#### `tensorflow/openvino-gpu/-1` (not tested)

If the machine has an Intel chip with an integrated GPU, set `--env.CUDA_VISIBLE_DEVICES=-1`:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=openvino-gpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --dep_add_tags.weights=rcnn-inception-v2-coco"
```

#### `tensorflow/openvino-gpu/0` (not tested)

If the machine has a discreet Intel GPU, set `--env.CUDA_VISIBLE_DEVICES=0`:
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct --skip_print_timers \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=openvino-gpu \
  --env.CUDA_VISIBLE_DEVICES=0 \
  \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  \
  --env.CK_MODEL_PROFILE=default_tf_object_det_zoo \
  --dep_add_tags.weights=rcnn-inception-v2-coco"
```

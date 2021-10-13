# TensorFlow object-detection program

Our program supports the following object detection models. We also offer a few options for inference engine and inference engine backend. 

| MODEL_NAME | MODEL_PROFILE | INFERENCE_ENGINE:INFERENCE_ENGINE_BACKEND |
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
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

### b) To run custom command and edit the environment
use the following to create a dummy container `ck`
```
docker run -td --entrypoint /bin/bash --name ck ${CK_IMAGE}
```
or also enable GPU support
```
docker run -td --runtime=nvidia --entrypoint /bin/bash --name ck ${CK_IMAGE}
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


## 1) With Different Model
Change the `--dep_add_tags.weights`, `--env.CK_MODEL_PROFILE`, `--env.CK_INFERENCE_ENGINE`, `--env.CK_INFERENCE_ENGINE_BACKEND` as listed in the top table to run different model. The flag `--env.CUDA_VISIBLE_DEVICES` will also be affected, [DEVICE_NUMBER] is `-1` for `-cpu` and `0` for `-gpu`.
```
time docker run -it --rm ${CK_IMAGE} \
"ck run program:mlperf-inference-vision --cmd_key=direct \ 
  --dep_add_tags.weights=[MODEL_NAME] \
  --env.CK_MODEL_PROFILE=[MODEL_PROFILE] \
  --env.CK_INFERENCE_ENGINE=[INFERENCE_ENGINE] \
  --env.CK_INFERENCE_ENGINE_BACKEND=[INFERENCE_ENGINE_BACKEND] \
  --env.CUDA_VISIBLE_DEVICES=[DEVICE_NUMBER] \
  \
  --env.CK_LOADGEN_SCENARIO=Offline \
  --env.CK_LOADGEN_MODE='--accuracy' \
  --env.CK_LOADGEN_EXTRA_PARAMS='--count 50' \
  --skip_print_timers"
```

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
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
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
  --env.CK_MODEL_PROFILE=tf_yolo \
  --env.CK_INFERENCE_ENGINE=tensorflow \
  --env.CK_INFERENCE_ENGINE_BACKEND=default-cpu \
  --env.CUDA_VISIBLE_DEVICES=-1 \
  --env.CK_LOADGEN_SCENARIO=SingleStream \
  --skip_print_timers"
```

## 3) With Different Scenario:

Change the tag `--env.CK_LOADGEN_SCENARIO` to specify it

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

## 4) Whether to use `optimize_for_inference` lib to optimise the graph
Default option is `False`, can be explicitly configured with `--env.CK_OPTIMIZE_GRAPH`. We recommended it to be set as `False` when running the accuracy mode, and `True` when running the performance mode.

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


## 5) With Different Inference Engine and Inference Engine Backend
Avalibale Inference_Engine:Inference_Engine_Backend pairs : `tensorflow:default-cpu`, `tensorflow:default-gpu`, `tensorflow:openvino-cpu`. Please also pass in relevant `CUDA_VISIBLE_DEVICES`.

| `-cpu` | `-gpu` |
| --- | ---|
|`--env.CUDA_VISIBLE_DEVICES=-1` | `--env.CUDA_VISIBLE_DEVICES=n` |
where `n=0,1,2,...` depending on which GPU to be used. 

#### Example `tensorflow:default-cpu`:
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

#### Example `tensorflow:default-gpu`:
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

#### Example `tensorflow:openvino-cpu`:
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

#### Example `tensorflow:openvino-gpu` (Untested): 

`openvino` only works with Intel devices.

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
If the machine has an Intel GPU, set `--env.CUDA_VISIBLE_DEVICES=0`:
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

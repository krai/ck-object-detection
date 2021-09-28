#!/bin/bash

function install_package () {
    echo "----------------------Installing ${1}----------------"
    INSTALL_CMD="ck install package --tags=object-detection,model,tf,tensorflow,tf1-zoo,${1}"
    echo INSTALL_CMD
    echo "$(${INSTALL_CMD})"
}

function remove_package {
    echo "$(ck clean env --force --tags=object-detection,model,tf,tensorflow,tf1-zoo)"
    echo "Removed Any Object Detection tf1-zoo Package"
}

# Debugging
# declare -a VARIATIONS=(
#     "ssd-inception-v2-coco" 
#     "rcnn-inception-v2-coco" 
#     "ssdlite-mobilenet-v2-coco" 
#     "rcnn-resnet101-lowproposals-coco" 
#     "rcnn-inception-resnet-v2-lowproposals-coco" 
#     "ssd-mobilenet-v1-fpn-sbp-coco" 
#     "rcnn-resnet50-lowproposals-coco" 
#     "ssd-resnet50-v1-fpn-sbp-coco"  
#     "rcnn-nas-lowproposals-coco" 
#     "rcnn-nas-coco"  
#     "yolo-v3-coco" 
#     "ssdlite-mobilenet-v2-kitti", 
#     "rcnn-nas-lowproposals-kitti"
# )

LIST_CMD="$(ck list_variations misc --query_module_uoa=package --tags=object-detection,model,tf,tensorflow,tf1-zoo --separator=:)"
IFS=':' read -ra VARIATIONS <<< "${LIST_CMD}"

for MODEL in "${VARIATIONS[@]}"; do 
    remove_package
    install_package "${MODEL}"
    MODEL_PATH="${HOME}/CK-TOOLS/model-tf1-object-detection-zoo-${MODEL}"
    if [ ! -d ${MODEL_PATH} ]
    then
        echo "-----------------------------------Error ${MODEL_PATH} doesn't exists.---------------------------"
        remove_package
        exit 1
    fi
    echo "--------------------PASS: ${MODEL}-----------------------------"
done

remove_package
echo "Successfully passed the unit test"
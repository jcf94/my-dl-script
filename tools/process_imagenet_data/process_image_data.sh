#!/bin/bash

DATA="`pwd`/data"

chmod +x "${DATA}/build_image_data.py"
chmod +x "${DATA}/build_imagenet_data.py"
chmod +x "${DATA}/download_and_preprocess_imagenet.sh"
chmod +x "${DATA}/download_imagenet.sh"
chmod +x "${DATA}/preprocess_imagenet_validation_data.py"
chmod +x "${DATA}/process_bounding_boxes.py"

"${DATA}/download_and_preprocess_imagenet.sh"
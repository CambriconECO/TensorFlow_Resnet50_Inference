#!/bin/bash

/bin/bash clean.sh


echo "#######################"
echo "runing cpu..."
python tf_forward.py --input_pb resnet50_v1.pb --mode cpu 

echo "#######################"
cp resnet50_v1_quant_param.txt_param resnet50_v1_quant_param.txt
echo "runing quant..."
python tf_forward.py --input_pb resnet50_v1.pb --mode quant 

echo "#######################"
echo "runing mlu online layer..."
python tf_forward.py --input_pb resnet50_v1.pb --mode online_layer 

echo "#######################"
echo "runing mlu online fusion..."
python tf_forward.py --input_pb resnet50_v1.pb --mode online_fusion 

echo "#######################"
echo "generating mlu offline model..."
python tf_forward.py --input_pb resnet50_v1.pb --mode offline 

echo "#######################"
echo "running mlu offline model..."
../cnrt_resnet50_demo/cnrt_resnet50_demo resnet50_v1.cambricon subnet0 0 0 fox.jpg 1  

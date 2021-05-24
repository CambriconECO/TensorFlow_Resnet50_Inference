#coding=utf-8

import os
#os.environ['MLU_VISIBLE_DEVICES']='0' 
#os.environ['MLU_STATIC_NODE_FUSION']='true'

import sys
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python.framework import tensor_shape
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import graph_pb2

tf.app.flags.DEFINE_integer(
	'batch_size', 1, 'The number of samples in each batch.')
    
tf.app.flags.DEFINE_string(
	'image_size', "224,224", 'Eval image size')
    
tf.app.flags.DEFINE_string(
	'input_pb', None,'')

tf.app.flags.DEFINE_string(
	'mode', None,'cpu, quant, online_layer, online_fusion, offline')
 
FLAGS = tf.app.flags.FLAGS


if FLAGS.mode=='cpu':
    os.environ['MLU_VISIBLE_DEVICES']=''

if FLAGS.mode=='quant':
    os.environ['MLU_VISIBLE_DEVICES']=''
    os.environ['MLU_QUANT_PARAM']='resnet50_v1_quant_param.txt' 
    os.environ['MLU_RUNNING_MODE']='0' 

if FLAGS.mode=='online_layer':
    os.environ['MLU_VISIBLE_DEVICES']='0'
    os.environ['MLU_QUANT_PARAM']='resnet50_v1_quant_param.txt' 
    os.environ['MLU_RUNNING_MODE']='1' 
    os.environ['MLU_STATIC_NODE_FUSION']='false'

if FLAGS.mode=='online_fusion':
    os.environ['MLU_VISIBLE_DEVICES']='0'
    os.environ['MLU_STATIC_NODE_FUSION']='true'
    os.environ['MLU_QUANT_PARAM']='resnet50_v1_quant_param.txt' 
    os.environ['MLU_RUNNING_MODE']='1' 

if FLAGS.mode=='offline':
    os.environ['MLU_VISIBLE_DEVICES']='0'
    os.environ['MLU_STATIC_NODE_FUSION']='true'
    os.environ['MLU_QUANT_PARAM']='resnet50_v1_quant_param.txt' 
    os.environ['MLU_RUNNING_MODE']='1' 


def load_classes(path):
    """
    Load class labels 
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')[:-1]
    fp.close()
    return names

def main():
    image_size = [int(x) for x in FLAGS.image_size.split(",")]
    config = tf.ConfigProto(allow_soft_placement=True)

    config.mlu_options.core_num = 16
#    config.mlu_options.convert_graph = True
    config.mlu_options.core_version = 'MLU270'
    if FLAGS.mode=='offline':
        config.mlu_options.save_offline_model = True
        config.mlu_options.offline_model_name = "resnet50_v1.cambricon"

    inf_graph = tf.Graph()
    with inf_graph.as_default():
        with tf.gfile.GFile(FLAGS.input_pb, 'rb') as fb:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fb.read())
#            tf.import_graph_def(graph_def, name='')
            output_graph_def = graph_pb2.GraphDef()

            for node in graph_def.node:
                new_node = node_def_pb2.NodeDef()
                if node.name in ["input"]:
                    new_node.name=node.name
                    new_node.op="Placeholder"
                    new_node.attr['dtype'].CopyFrom(attr_value_pb2.AttrValue(type='DT_FLOAT'))
                    shape = tensor_shape.TensorShape((FLAGS.batch_size,image_size[0],image_size[0],3))
                    new_node.attr['shape'].CopyFrom(attr_value_pb2.AttrValue(shape=shape.as_proto()))
                else:
                    new_node.CopyFrom(node)
                output_graph_def.node.extend([new_node])

            graph_nodes = [n for n in output_graph_def.node]
            input_nodes = []
            for t in graph_nodes:
                if t.op == 'Placeholder':
                    input_nodes.append(t)
            output_node = graph_nodes[-1]

            tf.import_graph_def(output_graph_def, name='')

            
        with tf.Session(config=config) as session:
            input_node="input:0"
            output_node="resnet_v1_50/predictions/Reshape_1:0"
            
            input = [tf.get_default_graph().get_tensor_by_name(node) for node in input_node.split(",")]
            output = [tf.get_default_graph().get_tensor_by_name(node) for node in output_node.split(",")]
            image = cv2.resize(cv2.imread('fox.jpg'), (image_size[0], image_size[1]))
            mean = [124, 116, 127]
            if FLAGS.mode=='cpu': 
               image = image - mean
            if FLAGS.mode=='quant':
               image = image - mean
            image=image[np.newaxis, :]#.astype(np.uint8)
            images = np.repeat(image, FLAGS.batch_size, axis=0)
            out = session.run(output, feed_dict={input[0]:images})
            top_k_index = np.array(out[0]).argsort()
            top_k_index = top_k_index[0][-5:]
            top_k_index = [index+1 for index in top_k_index]
            print('---------------------TOP_5-------------------------------------------')
            print(top_k_index)
            top_k_data = [out[0][0][index-1] for index in top_k_index]
            print(top_k_data)
            names = load_classes('../labels.txt')
            top_k_results = [names[index-1] for index in top_k_index]
            print(top_k_results)
            print('---------------------------------------------------------------------')

if __name__ == '__main__':
    main()


import os
import sys
import Blocks

sys.path.append(os.path.abspath('../'))

import torch
import yaml
import re
from manual_models import get_manual_graph
from model_builder import CNNBenchModel

def CNNBenchModel2Ops(config_file = None, model_name = None, batch = 1):
    print('Loading config file and instantiating graphObject ...', file=sys.stderr)

    with open(config_file) as cfg:
        try:
            config = yaml.safe_load(cfg)
        except yaml.YAMLError as exc:
            print(exc)

    graphObject = get_manual_graph(config, model_name)

    print('Instantiating the CNNBench model ...', file=sys.stderr)
    model = CNNBenchModel(config, graphObject)
    model.eval()

    print('Get model operations ...')
    ops = model.get_operations()
    
    stamp = 1   # initialize input output name stamp
    for item in ops:
        item.insert(1, str(stamp))      # create input name stamp
        item.insert(2, str(stamp + 1))  # create output name stamp
        stamp += 1
    
    # insert an input op
    ops.insert(0, ['input', '0', '1', str(batch), str(config['input_channels']), str(config['image_size']), str(config['image_size'])])
        
    conv_shapes, head_shapes = model.get_tensor_shapes()
    
    # debug print ###################
    print('##### Debug Print #####')
    for i in ops:
        print(i)
    for i in conv_shapes:
        print('conv_shapes')
        print(i)
    for i in head_shapes:
        print('head_shapes')
        print(i)
    print('#######################')
    # ################################

    return ops, conv_shapes, head_shapes

def Op2Blocks(block_dict, op, conv_shapes, head_shapes, Tile, batch):

    block_dict[op[2]] = []
    
    # inputs
    if op[0] == 'input':

        #debug print
        print(op)

        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        shape = [int(op[3]), int(op[5]), int(op[6]), int(op[4])]
        # debug print
        print(op[0], shape)
            
        count = 0
        for i in range(int((shape[0] + Tib - 1) / Tib)):
            for j in range(int((shape[1] + Tix - 1) / Tix)):
                for k in range(int((shape[2] + Tiy - 1) / Tiy)):
                    for p in range(int((shape[3] + Tif - 1) / Tif)):
                        T_input = [i * Tib, min((i + 1) * Tib - 1, shape[0] - 1), j * Tix, min((j + 1) * Tix - 1, shape[1] - 1), k * Tiy, min((k + 1) * Tiy - 1, shape[2] - 1), p * Tif, min((p + 1) * Tif - 1, shape[3] - 1)]
                        block = Blocks.MemoryLoadBlock(op[2], count, T_input, 'input')
                        count += 1
                        block_dict[op[2]].append(block)       
    
    # compute
    elif op[0] == 'Conv2d':

        #debug print
        print(op)

        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]

        kernel_size_idx = op.index('kernel_size')
        strides_idx = op.index('stride')

        if 'groups' in op:
            groups = int(op[op.index('groups')+1])
            input_channel = int(int(op[7]) / groups)
            output_channel = int(int(op[8]) / groups)
            print(op, groups, input_channel, output_channel)

            for i in range(groups):
                input_name = f'{op[1]}_{i}'
                filter_name = f'{op[1]}_weight_{i}'
                output_name = op[2]

                input_shape = list(conv_shapes[op[4]][op[6]-1])
                input_shape = [batch, input_shape[2], input_shape[3], input_channel]
                output_shape = list(conv_shapes[op[4]][op[6]])
                output_shape = [batch, output_shape[2], output_shape[3], output_channel]

                filter_shape = [output_channel, int(op[kernel_size_idx+1]), int(op[kernel_size_idx+2]), input_channel]
                strides = [1, int(op[strides_idx+1]), int(op[strides_idx+2]), 1]

                # debug print
                print(op[0], input_shape, output_shape, groups, filter_shape, strides)

                count = 0
                for i in range(int((input_shape[0] + Tib - 1) / Tib)):
                    for j in range(int((input_shape[1] + Tix - 1) / Tix)):
                        for k in range(int((input_shape[2] + Tiy - 1) / Tiy)):
                            for p in range(int((input_shape[3] + Tif - 1) / Tif)):
                                for q in range(int((filter_shape[0] + Tib - 1) / Tib)):
                                    T_input = [i * Tib, min((i + 1) * Tib - 1, input_shape[0] - 1), j * Tix, min((j + 1) * Tix - 1, input_shape[1] - 1), k * Tiy, min((k + 1) * Tiy - 1, input_shape[2] - 1), p * Tif, min((p + 1) * Tif - 1, input_shape[3] - 1)]
                                    T_filter = [q * Tib, min((q + 1) * Tib - 1, filter_shape[0] - 1), 0, filter_shape[1] - 1, 0, filter_shape[2] - 1, 0, filter_shape[3] - 1]
                                    T_output = [int(T_input[0] / strides[0]), int(T_input[1] / strides[0]), int(T_input[2] / strides[1]), int(T_input[3] / strides[1]), int(T_input[4] / strides[2]), int(T_input[5] / strides[2]), int(T_filter[0] / strides[3]), int(T_filter[1] / strides[3])]
                                    block = Blocks.Conv2DBlock(output_name, count, [T_input, T_filter], [input_name, filter_name], T_output)
                                    count += 1
                                    block_dict[output_name].append(block)

        else:
            input_name = op[1]
            filter_name = f'{op[1]}_weight'
            output_name = op[2]     
    
            # ['Conv2d', '1', '2', 'op_m', 0, '_v', 1, '3', '6', 'kernel_size', '5', '5', 'stride', '1', '1', 'bias', 'False']
            # input is [1, 3, 224, 224]
            # output is [1, 6, 220, 220]
    
            if op[3] == 'op_m':
                input_shape = list(conv_shapes[op[4]][op[6]-1])
                input_shape = [batch, input_shape[2], input_shape[3], input_shape[1]]
                output_shape = list(conv_shapes[op[4]][op[6]])
                output_shape = [batch, output_shape[2], output_shape[3], output_shape[1]]
                
            elif op[3] == 'proj_m':
                input_shape = list(conv_shapes[op[4]][0])
                input_shape = [batch, input_shape[2], input_shape[3], input_shape[1]]
                output_shape = [input_shape[0], input_shape[1], input_shape[2], int(op[8])]

            filter_shape = [int(op[8]), int(op[kernel_size_idx+1]), int(op[kernel_size_idx+2]), int(op[7])]
            strides = [1, int(op[strides_idx+1]), int(op[strides_idx+2]), 1]
                
            # debug print
            print(op[0], input_shape, output_shape, filter_shape, strides)
    
            count = 0
            for i in range(int((input_shape[0] + Tib - 1) / Tib)):
                for j in range(int((input_shape[1] + Tix - 1) / Tix)):
                    for k in range(int((input_shape[2] + Tiy - 1) / Tiy)):
                        for p in range(int((input_shape[3] + Tif - 1) / Tif)):
                            for q in range(int((filter_shape[0] + Tib - 1) / Tib)):
                                T_input = [i * Tib, min((i + 1) * Tib - 1, input_shape[0] - 1), j * Tix, min((j + 1) * Tix - 1, input_shape[1] - 1), k * Tiy, min((k + 1) * Tiy - 1, input_shape[2] - 1), p * Tif, min((p + 1) * Tif - 1, input_shape[3] - 1)]
                                T_filter = [q * Tib, min((q + 1) * Tib - 1, filter_shape[0] - 1), 0, filter_shape[1] - 1, 0, filter_shape[2] - 1, 0, filter_shape[3] - 1]
                                T_output = [int(T_input[0] / strides[0]), int(T_input[1] / strides[0]), int(T_input[2] / strides[1]), int(T_input[3] / strides[1]), int(T_input[4] / strides[2]), int(T_input[5] / strides[2]), int(T_filter[0] / strides[3]), int(T_filter[1] / strides[3])]
                                block = Blocks.Conv2DBlock(output_name, count, [T_input, T_filter], [input_name, filter_name], T_output)
                                count += 1
                                block_dict[output_name].append(block)
                            
    elif op[0] == 'Linear':

        #debug print
        print(op)

        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        input_name = op[1]
        filter_name = f'{op[1]}_weight'
        output_name = op[2]  
        
        in_idx = op.index('in_features')+1
        out_idx = op.index('out_features')+1

        input_shape = [batch, int(op[in_idx])]
        filter_shape = [int(op[in_idx]), int(op[out_idx])]
        
        # debug print
        print(op[0], input_shape, filter_shape)

        block = Blocks.MatMulBlock(output_name, [input_shape[0], input_shape[1], filter_shape[1]], [input_name, filter_name])
        block_dict[output_name].append(block)

    elif op[0] == 'BatchNorm2d':

        #debug print
        print(op)

        in_shape = list(conv_shapes[op[4]][op[6]])
        count = batch * in_shape[2] * in_shape[3]
        channels = in_shape[1]
        inputs_name = [op[1], f'{op[1]}_gamma', f'{op[1]}_beta', f'{op[1]}_mean', f'{op[1]}_variance']
        outputs_name = [op[2], f'{op[2]}_1', f'{op[2]}_2', f'{op[2]}_3', f'{op[2]}_4']
        
        # debug print
        print(op[0], count)
        
        block = Blocks.BatchNormBlock(op[2], count, channels, inputs_name, outputs_name)
        block_dict[op[2]].append(block)
        
        '''
        block = Blocks.InstantBlock(f'{op[2]}_1')
        block_dict[f'{op[2]}_1'] = []
        block_dict[f'{op[2]}_1'].append(block)
        block = Blocks.InstantBlock(f'{op[2]}_2')
        block_dict[f'{op[2]}_2'] = []
        block_dict[f'{op[2]}_2'].append(block)
        block = Blocks.InstantBlock(f'{op[2]}_3')
        block_dict[f'{op[2]}_3'] = []
        block_dict[f'{op[2]}_3'].append(block)
        block = Blocks.InstantBlock(f'{op[2]}_4')
        block_dict[f'{op[2]}_4'] = []
        block_dict[f'{op[2]}_4'].append(block)
        '''

    elif op[0] == 'ReLU' or op[0] == 'SiLU':

        #debug print
        print(op)

        if op[3] == 'op_m':
            shape = list(conv_shapes[op[4]][op[6]])
            shape[0] = batch
        elif op[3] == 'h_op_m':
            shape = [batch, list(head_shapes[op[4]-len(conv_shapes)][op[6]])[1]]
            
        # debug print
        print(op[0], shape)
        
        count = 1
        if shape != []:
            for num in shape:
                count *= num
        block = Blocks.ReluBlock(op[2], count, op[1])
        block_dict[op[2]].append(block)

    elif op[0] == 'AvgPool2d':

        #debug print
        print(op)

        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        
        shape = list(conv_shapes[op[4]][op[6]-1])
        shape = [batch, shape[2], shape[3], shape[1]]
        out_shape = list(conv_shapes[op[4]][op[6]])
        out_shape = [batch, out_shape[2], out_shape[3], out_shape[1]]

        number = batch * out_shape[3]
        size = out_shape[1] * out_shape[2]

        # debug print
        print(op[0], shape, out_shape)
       
        block = Blocks.PoolingBlock(op[2], 'AvgPool', number, size)
        block_dict[op[2]].append(block)

    elif op[0] == 'MaxPool2d':

        #debug print
        print(op)

        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        
        shape = list(conv_shapes[op[4]][op[6]-1])
        shape = [batch, shape[2], shape[3], shape[1]]
        out_shape = list(conv_shapes[op[4]][op[6]])
        out_shape = [batch, out_shape[2], out_shape[3], out_shape[1]]

        number = batch * out_shape[3]
        size = out_shape[1] * out_shape[2]

        # debug print
        print(op[0], shape, out_shape)  
        
        block = Blocks.PoolingBlock(op[2], 'MaxPool', number, size)
        block_dict[op[2]].append(block)

    elif op[0] == 'GlobalAvgPool':

        #debug print
        print(op)

        in_shape = [int(op[3]), int(op[4]), int(op[5]), int(op[6])]
        in_shape = [batch, in_shape[2], in_shape[3], in_shape[1]]
        out_shape = [batch, in_shape[3]]

        number = batch * in_shape[3]
        size = in_shape[1] * in_shape[2]
        
        # debug print
        print(op[0], in_shape, out_shape)  

        block = Blocks.GlobalAvgPoolingBlock(op[2], number, size)
        block_dict[op[2]].append(block)

    elif op[0] == 'AdaptiveAvgPooling2d':
        
        # debug print
        print(op)

        in_shape = list(conv_shapes[op[4]][op[6]-1])
        in_shape = [batch, in_shape[2], in_shape[3], in_shape[1]]
        out_shape = [batch, in_shape[3]]
        
        number = batch * in_shape[3]
        size = in_shape[1] * in_shape[2]

        # debug print
        print(op[0], in_shape, out_shape)  

        block = Blocks.GlobalAvgPoolingBlock(op[2], number, size)
        block_dict[op[2]].append(block)

    elif op[0] == 'UpsamplingBilinear2d':

        #debug print
        print(op)

        in_shape = list(conv_shapes[op[4]][op[6]-1])
        in_shape = [batch, in_shape[2], in_shape[3], in_shape[1]]
        out_shape = list(conv_shapes[op[4]][op[6]])
        out_shape = [batch, out_shape[2], out_shape[3], out_shape[1]]

        number = batch * in_shape[3]
        size = out_shape[1] * out_shape[2]
        
        # debug print
        print(op[0], in_shape, out_shape)  
        
        block = Blocks.UpsamplingBlock(op[2], number, size)
        block_dict[op[2]].append(block)

    return block_dict



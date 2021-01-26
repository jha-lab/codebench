import os
import sys
import tensorflow as tf
import Blocks

def Pb2Ops(input_pb):
    graph_def = None
    graph = None

    print('Loading graph definition ...', file=sys.stderr)
    try:
        with tf.io.gfile.GFile(input_pb, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
    except BaseException as e:
        print('Error loading the graph definition: {}'.format(str(e)))
        exit()

    print('Importing graph ...', file=sys.stderr)
    try:
        assert graph_def is not None
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name='',
                op_dict=None,
                producer_op_list=None
            )
    except BaseException as e:
        print('Error importing the graph: {}'.format(str(e)))
        exit()


    assert graph is not None
    ops = graph.get_operations()

    stream = []
    tensors = []
    pending = []

    for op in ops:
        if len(op.inputs) == 0:     # those operations without input
            stream.append(op)       # appending those ops in stream
            for out in op.outputs:  
                tensors.append(out) # appending the outputs of those ops in tensors
        else:
            pending.append(op)      # appending those ops with inputs in pending

    removed_num = 0
    while len(pending) > 0:
        #print("The number of remaining ops is ", len(pending))
        for i in range(len(pending)):
            op = pending[i - removed_num]
            finish = True
            for ins in op.inputs:
                if ins not in tensors:
                    finish = False
                    break
            if finish:
                stream.append(op)
                for out in op.outputs:
                    tensors.append(out)
                del pending[i - removed_num]
                removed_num += 1

    return stream

def RemoveRedundantOps(stream):
    tensor_dict = {}
    variable_names = []
    removed_num = 0

    for i in range(len(stream)):
        op = stream[i - removed_num]
        # remove Identity
        if op.type == 'Identity':
            tensor_dict[op.outputs[0]] = op.inputs[0]
            del stream[i - removed_num]
            removed_num += 1
        # remove VariableV2
        elif op.type == 'VariableV2':
            variable_names.append(op.outputs[0].name)
            del stream[i - removed_num]
            removed_num += 1
        # remove Assign
        elif op.type == 'Assign':
            tensor_dict[op.outputs[0]] = op.inputs[1]
            del stream[i - removed_num]
            removed_num += 1
        # remove Cast
        elif op.type == 'Cast':
            tensor_dict[op.outputs[0]] = op.inputs[0]
            del stream[i - removed_num]
            removed_num += 1

    return stream, tensor_dict, variable_names

def Op2Blocks(block_dict, op, Tile, frame):
    for out in op.outputs:
        block_dict[out.name] = []
    # inputs
    if op.type == 'Placeholder':
        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        shape = op.outputs[0].shape.as_list()
        if shape[0] == None:
            shape[0] = frame
        assert(len(shape) == 4)
        count = 0
        for i in range(int((shape[0] + Tib - 1) / Tib)):
            for j in range(int((shape[1] + Tix - 1) / Tix)):
                for k in range(int((shape[2] + Tiy - 1) / Tiy)):
                    for p in range(int((shape[3] + Tif - 1) / Tif)):
                        T_input = [i * Tib, min((i + 1) * Tib - 1, shape[0] - 1), j * Tix, min((j + 1) * Tix - 1, shape[1] - 1), k * Tiy, min((k + 1) * Tiy - 1, shape[2] - 1), p * Tif, min((p + 1) * Tif - 1, shape[3] - 1)]
                        block = Blocks.MemoryLoadBlock(op.outputs[0].name, count, T_input, 'input')
                        count += 1
                        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Const':
        block = Blocks.InstantBlock(op.outputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Fill':
        block = Blocks.InstantBlock(op.outputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Shape':
        block = Blocks.InstantBlock(op.outputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'RandomUniform':
        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        shape = op.outputs[0].shape.as_list()
        for i in range(len(shape)):
            if shape[i] == None:
                shape[i] = frame
        while len(shape) < 4:
            shape = [1] + shape
        count = 0
        for i in range(int((shape[0] + Tib - 1) / Tib)):
            for j in range(int((shape[1] + Tix - 1) / Tix)):
                for k in range(int((shape[2] + Tiy - 1) / Tiy)):
                    for p in range(int((shape[3] + Tif - 1) / Tif)):
                        T_input = [i * Tib, min((i + 1) * Tib - 1, shape[0] - 1), j * Tix, min((j + 1) * Tix - 1, shape[1] - 1), k * Tiy, min((k + 1) * Tiy - 1, shape[2] - 1), p * Tif, min((p + 1) * Tif - 1, shape[3] - 1)]
                        block = Blocks.MemoryLoadBlock(op.outputs[0].name, count, T_input, 'weight')
                        count += 1
                        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'TruncatedNormal':
        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        shape = op.outputs[0].shape.as_list()
        for i in range(len(shape)):
            if shape[i] == None:
                shape[i] = frame
        while len(shape) < 4:
            shape = [1] + shape
        count = 0
        for i in range(int((shape[0] + Tib - 1) / Tib)):
            for j in range(int((shape[1] + Tix - 1) / Tix)):
                for k in range(int((shape[2] + Tiy - 1) / Tiy)):
                    for p in range(int((shape[3] + Tif - 1) / Tif)):
                        T_input = [i * Tib, min((i + 1) * Tib - 1, shape[0] - 1), j * Tix, min((j + 1) * Tix - 1, shape[1] - 1), k * Tiy, min((k + 1) * Tiy - 1, shape[2] - 1), p * Tif, min((p + 1) * Tif - 1, shape[3] - 1)]
                        block = Blocks.MemoryLoadBlock(op.outputs[0].name, count, T_input, 'weight')
                        count += 1
                        block_dict[op.outputs[0].name].append(block)
    # compute
    elif op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative':
        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        input_name = op.inputs[0].name
        filter_name = op.inputs[1].name
        output_name = op.outputs[0].name
        data_format = op.node_def.attr['data_format'].s.decode()
        input_shape = op.inputs[0].shape.as_list()
        if input_shape[0] == None:
            input_shape[0] = frame
        filter_shape = op.inputs[1].shape.as_list()
        output_shape = op.outputs[0].shape.as_list()
        if output_shape[0] == None:
            output_shape[0] = frame
        strides = op.node_def.attr['strides'].list.i
        
        if data_format == 'NHWC':
            input_shape = [input_shape[0], input_shape[2], input_shape[1], input_shape[3]]
            output_shape = [output_shape[0], output_shape[2], output_shape[1], output_shape[3]]
            strides = [strides[0], strides[2], strides[1], strides[3]]
        elif data_format == 'NCHW':
            input_shape = [input_shape[0], input_shape[3], input_shape[2], input_shape[1]]
            output_shape = [output_shape[0], output_shape[3], output_shape[2], output_shape[1]]
            strides = [strides[0], strides[3], strides[2], strides[1]]
        filter_shape = [filter_shape[3], filter_shape[1], filter_shape[0], filter_shape[2]]

        count = 0
        for i in range(int((input_shape[0] + Tib - 1) / Tib)):
            for j in range(int((input_shape[1] + Tix - 1) / Tix)):
                for k in range(int((input_shape[2] + Tiy - 1) / Tiy)):
                    for p in range(int((input_shape[3] + Tif - 1) / Tif)):
                        for q in range(int((filter_shape[0] + Tib - 1) / Tib)):
                            T_input = [i * Tib, min((i + 1) * Tib - 1, input_shape[0] - 1), j * Tix, min((j + 1) * Tix - 1, input_shape[1] - 1), k * Tiy, min((k + 1) * Tiy - 1, input_shape[2] - 1), p * Tif, min((p + 1) * Tif - 1, input_shape[3] - 1)]
                            T_filter = [q * Tib, min((q + 1) * Tib - 1, filter_shape[0] - 1), 0, filter_shape[1] - 1, 0, filter_shape[2] - 1, 0, filter_shape[3] - 1]
                            T_output = [T_input[0] / strides[0], T_input[1] / strides[0], T_input[2] / strides[1], T_input[3] / strides[1], T_input[4] / strides[2], T_input[5] / strides[2], T_filter[0] / strides[3], T_filter[1] / strides[3]]
                            block = Blocks.Conv2DBlock(output_name, count, [T_input, T_filter], [input_name, filter_name], T_output)
                            count += 1
                            block_dict[output_name].append(block)
    elif op.type == 'MatMul':
        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        input_name = op.inputs[0].name
        filter_name = op.inputs[1].name
        output_name = op.outputs[0].name
        input_shape = op.inputs[0].shape.as_list()
        filter_shape = op.inputs[1].shape.as_list()
        if op.node_def.attr['transpose_a'].b:
            input_shape = [input_shape[1], input_shape[0]]
        if op.node_def.attr['transpose_b'].b:
            filter_shape = [filter_shape[1], filter_shape[0]]
        if input_shape[0] == None:
            input_shape[0] = frame
        if input_shape[1] == None:
            input_shape[1] = filter_shape[0]
        elif filter_shape[0] == None:
            filter_shape[0] = input_shape[1]
        assert input_shape[1] == filter_shape[0]

        block = Blocks.MatMulBlock(output_name, [input_shape[0], input_shape[1], filter_shape[1]], [input_name, filter_name])
        block_dict[output_name].append(block)

    elif op.type == 'Mul':
        a_shape = op.inputs[0].shape.as_list()
        b_shape = op.inputs[1].shape.as_list()
        a = 1
        b = 1
        if a_shape != []:
            for i in range(len(a_shape)):
                if a_shape[i] == None:
                    a_shape[i] = frame
            for num in a_shape:
                a *= num
        if b_shape != []:
            for i in range(len(b_shape)):
                if b_shape[i] == None:
                    b_shape[i] = frame
            for num in b_shape:
                b *= num
        block = Blocks.ScalarBlock(op.outputs[0].name, 'Mul', max(a, b), [op.inputs[0].name, op.inputs[1].name])
        block_dict[op.outputs[0].name].append(block)

    elif op.type == 'RealDiv':
        a_shape = op.inputs[0].shape.as_list()
        b_shape = op.inputs[1].shape.as_list()
        a = 1
        b = 1
        if a_shape != []:
            for i in range(len(a_shape)):
                if a_shape[i] == None:
                    a_shape[i] = frame
            for num in a_shape:
                a *= num
        if b_shape != []:
            for i in range(len(b_shape)):
                if b_shape[i] == None:
                    b_shape[i] = frame
            for num in b_shape:
                b *= num
        block = Blocks.ScalarBlock(op.outputs[0].name, 'Div', max(a, b), [op.inputs[0].name, op.inputs[1].name])
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Add' or op.type == 'BiasAdd':
        a_shape = op.inputs[0].shape.as_list()
        b_shape = op.inputs[1].shape.as_list()
        a = 1
        b = 1
        if a_shape != []:
            for i in range(len(a_shape)):
                if a_shape[i] == None:
                    a_shape[i] = frame
            for num in a_shape:
                a *= num
        if b_shape != []:
            for i in range(len(b_shape)):
                if b_shape[i] == None:
                    b_shape[i] = frame
            for num in b_shape:
                b *= num
        block = Blocks.ScalarBlock(op.outputs[0].name, 'Add', max(a, b), [op.inputs[0].name, op.inputs[1].name])
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Sub' or op.type == 'AssignSub':
        a_shape = op.inputs[0].shape.as_list()
        b_shape = op.inputs[1].shape.as_list()
        a = 1
        b = 1
        if a_shape != []:
            for i in range(len(a_shape)):
                if a_shape[i] == None:
                    a_shape[i] = frame
            for num in a_shape:
                a *= num
        if b_shape != []:
            for i in range(len(b_shape)):
                if b_shape[i] == None:
                    b_shape[i] = frame
            for num in b_shape:
                b *= num
        block = Blocks.ScalarBlock(op.outputs[0].name, 'Sub', max(a, b), [op.inputs[0].name, op.inputs[1].name])
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Floor':
        shape = op.inputs[0].shape.as_list()
        count = 1
        if shape != []:
            for i in range(len(shape)):
                if shape[i] == None:
                    shape[i] = frame
            for num in shape:
                count *= num
        block = Blocks.ScalarBlock(op.outputs[0].name, 'Floor', count, [op.inputs[0].name])
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Minimum':
        a_shape = op.inputs[0].shape.as_list()
        b_shape = op.inputs[1].shape.as_list()
        a = 1
        b = 1
        if a_shape != []:
            for i in range(len(a_shape)):
                if a_shape[i] == None:
                    a_shape[i] = frame
            for num in a_shape:
                a *= num
        if b_shape != []:
            for i in range(len(b_shape)):
                if b_shape[i] == None:
                    b_shape[i] = frame
            for num in b_shape:
                b *= num
        block = Blocks.ScalarBlock(op.outputs[0].name, 'Minimum', max(a, b), [op.inputs[0].name, op.inputs[1].name])
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Mean':
        block = Blocks.InstantBlock(op.outputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'FusedBatchNorm':
        count = op.inputs[0].shape.as_list()[3]
        inputs_name = [op.inputs[0].name, op.inputs[1].name, op.inputs[2].name, op.inputs[3].name, op.inputs[4].name]
        outputs_name = [op.outputs[0].name, op.outputs[1].name, op.outputs[2].name, op.outputs[3].name, op.outputs[4].name]
        block = Blocks.BatchNormBlock(op.outputs[0].name, count, inputs_name, outputs_name)
        block_dict[op.outputs[0].name].append(block)
        
        block = Blocks.InstantBlock(op.outputs[1].name)
        block_dict[op.outputs[1].name].append(block)
        block = Blocks.InstantBlock(op.outputs[2].name)
        block_dict[op.outputs[2].name].append(block)
        block = Blocks.InstantBlock(op.outputs[3].name)
        block_dict[op.outputs[3].name].append(block)
        block = Blocks.InstantBlock(op.outputs[4].name)
        block_dict[op.outputs[4].name].append(block)
    elif op.type == 'Relu' or op.type == 'Relu6':
        shape = op.inputs[0].shape.as_list()
        count = 1
        if shape != []:
            for i in range(len(shape)):
                if shape[i] == None:
                    shape[i] = frame
            for num in shape:
                count *= num
        block = Blocks.ReluBlock(op.outputs[0].name, count, op.inputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'AvgPool':
        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        data_format = op.node_def.attr['data_format'].s.decode()
        shape = op.inputs[0].shape.as_list()
        for i in range(len(shape)):
            if shape[i] == None:
                shape[i] = frame
        strides = op.node_def.attr['strides'].list.i

        if data_format == 'NHWC':
            shape = [shape[0], shape[2], shape[1], shape[3]]
            strides = [strides[0], strides[2], strides[1], strides[3]]
        elif data_format == 'NCHW':
            shape = [shape[0], shape[3], shape[2], shape[1]]
            strides = [strides[0], strides[3], strides[2], strides[1]]

        count = 0
        for i in range(int((shape[0] + Tib - 1) / Tib)):
            for j in range(int((shape[1] + Tix - 1) / Tix)):
                for k in range(int((shape[2] + Tiy - 1) / Tiy)):
                    for p in range(int((shape[3] + Tif - 1) / Tif)):
                        T_input = [i * Tib, min((i + 1) * Tib - 1, shape[0] - 1), j * Tix, min((j + 1) * Tix - 1, shape[1] - 1), k * Tiy, min((k + 1) * Tiy - 1, shape[2] - 1), p * Tif, min((p + 1) * Tif - 1, shape[3] - 1)]
                        T_output = [T_input[0] / strides[0], T_input[1] / strides[0], T_input[2] / strides[1], T_input[3] / strides[1], T_input[4] / strides[2], T_input[5] / strides[2], T_input[6] / strides[3], T_input[7] / strides[3]]
                        block = Blocks.PoolingBlock(op.outputs[0].name, 'AvgPool', count, T_input, op.inputs[0].name, T_output)
                        count += 1
                        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'MaxPool':
        Tib = Tile[0]
        Tix = Tile[1]
        Tiy = Tile[2]
        Tif = Tile[3]
        data_format = op.node_def.attr['data_format'].s.decode()
        shape = op.inputs[0].shape.as_list()
        for i in range(len(shape)):
            if shape[i] == None:
                shape[i] = frame
        strides = op.node_def.attr['strides'].list.i

        if data_format == 'NHWC':
            shape = [shape[0], shape[2], shape[1], shape[3]]
            strides = [strides[0], strides[2], strides[1], strides[3]]
        elif data_format == 'NCHW':
            shape = [shape[0], shape[3], shape[2], shape[1]]
            strides = [strides[0], strides[3], strides[2], strides[1]]

        count = 0
        for i in range(int((shape[0] + Tib - 1) / Tib)):
            for j in range(int((shape[1] + Tix - 1) / Tix)):
                for k in range(int((shape[2] + Tiy - 1) / Tiy)):
                    for p in range(int((shape[3] + Tif - 1) / Tif)):
                        T_input = [i * Tib, min((i + 1) * Tib - 1, shape[0] - 1), j * Tix, min((j + 1) * Tix - 1, shape[1] - 1), k * Tiy, min((k + 1) * Tiy - 1, shape[2] - 1), p * Tif, min((p + 1) * Tif - 1, shape[3] - 1)]
                        T_output = [T_input[0] / strides[0], T_input[1] / strides[0], T_input[2] / strides[1], T_input[3] / strides[1], T_input[4] / strides[2], T_input[5] / strides[2], T_input[6] / strides[3], T_input[7] / strides[3]]
                        block = Blocks.PoolingBlock(op.outputs[0].name, 'MaxPool', count, T_input, op.inputs[0].name, T_output)
                        count += 1
                        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Softmax':
        block = Blocks.LossBlock(op.outputs[0].name, 'Softmax', frame, op.inputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    # reshape
    elif op.type == 'Reshape':
        block = Blocks.TransposeBlock(op.outputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Squeeze':
        block = Blocks.InstantBlock(op.outputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Pack':
        block = Blocks.InstantBlock(op.outputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'StridedSlice':
        block = Blocks.InstantBlock(op.outputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'ConcatV2':
        block = Blocks.InstantBlock(op.outputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    elif op.type == 'Pad':
        block = Blocks.InstantBlock(op.outputs[0].name)
        block_dict[op.outputs[0].name].append(block)
    # skip
    #elif op.type == 'Identity':
    #    pass
    #elif op.type == 'VariableV2':
    #    pass
    #elif op.type == 'Assign':
    #    pass
    #elif op.type == 'Cast':
    #    pass

    return block_dict



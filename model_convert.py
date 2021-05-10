import os
import time
import subprocess


class Model(object):
    """
    [Input]
    --model             Model file
    --weight            Weight file. Required when framework is Caffe
    --om                The model file to be converted to json
    --framework         Framework type. 0:Caffe; 1:MindSpore; 3:Tensorflow; 5:Onnx
    --input_format      Format of input data. E.g.: "NCHW"
    --input_shape       Shape of input data. Separate multiple nodes with semicolons (;). Use double quotation marks (") to enclose each argument.
                        E.g.: "input_name1:n1,c1,h1,w1;input_name2:n2,c2,h2,w2"
    --dynamic_batch_size Set dynamic batch size. E.g.: "batchsize1,batchsize2,batchsize3"
    --dynamic_image_size Set dynamic image size. Separate multiple nodes with semicolons (;). Use double quotation marks (") to enclose each argument.
                        E.g.: "imagesize1_height,imagesize1_width;imagesize2_height,imagesize2_width"
    --dynamic_dims      Set dynamic dims. Separate multiple nodes with semicolons (;). Use double quotation marks (") to enclose each argument.
                        E.g.: "dims1_n1,dims1_n2;dims2_n1,dims2_n2"
    --singleop          Single op definition file. atc will generate offline model(s) for single op if --singleop is set.

    [Output]
    --output            Output file path&name(needn't suffix, will add .om automatically). 
                        If --singleop is set, this arg specifies the directory to which the single op offline model will be generated
    --output_type       Set net output type. Support FP32, FP16, UINT8. E.g.: FP16, indicates that all out nodes are set to FP16.
                        "node1:0:FP16;node2:1:FP32", indicates setting the datatype of multiple out nodes.
    --check_report      The pre-checking report file. Default value is: "check_result.json"
    --json              The output json file path&name which is converted from a model

    [Target]
    --soc_version       The soc version.
    --core_type         Set core type AiCore or VectorCore. VectorCore: use vector core. Default value is: AiCore
    --aicore_num        Set aicore num
    
    ===== Advanced Functionality =====
    [Feature]
    --out_nodes         Output nodes designated by users. Separate multiple nodes with semicolons (;).Use double quotation marks (") to enclose each argument.
                        E.g.: "node_name1:0;node_name1:1;node_name2:0"
    --input_fp16_nodes  Input node datatype is fp16. Separate multiple nodes with semicolons (;). Use double quotation marks (") to enclose each argument. E.g.: "node_name1;node_name2"
    --insert_op_conf    Config file to insert new op
    --op_name_map       Custom op name mapping file
                        Note: A semicolon(;) cannot be included in each path, otherwise the resolved path will not match the expected one.
    --is_input_adjust_hw_layout    Intput node datatype is fp16 and format is NC1HWC0, used with input_fp16_nodes. E.g.: "true,true,false,true"
    --is_output_adjust_hw_layout   Net output node datatype is fp16 and format is NC1HWC0, used with out_nodes. E.g.: "true,true,false,true"

    [Model Tuning]
    --disable_reuse_memory    The switch of reuse memory. Default value is : 0. 0 means reuse memory, 1 means do not reuse memory.
    --fusion_switch_file      Set fusion switch file path
    --enable_scope_fusion_passes    validate the non-general scope fusion passes, multiple names can be set and separated by ','. E.g.: ScopePass1,ScopePass2,...
    --enable_single_stream    Enable single stream. true: enable; false(default): disable
    --enable_small_channel    Set enable small channel. 0(default): disable; 1: enable
    --enable_compress_weight  Enable compress weight. true: enable; false(default): disable
    --compress_weight_conf    Config file to compress weight
    --buffer_optimize         Set buffer optimize. "l2_optimize" (default). Set "off_optimize" to close

    [Operator Tuning]
    --precision_mode        precision mode, support force_fp16(default), allow_mix_precision, allow_fp32_to_fp16, must_keep_origin_dtype.
    --keep_dtype            Retains the precision of certain operators in inference scenarios by using a configuration file.
    --auto_tune_mode        Set tune mode. E.g.: "GA,RL", support configure multiple, spit by ,
    --op_select_implmode    Set op select implmode. Support high_precision, high_performance. default: high_performance
    --optypelist_for_implmode    Appoint which op to select implmode, cooperated with op_select_implmode.
                                Separate multiple nodes with commas (,). Use double quotation marks (") to enclose each argument. E.g.: "node_name1,node_name2"
    --op_debug_level        Debug enable for TBE operator building.
                            0 (default): Disable debug; 1: Enable TBE pipe_all, and generate the operator CCE file and Python-CCE mapping file (.json);
                            2: Enable TBE pipe_all, generate the operator CCE file and Python-CCE mapping file (.json), and enable the CCE compiler -O0-g.
                            3: Disable debug, and keep generating kernel file (.o and .json)
    """

    def __init__(self):
        self.framework = "mindspore"
        self.soc_version = "Ascend310"
        self.src_format = "air"
        self.dst_format = "om"
        self.framework2id = {"mindspore": 1, "caffe": 0, "tensorflow": 3,  "onnx": 5}

    def convert(self, input_dir, output_dir, kvs, framework="mindspore", src_format="air", dst_format="om"):
        """
        Args:
            src_format: source serilization format
            dst_format: target serilization format
            input_dir:  directory of source files 
            output_dir: directory of output files
            kvs:        extended params, dictionary

        Return:
            (code, msg)
            code=0, succ
            code=other, fail

        atc command
        atc --input_format=NCHW \
            --framework=1 \
            --model=$model \
            --input_shape="x:1, 3, 768, 1280; im_info: 1, 4" \
            --output=fasterrcnn \
            --insert_op_conf=../src/aipp.cfg \
            --precision_mode=allow_fp32_to_fp16 \
            --soc_version=Ascend310
        
        command =  ["atc", "--input_format=NCHW", "--framework=1", "--model=lenet.air",  "--output=lenet", "--soc_version=Ascend310"]
        subprocess.call(command)
        """
        code = 0
        command = ["atc", "--input_format=NCHW", "--soc_version=Ascend310"]

        assert framework in self.framework2id, 'not support framework'
        frameworkidx = self.framework2id[framework]
        param = "--" + str("framework") + "=" + str(frameworkidx)
        command.append(param)

        for key, value in kvs.items():
            param = "--" + str(key) + "=" + str(value)
            command.append(param)
        code = subprocess.call(command)
        if code == 0:
            msg = "ATC run success, welcome to the next use."
        else:
            msg = "ATC run failed, Please check the detail log, Try 'atc --help' for more information"

        return (code, msg)


if __name__ == '__main__':
    model = Model()
    kvs = {
        "model": "lenet.air",
        "output": "lenet"
    }
    (code, msg) = model.convert(input_dir=None, output_dir=None, kvs=kvs)
    print(code, msg)
# 使用ONNX部署深度学习和传统机器学习模型

## 目录
* [ONNX简介](#ONNX简介)
* [ONNX标准介绍](#ONNX标准介绍)
* [ONNX与PMML](#ONNX与PMML)
* [DaaS简介](#DaaS简介)
* [使用ONNX部署传统机器学习模型](#使用ONNX部署传统机器学习模型)
* [使用ONNX部署深度神经网络模型](#使用ONNX部署深度神经网络模型)
* [总结](#总结)
* [参考](#参考)

## ONNX简介
开放神经网络交换ONNX（Open Neural Network Exchange）是一套表示深度神经网络模型的开放格式，由微软和Facebook于2017推出，然后迅速得到了各大厂商和框架的支持。通过短短几年的发展，已经成为表示深度学习模型的实际标准，并且通过`ONNX-ML`，可以支持传统非神经网络机器学习模型，大有一统整个AI模型交换标准。

ONNX定义了一组与环境和平台无关的标准格式，为AI模型的互操作行提供了基础，使AI模型可以在不同框架和环境下交互使用。硬件和软件厂商可以基于ONNX标准优化模型性能，让所有兼容ONNX标准的框架受益。目前，ONNX主要关注在模型预测方面（inferring），使用不同框架训练的模型，转化为ONNX格式后，可以很容易的部署在兼容ONNX的运行环境中。

## ONNX标准介绍
ONNX规范由以下几个部分组成：
* 一个可扩展的计算图模型：提供了通用的计算图中间表示法（Intermediate Representation）。
* 内置操作符集：`ai.onnx`和`ai.onnx.ml`，`ai.onnx`是默认的操作符集，主要针对神经网络模型，`ai.onnx.ml`主要适用于传统非神经网络机器学习模型。
* 标准数据类型。包括张量（tensors）、序列（sequences）和映射（maps）。

目前，ONNX规范有两个官方变体，主要区别在与支持的类型和默认的操作符集。ONNX神经网络变体只使用张量作为输入和输出；而作为支持传统机器学习模型的`ONNX-ML`，还可以识别序列和映射，`ONNX-ML`为支持非神经网络算法扩展了ONNX操作符集。

ONNX使用protobuf序列化AI模型，顶层是一个模型（Model）结构，主要由关联的元数据和一个图（Graph）组成；图由元数据、模型参数、输入输出、和计算节点（`Node`）序列组成，这些节点构成了一个计算无环图，每一个计算节点代表了一次操作符的调用，主要由节点名称、操作符、输入列表、输出列表和属性列表组成，属性列表主要记录了一些运行时常量，比如模型训练时生成的系数值。

为了更直观的了解ONNX格式内容，下面，我们训练一个简单的LogisticRegression模型，然后导出ONNX。仍然使用常用的分类数据集`iris`：

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

clr = LogisticRegression()
clr.fit(X_train, y_train)
```

使用`skl2onnx`把Scikit-learn模型序列化为ONNX格式：

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([1, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("logreg_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

使用ONNX Python API查看和验证模型：

```python
import onnx

model = onnx.load('logreg_iris.onnx')
print(model)
```

输出模型信息如下：

```
ir_version: 5
producer_name: "skl2onnx"
producer_version: "1.5.1"
domain: "ai.onnx"
model_version: 0
doc_string: ""
graph {
  node {
    input: "float_input"
    output: "label"
    output: "probability_tensor"
    name: "LinearClassifier"
    op_type: "LinearClassifier"
    attribute {
      name: "classlabels_ints"
      ints: 0
      ints: 1
      ints: 2
      type: INTS
    }
    attribute {
      name: "coefficients"
      floats: 0.375753253698349
      floats: 1.3907358646392822
      floats: -2.127762794494629
      floats: -0.9207873344421387
      floats: 0.47902926802635193
      floats: -1.5524250268936157
      floats: 0.46959221363067627
      floats: -1.2708674669265747
      floats: -1.5656673908233643
      floats: -1.256540060043335
      floats: 2.18996000289917
      floats: 2.2694246768951416
      type: FLOATS
    }
    attribute {
      name: "intercepts"
      floats: 0.24828049540519714
      floats: 0.8415762782096863
      floats: -1.0461325645446777
      type: FLOATS
    }
    attribute {
      name: "multi_class"
      i: 1
      type: INT
    }
    attribute {
      name: "post_transform"
      s: "LOGISTIC"
      type: STRING
    }
    domain: "ai.onnx.ml"
  }
  node {
    input: "probability_tensor"
    output: "probabilities"
    name: "Normalizer"
    op_type: "Normalizer"
    attribute {
      name: "norm"
      s: "L1"
      type: STRING
    }
    domain: "ai.onnx.ml"
  }
  node {
    input: "label"
    output: "output_label"
    name: "Cast"
    op_type: "Cast"
    attribute {
      name: "to"
      i: 7
      type: INT
    }
    domain: ""
  }
  node {
    input: "probabilities"
    output: "output_probability"
    name: "ZipMap"
    op_type: "ZipMap"
    attribute {
      name: "classlabels_int64s"
      ints: 0
      ints: 1
      ints: 2
      type: INTS
    }
    domain: "ai.onnx.ml"
  }
  name: "deedadd605a34d41ac95746c4feeec1f"
  input {
    name: "float_input"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_value: 1
          }
          dim {
            dim_value: 4
          }
        }
      }
    }
  }
  output {
    name: "output_label"
    type {
      tensor_type {
        elem_type: 7
        shape {
          dim {
            dim_value: 1
          }
        }
      }
    }
  }
  output {
    name: "output_probability"
    type {
      sequence_type {
        elem_type {
          map_type {
            key_type: 7
            value_type {
              tensor_type {
                elem_type: 1
              }
            }
          }
        }
      }
    }
  }
}
opset_import {
  domain: ""
  version: 9
}
opset_import {
  domain: "ai.onnx.ml"
  version: 1
}
```

我们可以看到顶层字段记录了一些模型的元数据信息，代表的含义都比较直观，字段详细解释可以参考文档[IR.md](https://github.com/onnx/onnx/blob/master/docs/IR.md)。`opset_import`记录了该模型引入的操作符集。空的`domain`操作符集表示引入ONNX默认的操作符集`ai.onnx`。`ai.onnx.ml`代表支持传统非神经网络模型操作符集，比如以上模型中的`LinearClassifier`、`Normalizer`和`ZipMap`。图（graph）中定义了以下元素：

* 四个计算节点（node）。
* 一个输入变量`float_input`，类型为1*4的张量，`elem_type`是一个DataType枚举型变量，1代表FLOAT。
* 两个输出变量`output_label`和`output_probability`，`output_label`类型为维数为1的INT64（elem_type: 7）张量，代表预测目标分类； `output_probability`类型是映射的序列，映射的键是INT64（key_type: 7），值为维数为1的FLOAT，代表每一个目标分类的概率。

可以使用[netron](https://lutzroeder.github.io/netron/)，图像化显示ONNX模型的计算拓扑图，以上模型如下图：

   ![ONNX-graph](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/logreg_iris.onnx.png)

下面我们使用ONNX Runtime Python API预测该ONNX模型，当前仅使用了测试数据集中的第一条数据：

```python
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("logreg_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
probability_name = sess.get_outputs()[1].name
pred_onx = sess.run([label_name, probability_name], {input_name: X_test[0].astype(numpy.float32)})

# print info
print('input_name: ' + input_name)
print('label_name: ' + label_name)
print('probability_name: ' + probability_name)
print(X_test[0])
print(pred_onx)
```

打印的模型信息和预测值如下：

```
input_name: float_input
label_name: output_label
probability_name: output_probability
[5.5 2.6 4.4 1.2]
[array([1], dtype=int64), [{0: 0.012208569794893265, 1: 0.5704444646835327, 2: 0.4173469841480255}]]
```

完整的程序，可以参考以下notebook：[onnx.ipynb](https://github.com/aipredict/ai-deployment/blob/master/deploy-ml-dl-using-onnx/onnx.ipynb)

## ONNX与PMML
ONNX和PMML都是与平台和环境无关的模型表示标准，可以让模型部署脱离模型训练环境，简化了部署流程，加速模型快速上线到生产环境中。这两个标准都得到了各大厂商和框架的支持，得到了广泛的应用。

* PMML是一个比较成熟的标准，在ONNX诞生之前，可以说是模型表示的实际标准，对传统数据挖掘模型有丰富的支持，最新 [PMML4.4](http://dmg.org/pmml/v4-4/GeneralStructure.html) 可以支持多达19种模型类型。但是，目前PMML缺乏对深度学习模型的支持，下一版本5.0有可能会添加对深度神经网络的支持，但是因为PMML是基于老式的XML格式，使用文本格式来存储深度神经网络模型结构和参数会带来模型大小和性能的问题，目前该问题还没有一个完美的解决方案。关于PMML的详细介绍，可以参考文章[《使用PMML部署机器学习模型》](https://github.com/aipredict/ai-deployment/blob/master/deploy-ml-using-pmml/README.md)。

* ONNX作为一个新的标准，刚开始主要提供对深度神经网络模型的支持，解决模型在不同框架下互操作和交换的问题。目前通过`ONNX-ML`，ONNX已经可以支持传统非神经网络机器学习模型，但是目前模型类型还不够丰富。ONNX使用protobuf二进制格式来序列化模型，可以提供更好的传输性能。

ONNX和PMML这两种格式都有成熟的开源类库和框架支持，PMML有JPMML，PMML4S，PyPMML等。ONNX有微软的ONNX runtime，NVIDIA TensorRT等。用户可以根据自己的实际情况选择合适的跨平台格式来部署AI模型。

## DaaS简介

DaaS（Deployment-as-a-Service）是AutoDeployAI公司出品的AI模型自动部署系统，支持多种模型类型的上线部署，以下我们介绍如何在DaaS中使用ONNX格式来部署传统机器学习模型和深度神经网络学习模型，DaaS使用ONNX Runtime作为ONNX模型的执行引擎，ONNX Runtime是微软开源的ONNX预测类库，提供高性能预测服务功能。首先，登陆DaaS系统后，创建一个新的工程`ONNX`，下面的操作都在该工程下进行。关于DaaS的详细信息，可以参考文章[《在DaaS中部署PMML模型生成REST API》](https://github.com/aipredict/ai-deployment/blob/master/deploy-pmml-in-daas/README.md)。

## 使用ONNX部署传统机器学习模型

1. 导入模型。选择上面训练的Logistic Regression模型`logreg_iris.onnx`：

    ![daas-import-model-logreg](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-import-model-logreg.png)

    导入成功后，页面转到模型主页面。可以看到模型有一个输入字段`float_input`，类型是`tensor(float)`，维数`(1,4)`。两个输出字段：`output_label`和`output_probability`。

    ![daas-model-overview-logreg](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-model-overview-logreg.png)

2. 测试模型。点击标签页`测试`，输入预测数据`[[5.5, 2.6, 4.4, 1.2]]`，然后点击`提交`命令，输出页面显示预测测试结果：

    ![daas-model-test-logreg](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-model-test-logreg.png)

3. 创建默认实时预测Web服务。点击标签页`部署`，然后点击`添加服务`命令，输入服务名称，其他使用默认值：

    ![daas-create-web-service-logreg](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-create-web-service-logreg.png)

4. 测试Web服务。服务创建成功后，页面转到服务部署主页，当服务副本状态为`运行中`时，代表Web服务已经成功上线，可以接受外部请求。有两种方式测试该服务：
    
    * 在DaaS系统中通过测试页面。点击标签页`测试`，输入JSON格式的请求正文，点击`提交`命令:

      ![daas-test-web-service-logreg](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-test-web-service-logreg.png)


    * 通过任意的RSET客户端，使用标准的REST API来测试。这里我们使用curl命令行程序来调用Web服务，点击`生成代码`命令，弹出显示使用curl命令调用REST API的对话框：

      ![daas-curl-logreg](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-curl-logreg.png)


      复制该curl命令，打开shell页面，执行命令：

      ![daas-run-curl-logreg](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-run-curl-logreg.png)


## 使用ONNX部署深度神经网络模型

我们尝试部署[ONNX Model Zoo](https://github.com/onnx/models)中已经训练好的模型，这里我们选择[MNIST-手写数字识别](https://github.com/onnx/models/tree/master/vision/classification/mnist)CNN模型，下载基于ONNX1.3的模型最新版本：[mnist.tar.gz](https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz)。

1. 导入模型。选择已下载模型`mnist.tar.gz`：

    ![daas-import-model-mnist](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-import-model-mnist.png)

    导入成功后，页面转到模型主页面。可以看到模型有一个输入字段`Input3`，类型是`tensor(float)`，维数`(1,1,28,28)`。一个输出字段：`Plus214_Output_0`，类型同样是`tensor(float)`，维数`(1,10)`。

    ![daas-model-overview-mnist](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-model-overview-mnist.png)

2. 测试模型。点击标签页`测试`，然后点击`JSON`命令，DaaS系统会自动创建符合输入数据格式的随机数据，以方便测试。点击`提交`命令，输出页面显示预测测试结果：

    ![daas-model-test-mnist](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-model-test-mnist.png)

3. 创建自定义实施预测脚本。为了能支持输入图像，并且直接输出预测值，我们需要创建自定义预测脚本。点击标签页`实时预测`，然后点击`生成自定义实时预测脚本`命令，

    ![daas-generate-custom-scoring-mnist](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-generate-custom-scoring-mnist.png)

    脚本生成后，点击命令`作为API测试`，进入脚本测试页面，我们可以自由添加自定义预处理和后处理功能。添加以下函数预处理图像：

    ```python
    def rgb2gray(rgb):
        """Convert the input image into grayscale"""
        import numpy as np
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    def resize_img(img_to_resize):
        """Resize image to MNIST model input dimensions"""
        import cv2
        r_img = cv2.resize(img_to_resize, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        r_img.resize((1, 1, 28, 28))
        return r_img


    def preprocess_image(img_to_preprocess):
        """Resize input images and convert them to grayscale."""
        if img_to_preprocess.shape == (28, 28):
            img_to_preprocess.resize((1, 1, 28, 28))
            return img_to_preprocess

        grayscale = rgb2gray(img_to_preprocess)
        processed_img = resize_img(grayscale)
        return processed_img
    ```

    在已有`preprocess_files`函数中调用`preprocess_image`，代码如下：

    ```python
    import matplotlib.image as mpimg
    for key, file in files.items():
        img = mpimg.imread(file)
        record[key] = preprocess_image(img)
    ```

    在已有`postprocess`函数中添加如下代码后处理预测结果以获取最终的预测值：

    ```python
    def postprocess(result):
        """postprocess the predicted results"""
        import numpy as np
        return [int(np.argmax(np.array(result).squeeze(), axis=0))]
    ```

    点击命令`保存`，然后在请求页面中输入函数名为`predict`，选择请求正文基于`表单`，输入表单名称为模型唯一的输入字段名`Input3`，类型选择`文件`，点击上传，选择测试图像`2.png`，最后点击`提交`命令，测试该脚本是否按照我们的期望工作：

    ![daas-test-custom-scoring-mnist](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-test-custom-scoring-mnist.png)

4. 创建正式部署Web服务。当脚本测试成功后，点击`部署`标签页，然后点击`添加网络服务`命令，输入服务名称，其他使用默认值：

    ![daas-create-web-service-mnist](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-create-web-service-mnist.png)

5. 测试Web服务。服务创建成功后，页面转到服务部署主页，当服务副本状态为`运行中`时，代表Web服务已经成功上线，可以接受外部请求。有两种方式测试该服务：
    
    * 在DaaS系统中通过测试页面。点击标签页`测试`，选择请求正文基于`表单`，选择输入测试图像`5.jpg`，点击`提交`命令:

      ![daas-test-web-service-mnist.png](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-test-web-service-mnist.png)


    * 通过任意的RSET客户端，使用标准的REST API来测试。这里我们使用curl命令行程序来调用Web服务，点击`生成代码`命令，弹出显示使用curl命令调用REST API的对话框：

      ![daas-curl-mnist](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-curl-mnist.png)

      复制该curl命令，打开shell页面，切换到图像目录下，执行命令：

      ![daas-run-curl-mnist](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ml-dl-using-onnx/daas-run-curl-mnist.png)

## 总结
本文中我们介绍了ONNX这种跨平台AI模型表示标准，以及与PMML的区别，最后演示了如何在DaaS系统中通过ONNX部署传统机器学习模型和深度神经网络模型，可以看到ONNX让模型部署脱离了模型训练环境，极大简化了整个部署流程。

## 参考
* ONNX官网：[https://onnx.ai/](https://onnx.ai/)
* AutoDeployAI官网：[https://www.autodeploy.ai/](https://www.autodeploy.ai/)
* ONNX Github：[https://github.com/onnx/onnx](https://github.com/onnx/onnx)
* ONNX Runtime：[https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

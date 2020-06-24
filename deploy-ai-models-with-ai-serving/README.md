# 使用开源AI-Serving部署推理PMML和ONNX模型

## 目录
* [AI-Serving介绍](#AI-Serving介绍)
* [部署PMML模型](#部署PMML模型)
* [部署ONNX模型](#部署ONNX模型)
* [总结](#总结)
* [参考](#参考)

## AI-Serving介绍
AI-Serving是一款开源的机器学习和深度学习模型部署推理（inference）系统，支持标准的PMML和ONNX格式，同时提供HTTP和gRPC两种接口，方便在不同的生产环境中使用。

## 部署PMML模型
预测模型标记语言PMML（Predictive Model Markup Language）是一套成熟的表示经典机器学习模型的标准。AI-Serving通过PMML4S提供对PMML模型的高效预测服务，详情参考《[Inferencing Iris XGBoost PMML Model using AI-Serving](https://github.com/autodeployai/ai-serving/blob/master/examples/AIServingIrisXGBoostPMMLModel.ipynb)》。

关于PMML详情，参考文章《[使用PMML部署机器学习模型](https://github.com/aipredict/ai-deployment/blob/master/deploy-ml-using-pmml/README.md)》。

## 部署ONNX模型
开放神经网络交换ONNX（Open Neural Network Exchange）是一套表示深度神经网络模型的开放格式。AI-Serving通过ONNX Runtime提供高性能ONNX模型推断服务，详情参考《[Inferencing MNIST ONNX Model using AI-Serving](https://github.com/autodeployai/ai-serving/blob/master/examples/AIServingMnistOnnxModel.ipynb)》。

关于ONNX详情，参考文章《[使用ONNX部署深度学习和传统机器学习模型](https://github.com/aipredict/ai-deployment/blob/master/deploy-ml-dl-using-onnx/README.md)》。

## 总结
AI-Serving主要关注在标准交换格式的模型部署，目前PMML和ONNX是在部署机器学习中使用最广泛的的两种格式。其他格式，比如PFA，也会在后续的考虑中。

## 参考
* AI-Serving：[https://github.com/autodeployai/ai-serving](https://github.com/autodeployai/ai-serving)


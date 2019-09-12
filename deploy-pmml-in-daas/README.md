# 在DaaS中部署PMML模型生成REST API

## 目录
* [DaaS介绍](#DaaS介绍)
* [PMML简介](#PMML简介)
* [模型部署准备](#模型部署准备)
* [部署实时预测Web服务](#部署实时预测Web服务)
* [部署自定义实时预测Web服务](#部署自定义实时预测Web服务)
* [部署离线批量预测任务服务](#部署离线批量预测任务服务)
* [部署模型评估任务服务](#部署模型评估任务服务)
* [总结](#总结)
* [参考](#参考)

## DaaS介绍
DaaS（Deployment-as-a-Service）是AutoDeployAI公司推出的AI模型自动部署系统，支持PMML，Scikit-learn，XGBoost，LightGBM，Spark以及主流深度学习Keras，TensorFlow，Pytorch，MxNet等多种模型的部署。

DaaS基于Kubernetes构建，提供可靠和可扩展的模型部署服务，弹性部署用户AI和ML解决方案到生产环境中。用户可以自由选择在公有云或私有云的Kubernetes上安装DaaS系统，以满足用户对AI部署的多样需求。本文中的DaaS演示系统部署在本地的Minikube上。

DaaS设计框架：

![DaaS-Design](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-design.jpg)

## PMML简介
PMML是一套与平台和环境无关的AI模型序列化标准，为模型的跨平台部署提供了基础，简化了部署流程，可实现模型的快速上线。关于PMML的详细信息，可以参考文章[《使用PMML部署机器学习模型》](https://github.com/aipredict/ai-deployment/blob/master/deploy-ml-using-pmml/README.md)。

## 模型部署准备

DaaS系统提供多种模型部署方式，下面，我们演示在DaaS系统中如何部署模型，生成Web服务；如何通过REST API测试Web服务；以及如何部署任务来执行批量离线预测和模型评估操作。

在部署模型之前，我们需要完成以下操作：

1. 打开浏览器，登陆DaaS系统。

    ![DaaS-login](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-login.jpg)

2. 创建项目。登陆成功后，进入项目列表页，点击`新建项目`。

    ![DaaS-projects](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-projects.jpg)

    DaaS使用项目来管理用户的不同分析任务。项目中可以包含模型、部署、程序脚本、数据、数据源等多种分析资产。

    ![DaaS-new-project](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-new-project.jpg)

3. 导入模型。项目创建成功后，进入项目主页（仪表盘），切换到`模型`标签页，点击命令`导入模型`。

    ![DaaS-models](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-models.jpg)

    选择要部署的PMML模型文件，点击[此处](https://github.com/aipredict/ai-deployment/blob/master/deploy-ml-using-pmml/xgb-iris.pmml)可下载当前使用的PMML模型`xgb-iris.pmml`。在该流程中，首先会对模型进行验证，如果模型不是一个有效的PMML，会导致添加失败，DaaS将返回错误信息。

    ![DaaS-import-model](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-import-model.jpg)

4. 模型概述。PMML导入成功后，进入模型主页（`概述`），显示了模型的基本信息，比如输入和目标变量、模型类型、使用算法、运行引擎等。

    ![DaaS-model-overview](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-model-overview.jpg)

5. 测试模型。切换到`测试`标签页，通过表单输入数据或者点击`JSON`命令直接输入JSON格式的数据，然后点击`提交`命令，等待预测结果的返回。

    ![DaaS-model-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-model-test.jpg)

## 部署实时预测Web服务

1. 添加Web服务。当模型测试成功后，切换到`部署`标签页，点击命令`添加服务`。

    ![DaaS-model-deployments](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-model-deployments.jpg)

    这里有几个重要的部署选项：
    * 模型版本：当前模型只有一个版本，选择版本1。
    * 网络服务运行环境：指定模型部署运行的docker环境，DaaS默认包含两个网络部署环境分别针对Python 2.7和Python 3.7，每个环境都已经安装了以上常用模型库。我们可以在项目的`运行时定义`中查看系统中包含的运行时定义，DaaS允许用户添加自定义运行环境。这里选择`Python 3.7 - Function as a Service`。
    * 预留CPU和预留内存：为了降低系统的不稳定风险，用户可以选择为部署分配指定的CPU核数和内存量。
    * 副本：提供Web服务的负载均衡。默认为1。
    ![DaaS-add-service](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-add-service.jpg)

2. 测试Web服务。部署创建成功后，进入部署页面（`概述`），可以看到一个副本，状态是`启动中`，等待状态变成`运行中`后，该部署才算创建完成。这时候就可以接受预测请求。切换到`测试`标签页，在请求正文中输入测试数据，测试该服务。

    ![DaaS-web-service-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-web-service-test.jpg)

    DaaS为部署服务提供标准的REST API，可以通过任意REST客户端来调用，方便生产环境的集成。点击`生成代码`命令，会生成通过curl调用REST API的命令参数。

    ![DaaS-web-service-generrate-code](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-web-service-generrate-code.jpg)

    复制完整命令，在Shell中执行：

    ![DaaS-run-generrate-code-web-service](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-run-generrate-code-web-service.jpg)

    返回DaaS，切换到部署`模型`标签页。`指标`页面显示Web服务性能指标：执行次数、平均响应时间、最大最小时间等。可以看到，我们执行了二次调用：第一次通过DaaS部署测试界面，第二次通过在Shell中执行curl命令。一般来说，第一次调用是要慢一些，后面就会快很多。

    ![DaaS-web-service-overview](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-web-service-overview.jpg)

## 部署自定义实时预测Web服务
以上流程生成的是默认实时预测部署，只是调用模型本身进行预测。如果部署的Web服务需要执行一些额外的操作，比如增加数据的预处理功能。这时候我们可以使用DaaS提供的自定义实时预测功能：

1. 打开模型主页，切换到`实时预测`标签页，点击`生成自定义实时预测脚本`命令。

    ![DaaS-generate-custom-scoring](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-generate-custom-scoring.jpg)

    DaaS的Web服务部署功能是基于函数即服务实现的，我们可以在脚本的`prepare`函数中添加自定义的数据预处理操作。比如给每一个输入变量值增加0.2，完成后，点击右下方命令`作为API测试`，来测试输入的自定义代码是否按预期的工作。可以通过当前的测试页面来测试，也可以通过测试REST API来进行。

    ![DaaS-custom-scoring-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-custom-scoring-test.jpg)

2. 测试完成后，切换到`部署`标签页，点击`添加网络服务`，为该脚本生成正式的Web服务。后续操作与以上流程1、2相同。

    ![DaaS-custom-scoring-deployments](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-custom-scoring-deployments.jpg)

## 部署离线批量预测任务服务

除了部署网络（Web）服务，DaaS还支持部署任务（Job），在任务部署中我们可以完成一些模型相关的操作，比如批量预测，模型评估等。下面我们首先看一下如何部署模型批量预测任务。

1. 导入数据。首先，返回工程页面，切换到`数据集`标签页，点击`添加数据集`命令。
    ![DaaS-datasets-empty](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-datasets-empty.jpg)

    DaaS支持添加本地文件和远程数据，远程数据支持多种数据源（HDFS以及常用关系数据数库等）。当前我们添加一个本地CSV文件。

    ![DaaS-add-dataset](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-add-dataset.jpg)

2. 生成批预测脚本。打开模型页面，切换到`批量预测`标签，选择添加CSV文件`iris.csv`作为输入数据集，输入输出的数据集仍为本地文件`iris-batch-scoring.csv`。点击`生成批预测脚本`命令。

    ![DaaS-generate-batch-scoring](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-generate-batch-scoring.jpg)

3. 设置任务属性。脚本成功后，点击`高级设置`命令，弹出设置对话框，有几个重要的参数：
    * 任务运行环境：执行该脚本任务的docker环境，和Web服务运行环境类似，DaaS默认包含两个任务部署环境分别针对Python 2.7和Python 3.7，每个环境都已经安装了以上常用模型库。我们可以在项目的`运行时定义`中查看系统中包含的运行环境定义，允许用户添加自定义运行环境。这里默认选择`Python 3.7 - Script as a Service`。
    * 环境变量和命令参数：设置执行脚本的系统环境变量和命令行参数，默认是空。
    * 调度：设置任务执行调度，可以按需或者按计划，比如每天几时执行该任务。默认`按需`。

    ![DaaS-job-settings](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-job-settings.jpg)

4. 任务部署。脚本成功后，点击`立即执行`命令。DaaS会自动部署任务，并且立刻运行一次该任务。页面自动转移到任务部署界面，我们可以看到现在正在执行的一次运行操作。

    ![DaaS-job](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-job.jpg)

5. 查看结果。当执行状态由`启动中`和`运行中`变成`成功`后，返回到项目界面，切换到`数据集`标签页，可以看到多了一项新生成的数据文件`iris-batch-scoring.csv`，包含模型批量预测结果。在`操作`菜单下，可以选择预览或者下载该数据。

    ![DaaS-datasets](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-datasets.jpg)

6. 再次执行该任务。返回到任务部署页面，切换到`测试`标签页，同Web服务部署类似，可以在该页面或者通过REST API执行相关操作，对于任务操作，主要包含以下命令：
    * 执行任务：输入可选的环境变量和命令参数。
    * 获取某次执行的状态：必须提供该执行的ID
    * 停止某次执行：同上，必须提供该执行的ID

    ![DaaS-job-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-job-test.jpg)

    点击`生成代码`命令，同样提供可执行的curl命令调用对应的REST API，在Shell中运行`执行任务`:

    ![DaaS-run-generate-code-job](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-run-generate-code-job.jpg)
    
## 部署模型评估任务服务：
1. 导入数据。仍然使用批量预测添加的本地CSV文件`iris.csv`。

2. 生成模型评估脚本。打开模型页面，切换到`模型评估`标签页，选择该CSV文件为输入数据集，选择评估指标`Accuracy Score`，评估阀值使用默认的[0.3, 0.7]。点击`生成模型评估脚本`命令。

    ![DaaS-generate-model-evaluation](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-generate-model-evaluation.jpg)

3. 设置任务属性。参考以上流程3

4. 任务部署。参考以上流程4

5. 查看结果。当执行状态由`运行中`变成`成功`后，返回到模型界面（概述），查看`评估结果`，第一项就是最新的模型评估结果。

    ![DaaS-model-overview-evaluation](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pmml-in-daas/daas-model-overview-evaluation.jpg)

6. 再次执行该任务。参考以上流程6

## 总结
通过以上的演示，我们可以看到，DaaS作为一个通用AI模型管理部署平台，具有很大的灵活性和可扩展性，能满足用户的各种自定义部署需求。

## 参考
AutoDeployAI官网：[https://www.autodeploy.ai/](https://www.autodeploy.ai/)

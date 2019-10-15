# 自动部署开源AI模型到生产环境：Scikit-learn、XGBoost、LightGBM、和PySpark

## 目录
* [背景介绍](#背景介绍)
* [部署准备](#部署准备)
* [部署Scikit-learn模型](#部署Scikit-learn模型)
* [部署XGBoost模型](#部署XGBoost模型)
* [部署LightGBM模型](#部署LightGBM模型)
* [部署PySpark模型](#部署PySpark模型)
* [模型部署管理](#模型部署管理)
* [总结](#总结)
* [参考](#参考)

## 背景介绍
AI的广泛应用是由AI在开源技术的进步推动的，利用功能强大的开源模型库，数据科学家们可以很容易的训练一个性能不错的模型。但是因为模型生产环境和开发环境的不同，涉及到不同角色人员：模型训练是数据科学家和数据分析师的工作，但是模型部署是开发和运维工程师的事情，导致模型上线部署却不是那么容易。

DaaS（Deployment-as-a-Service）是AutoDeployAI公司推出的基于Kubernetes的AI模型自动部署系统，提供一键式自动部署开源AI模型生成REST API，以方便在生产环境中调用。下面，我们主要演示在DaaS中如何部署经典机器学习模型，包括Scikit-learn、XGBoost、LightGBM、和PySpark ML Pipelines。关于深度学习模型的部署，会在下一章中介绍。

## 部署准备
我们使用DaaS提供的Python客户端（DaaS-Client）来部署模型，对于XGBoost和LightGBM，我们同样使用它们的Python API来作模型训练。在训练和部署模型之前，我们需要完成以下操作。

1. 安装Python DaaS-Client。

    ```
    pip install --upgrade git+https://github.com/autodeployai/daas-client.git
    ```

2. 初始化DaasClient。使用DaaS系统的URL、账户、密码登陆系统，文本使用的DaaS演示系统安装在本地的Minikube上。完整Jupyter Notebook，请参考：[deploy-sklearn-xgboost-lightgbm-pyspark.ipynb](https://github.com/aipredict/ai-deployment/blob/master/deploy-ai-models-in-daas/deploy-sklearn-xgboost-lightgbm-pyspark.ipynb)

    ```python
    from daas_client import DaasClient

    client = DaasClient('https://192.168.64.3:30931', 'username', 'password')
    ```

3. 创建项目。DaaS使用项目管理用户不同的分析任务，一个项目中可以包含用户的各种分析资产：模型、部署、程序脚本、数据、数据源等。项目创建成功后，设置为当前活动项目，发布的模型和创建的部署都会存储在该项目下。`create_project`函数接受三个参数：

    1. 项目名称：可以是任意有效的Linux文件目录名。
    2. 项目路由：使用在部署的REST URL中来唯一表示当前项目，只能是小写英文字符(a-z)，数字(0-9)和中横线`-`，并且`-`不能在开头和结尾处。
    3. 项目说明（可选）：可以是任意字符。

    ```python
    project = '部署测试'
    if not client.project_exists(project):
        client.create_project(project, 'deployment-test', '部署测试项目')
    client.set_project(project)
    ```

4. 初始化数据。我们使用流行的分类数据集`iris`来训练不同的模型，并且把数据分割为训练数据集和测试数据集以方便后续使用。

    ```python
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import pandas as pd

    seed = 123456

    iris = datasets.load_iris()
    iris_target_name = 'Species'
    iris_feature_names = iris.feature_names
    iris_df = pd.DataFrame(iris.data, columns=iris_feature_names)
    iris_df[iris_target_name] = iris.target

    X, y = iris_df[iris_feature_names], iris_df[iris_target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)    
    ```

5. 模型部署流程。主要包含以下几步：

    1. 训练模型。使用模型库提供的API，在`iris`数据集上训练模型。
    2. 发布模型。调用`publish`函数发布模型到DaaS系统。
    3. 测试模型（可选）。调用`test`函数获取测试API信息，可以使用任意的REST客户端程序测试模型在DaaS中是否工作正常，使用的是DaaS系统模型测试API。第一次执行`test`会比较慢，因为DaaS系统需要启动测试运行时环境。
    4. 部署模型。发布成功后，调用`deploy`函数部署部署模型。可以使用任意的REST客户端程序测试模型部署，使用的是DaaS系统正式部署API。

## 部署Scikit-learn模型

1. 训练一个Scikit-learn分类模型：SVC。

    ```python
    from sklearn.svm import SVC

    model = SVC(probability=True, random_state=seed)
    model.fit(X_train, y_train)
    ```

2. 发布Scikit-learn模型。

    ```python
    publish_resp = client.publish(model,
                                name='iris',
                                mining_function='classification',
                                X_test=X_test,
                                y_test=y_test,
                                description='A SVC model')
    pprint(publish_resp)
    ```

    `test`函数必须要指定前两个参数，第一个`model`是训练的模型对象，第二个是模型名称，其余是可选参数：
    * mining_function：指定挖掘功能，可以指定为`regression`（回归）、`classification`（分类）、和`clustering`（聚类）。
    * X_test和y_test：指定测试训练集，发布时计算模型评估指标，比如针对分类模型，计算正确率（Accuracy），对于回归模型，计算可释方差（explained Variance）。
    * data_test： 同样是指定测试训练集，但是该参数用在Spark模型上，非Spark模型通过`X_test`和`y_test`指定。
    * description：模型描述。
    * params：记录模型参数设置。

    `publish_resp`是一个字典类型的结果，记录了模型名称，和发布的模型版本。该模型是`iris`模型的第一个版本。
    
    ```python
    {'model_name': 'iris', 'model_version': '1'}
    ```
3. 测试Scikit-learn模型。

    ```
    test_resp = client.test(publish_resp['model_name'], model_version=publish_resp['model_version'])
    pprint(test_resp)
    ```

    `test_resp`是一个字典类型的结果，记录了测试REST API信息。如下，其中`access_token`是访问令牌，一个长字符串，这里没有显示出来。`endpoint_url`指定测试REST API地址，`payload`提供了测试当前模型需要输入的请求正文格式。

    ```python
    {'access_token': 'A-LONG-STRING-OF-BEARER-TOKEN-USED-IN-HTTP-HEADER-AUTHORIZATION',
    'endpoint_url': 'https://192.168.64.3:30931/api/v1/test/deployment-test/daas-python37-faas/test',
    'payload': {'args': {'X': [{'petal length (cm)': 1.5,
                                'petal width (cm)': 0.4,
                                'sepal length (cm)': 5.7,
                                'sepal width (cm)': 4.4}],
                        'model_name': 'iris',
                        'model_version': '1'}}}
    ```

    使用requests调用测试API，这里我们直接使用`test_resp`返回的测试payload，您也可以使用自定义的数据`X`，但是参数`model_name`和`model_version`必须使用上面输出的值。

    ```python
    response = requests.post(test_resp['endpoint_url'],
                            headers={'Authorization': 'Bearer {token}'.format(token=test_resp['access_token'])},
                            json=test_resp['payload'],
                            verify=False)
    pprint(response.json())
    ```

    返回结果，不同于正式部署API，除了预测结果，测试API会同时返回标准控制台输出和标准错误输出内容，以方便用户碰到错误时，查看相关信息。

    ```python
    {'result': [{'PredictedValue': 0,
                'Probabilities': [0.8977133931668801,
                                0.05476023239878367,
                                0.047526374434336216]}],
    'stderr': [],
    'stdout': []}
    ```

4. 部署模型。

    ```python
    deploy_resp = client.deploy(model_name='iris', 
                                deployment_name='iris-svc',
                                model_version=publish_resp['model_version'],
                                replicas=1)
    pprint(deploy_resp)
    ```

    `deploy`函数必须要指定模型名称，和部署名称。模型版本默认为当前最新版本（`latest`），副本数默认是1。为了确保部署服务的稳定性，还可以输入部署运行时环境分配指定CPU核数和使用内存量，默认为None，让系统自动分配。

    `deploy_resp`是一个字典类型的结果，记录了正式部署REST API信息。如下，可以看到和测试结果类似，在`payload`中，我们不需要在输入模型名称和版本，因为正式部署服务在创建是已经记录了这些信息，并且是一个独占式服务。

    ```python
    {'access_token': 'A-LONG-STRING-OF-BEARER-TOKEN-USED-IN-HTTP-HEADER-AUTHORIZATION',
    'endpoint_url': 'https://192.168.64.3:30931/api/v1/svc/deployment-test/iris-svc/predict',
    'payload': {'args': {'X': [{'petal length (cm)': 1.5,
                                'petal width (cm)': 0.4,
                                'sepal length (cm)': 5.7,
                                'sepal width (cm)': 4.4}]}}}
    ```

    使用requests调用测试API，这里我们直接使用`test_resp`返回的测试payload，您也可以使用自定义的数据。

    ```python
    response = requests.post(deploy_resp['endpoint_url'],
                         headers={'Authorization': 'Bearer {token}'.format(token=deploy_resp['access_token'])},
                         json=deploy_resp['payload'],
                         verify=False)
    pprint(response.json())
    ```

    返回结果:

    ```python
    {'result': [{'PredictedValue': 0,
                'Probabilities': [0.8977133931668801,
                                0.05476023239878367,
                                0.047526374434336216]}]}
    ```

## 部署XGBoost模型
XGBoost提供了两套Python API，一套是原生Python API，另一套是基于Scikit-learn包装API。您可以使用任何一种，下面的例子中我们使用基于Scikit-learn的Python API。

1. 训练一个分类XGBoost模型：

    ```python
    from xgboost import XGBClassifier

    model = XGBClassifier(max_depth=3, objective='multi:softprob', random_state=seed)
    model = model.fit(X_train, y_train)
    ```

2. 发布XGBoost模型。

    ```python
    publish_resp = client.publish(model,
                                name='iris',
                                mining_function='classification',
                                X_test=X_test,
                                y_test=y_test,
                                description='A XGBClassifier model')
    pprint(publish_resp)
    ```

    因为仍然使用了`iris`这个模型名称，所以该模型是`iris`的第二个版本。
    
    ```python
    {'model_name': 'iris', 'model_version': '2'}
    ```
3. 测试XGBoost模型。和Scikit-learn流程相同。
4. 部署模型。和Scikit-learn流程相同，这里我们暂时先不创建独立部署，后面我们会介绍如何在DaaS系统中管理部署，如何切换部署模型版本。

## 部署LightGBM模型
同XGBoost类似，LightGBM同样提供了两套Python API，一套是原生Python API，另一套是基于Scikit-learn包装API。您可以使用任何一种，下面的例子中我们使用基于Scikit-learn的Python API。

1. 训练一个分类LightGBM模型：

    ```python
    from lightgbm import LGBMClassifier

    model = LGBMClassifier()
    model = model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    ```

2. 发布LightGBM模型。

    ```python
    publish_resp = client.publish(model,
                                name='iris',
                                mining_function='classification',
                                X_test=X_test,
                                y_test=y_test,
                                description='A LGBMClassifier model')
    pprint(publish_resp)
    ```

    LightGBM模型是`iris`的第三个版本。
    
    ```python
    {'model_name': 'iris', 'model_version': '3'}
    ```
3. 测试LightGBM模型。和Scikit-learn流程相同。
4. 部署模型。和Scikit-learn流程相同，这里我们暂时先不创建独立部署。

## 部署PySpark模型

1. 训练一个PySpark分类模型：RandomForestClassifier。PySpark模型必须是一个`PipelineModel`，也就是说必须使用Pipeline来建立模型，哪怕只有一个Pipeline节点。

    ```python
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml import Pipeline

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(iris_df)

    df_train, df_test = df.randomSplit([0.7, 0.3], seed=seed)
    assembler = VectorAssembler(inputCols=iris_feature_names,
                                outputCol='features')

    rf = RandomForestClassifier(seed=seed).setLabelCol(iris_target_name)
    pipe = Pipeline(stages=[assembler, rf])
    model = pipe.fit(df_train)
    ```

2. 发布PySpark模型。

    ```python
    publish_resp = client.publish(model,
                                name='iris',
                                mining_function='classification',
                                data_test=df_test,
                                description='A RandomForestClassifier of Spark model')
    pprint(publish_resp)
    ```

    PySpark模型是`iris`的第四个版本。
    
    ```python
    {'model_name': 'iris', 'model_version': '4'}
    ```
3. 测试PySpark模型。和Scikit-learn流程相同。
4. 部署模型。和Scikit-learn流程相同，这里我们暂时先不创建独立部署。

## 模型部署管理
打开浏览器，登陆DaaS管理系统。进入项目`部署测试`，切换到`模型`标签页，有一个`iris`模型，最新版本是`v4`，类型是`Spark`即我们最后发布的模型。

![DaaS-models](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ai-models-in-daas/daas-models.jpg)

点击模型，进入模型主页（概述）。当前`v4`是一个Spark Pipeline模型，正确率是94.23%，并且显示了`iris`不同版本正确率历史图。下面罗列了模型的输入和输出变量，以及评估结果，当前为空，因为还没有在DaaS中执行任何的模型评估任务。

![DaaS-model-overview-v4](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ai-models-in-daas/daas-model-overview-v4.jpg)

点击`v4`，可以自由切换到其他版本。比如，切换到`v1`。

![DaaS-model-versions](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ai-models-in-daas/daas-model-versions.jpg)

`v1`版本是一个Scikit-learn SVM分类模型，正确率是98.00%。其他信息与`v4`类似。

![DaaS-model-overview-v1](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ai-models-in-daas/daas-model-overview-v1.jpg)

切换到模型`部署`标签页，有一个我们刚才创建的部署`iris-svc`，鼠标移动到操作菜单，选择`修改设置`。可以看到，当前部署服务关联的是模型`v1`，就是我们刚才通过`deploy`函数部署的`iris`第一个版本Scikit-learn模型。选择最新的`v4`，点击命令`保存并且重新部署`，该部署就会切换到`v4`版本。

![DaaS-edit-service](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-ai-models-in-daas/daas-edit-service.jpg)

## 总结
通过Python DaaS-Client我们可以很容易的部署训练好的模型，并且在DaaS网络客户端管理这些模型和部署，可以支持自由切换部署中的模型版本。除了支持部署网络（Web）服务，DaaS还支持部署任务（Job）服务，通过任务我们可以运行离线批量预测和模型评估等，具体可以参考文章[《在DaaS中部署PMML模型生成REST API》](https://github.com/aipredict/ai-deployment/blob/master/deploy-pmml-in-daas/README.md)。

## 参考
* DaaS-Client：[https://github.com/autodeployai/daas-client](https://github.com/autodeployai/daas-client)
* AutoDeployAI：[https://www.autodeploy.ai/](https://www.autodeploy.ai/)

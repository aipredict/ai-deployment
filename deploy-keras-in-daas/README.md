# 自动部署深度神经网络模型TensorFlow（Keras）到生产环境中

## 目录
* [Keras简介](#Keras简介)
* [Keras模型分类](#Keras模型分类)
* [Keras模型部署准备](#Keras模型部署准备)
* [默认部署Keras模型](#默认部署Keras模型)
* [自定义部署Keras模型](#自定义部署Keras模型)
* [总结](#总结)
* [参考](#参考)

## Keras简介
[Keras](https://keras.io/)是一套由Python实现的高级神经网络API，可以基于不同的后台（backend）实现，包括[TensoreFlow](https://www.tensorflow.org/guide/keras/overview)，[CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras)，[MXNet](https://mxnet.apache.org/)（基于[Keras-MXNet](https://pypi.org/project/keras-mxnet/)，处于孵化阶段）和[Theano](https://github.com/Theano/Theano)（已停止开发）。Keras可以让用户能快速的开发出原型，支持流行的卷积网络（Convolutional Networks）、循环网络（Recurrent Networks）或者两者的组合，并且可以无缝的在CPU和GPU上运行。

## Keras模型分类
Keras提供了两种类型的模型，顺序（Sequential）模型和函数式（Functional）模型：
* 顺序模型，由一组线性的层堆栈组成（a linear stack of layers），是一种最简单的模型类型，API简单明了，详情查看官方文档：[Guide to the Sequential model](https://keras.io/getting-started/sequential-model-guide/)
* 函数式模型，可以定义更复杂的模型网络结构，比如多输出模型，有向无环图，具有共享层的多模型，详情查看官方文档：[Guide to the Functional API](https://keras.io/getting-started/functional-api-guide/)

以上两套API，生成的模型类型是keras.models.Sequential（继承自Model）和keras.models.Model类，除了以上两种类型模型，从Keras 2.2.0开始，支持用户继承keras.models.Model来创建完全自定义化模型类型，详情查看官方文档模型自类化[（Model subclassing）](https://keras.io/models/about-keras-models/#model-subclassing)。本文中我们主要讨论非子类化（non-subclassing）模型的部署，关于子类化模型（subclassing）的部署会在后面介           绍。

## Keras模型部署准备
在上一篇文章《[自动部署开源AI模型到生产环境：Sklearn、XGBoost、LightGBM、和PySpark](https://github.com/aipredict/ai-deployment/blob/master/deploy-ai-models-in-daas/README.md)》中我们介绍了如何通过AutoDeployAI的AI模型自动部署和管理系统DaaS（Deployment-as-a-Service）来自动部署传统开源机器学习模型（包括Scikit-learn、XGBoost、LightGBM、和PySpark等），这里我们详细介绍如果通过DaaS来自动部署Keras深度神经网络模型，同样我们需要：

* 安装Python [DaaS-Client](https://github.com/autodeployai/daas-client)
* 初始化DaasClient
* 创建项目

详细准备工作，请参考以上文章中的`部署准备`部分。完整的代码，请参考Github上的Notebook：[deploy-keras.ipynb](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-keras-in-daas/deploy-keras.ipynb)

## 默认部署Keras模型
1. 基于Keras Sequence API训练一个简单的多分类模型，使用Scikit-learn中的`Iris`数据：

```
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(Adam(lr=0.04), 'categorical_crossentropy', ['accuracy'])
model.fit(X_train, pd.get_dummies(y_train).values, epochs=100)
```

2. 发布Keras模型：

```
publish_resp = client.publish(model,
                              name='keras-iris',
                              mining_function='classification',
                              x_test=X_test,
                              y_test=y_test,
                              description='A Keras classification model')
pprint(publish_resp)
```

publish_resp是一个字典类型的结果，记录了模型名称，和发布的模型版本。

```
{'model_name': 'keras-iris', 'model_version': '1'}
```

3. 测试Keras模型：

```
test_resp = client.test(publish_resp['model_name'], 
                        model_version=publish_resp['model_version'])
pprint(test_resp)
```
test_resp是一个字典类型的结果，记录了测试REST API信息，如下：

```
{'access_token': 'A-LONG-STRING-OF-BEARER-TOKEN-USED-IN-HTTP-HEADER-AUTHORIZATION',
 'endpoint_url': 'https://daas.autodeploy.ai/api/v1/test/deployment-test/daas-python37-faas/test',
 'payload': {'args': {'X': [{'dense_1_input': [5.7, 4.4, 1.5, 0.4]}],
                      'model_name': 'keras-iris',
                      'model_version': '1'}}}
```

使用`requests`库调用测试API：

```
response = requests.post(test_resp['endpoint_url'],
                         headers={'Authorization': 'Bearer {token}'.format(token=test_resp['access_token'])},
                         json=test_resp['payload'],
                         verify=False)
pprint(response.json())
```

返回结果：

```
{'result': [{'dense_4': [[0.996542751789093,
                          0.0034567660186439753,
                          4.955750227964018e-07]]}],
 'stderr': [],
 'stdout': []}
```

4. 正式部署Keras模型：

```
deploy_resp = client.deploy(model_name='keras-iris', 
                            deployment_name='keras-iris-svc',
                            model_version=publish_resp['model_version'],
                            replicas=1)
pprint(deploy_resp)
```

返回结果：

```
{'access_token': 'A-LONG-STRING-OF-BEARER-TOKEN-USED-IN-HTTP-HEADER-AUTHORIZATION',
 'endpoint_url': 'https://daas.autodeploy.ai/api/v1/svc/deployment-test/keras-iris-svc/predict',
 'payload': {'args': {'X': [{'dense_1_input': [5.7, 4.4, 1.5, 0.4]}]}}}
```

使用`requests`库调用正式API：

```
response = requests.post(deploy_resp['endpoint_url'],
                         headers={'Authorization': 'Bearer {token}'.format(token=deploy_resp['access_token'])},
                         json=deploy_resp['payload'],
                         verify=False)
pprint(response.json())
```

结果如下：

```
{'result': [{'dense_4': [[0.996542751789093,
                          0.0034567660186439753,
                          4.955750227964018e-07]]}]}
```

除了使用Keras API，对于TensorFlow，DaaS也同样支持使用tf.keras API训练的模型。

## 自定义部署Keras模型
1. 基于tf.Keras Functional API训练模型，使用Keras中的`MNist`数据来识别用户输入的数字，以下代码参考[Functional API on MNIST](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/cnn-functional-2.1.1.py)：

```
# load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# from sparse label to categorical
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# reshape and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3

# use functional API to build cnn layers
inputs = Input(shape=input_shape)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(inputs)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(y)
y = MaxPooling2D()(y)
y = Conv2D(filters=filters,
           kernel_size=kernel_size,
           activation='relu')(y)
# image to vector before connecting to dense layer
y = Flatten()(y)
# dropout regularization
y = Dropout(dropout)(y)
outputs = Dense(num_labels, activation='softmax')(y)

# build the model by supplying inputs/outputs
model = Model(inputs=inputs, outputs=outputs)
# network model in text
model.summary()

# classifier loss, Adam optimizer, classifier accuracy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train the model with input images and labels
model.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          epochs=3,
          batch_size=batch_size)
```
为了可以快速训练出模型，修改原程序中epochs=20为3。

2. 发布Keras模型：

```
publish_resp = client.publish(model,
                              name='keras-mnist',
                              mining_function='classification',
                              x_test=x_test,
                              y_test=y_test,
                              description='A tf.Keras classification model')
pprint(publish_resp)
```

结果如下：

```
{'model_name': 'keras-mnist', 'model_version': '1'}
```

3. 测试Keras模型。登陆DaaS Web客户端，查看`keras-mnist`模型信息：

![DaaS-model-overview](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-keras-in-daas/daas-model-overview.png)

模型输入字段`input_1`，维数为(,28,28,1)，输出字段`dense`，维数为(,10)。切换到`测试`标签页，我们看到DaaS自动存储了一条测试数据，点击`提交`命令，测试该条数据，如图：

![DaaS-model-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-keras-in-daas/daas-model-test.png)

4. 自定义部署Keras模型。`keras-mnist`模型输入数据是由一张28*28的黑底白字灰度图像生成的numpy数组，在REST API使用JSON传输数据时，因为JSON作为一种文本格式，在存储传输大的列表时有性能劣势，并且需要调用端做图像预处理工作，增加了客户端使用的负担。我们希望这些都可以在服务器端完成：接收二进制图像文件，预处理图像，并且转成模型需要的numpy数组。在DaaS中，我们可以通过创建自定义部署脚本来完成该该任务，它允许用户在模型部署中添加任意的数据预处理和后处理操作。切换到`实时预测`标签页，点击命令`生成自定义实时预测脚本`，生成预定义脚本：

![DaaS-model-realtime](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-keras-in-daas/daas-model-realtime.png)

点击`作为API测试`命令，页面切换到测试页面，修改`preprocess_files`函数，并且添加处理图像的函数：

```
def rgb2gray(rgb):
    """Convert the input image into grayscale"""
    import numpy as np
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def resize_img(img_to_resize):
    """Resize image to MNIST model input dimensions"""
    import cv2
    r_img = cv2.resize(img_to_resize, dsize=(28, 28), interpolation=cv2.INTER_AREA)
    r_img.resize((1, 28, 28, 1))
    r_img = 1 - r_img
    return r_img


def preprocess_image(img_to_preprocess):
    """Resize input images and convert them to grayscale."""
    if img_to_preprocess.shape == (28, 28):
        img_to_preprocess.resize((1, 28, 28, 1))
        img_to_preprocess = 1 - img_to_preprocess / 255
        return img_to_preprocess

    grayscale = rgb2gray(img_to_preprocess)
    processed_img = resize_img(grayscale)
    return processed_img


def preprocess_files(args):
    """preprocess the uploaded files"""
    files = args.get('files')
    if files is not None:
        # get the first record object in X if it's present
        if 'X' in args:
            record = args['X'][0]
        else:
            record = {}
            args['X'] = [record]

        # TODO add your own custom opeartions, e.g. loading images, make transformation, then write back into X
        import matplotlib.image as mpimg
        for key, file in files.items():
            img = mpimg.imread(file)
            record[key] = preprocess_image(img)

    return args
```

输入函数名`predict`，选择请求正文基于`表单`，输入名称`input_1`，选择文件，点击上传测试图像`2.png`，点击`提交`，右侧响应页面显示结果为：

```
{
  "result": [
    {
      "dense": [
        [
          4.0412282942270394e-7,
          5.612335129967505e-8,
          0.9999896287918091,
          0.0000014349453749673557,
          2.1778572326623669e-13,
          3.3688251840913175e-12,
          1.8931906042851665e-10,
          1.7151558395767097e-8,
          0.000008489014362567104,
          6.33769703384246e-10
        ]
      ]
    }
  ],
  "stderr": [
    "Some warning messages here"
  ],
  "stdout": []
}
```

继续修改`postprocess`函数为：

```
def postprocess(result):
    """postprocess the predicted results"""
    import numpy as np
    return [int(np.argmax(np.array(result).squeeze(), axis=0))]
```

重新`提交`，右侧响应页面显示结果为：

![DaaS-model-script-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-keras-in-daas/daas-model-script-test.png)

测试完成后，可以创建正式的部署，切换到`部署`标签页，点击命令`添加网络服务`，输入服务名称`mnist-svc`，其他使用默认选项，点击`创建`。进入到部署页面后，点击`测试`标签页，该界面类似之前的脚本测试界面，输入函数名`predict`，请求正文选择基于`表单`，输入名称`input_1`，类型选择文件，点击上传测试的图片后，点击提交：

![DaaS-deployment-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-keras-in-daas/daas-deployment-test.png)

到此，正式部署已经测试和创建完成，用户可以使用任意的客户端程序调用该部署服务。点击以上界面中的`生成代码`命令，显示如何通过curl命令调用该服务，测试如下：

![DaaS-deployment-curl-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-keras-in-daas/daas-deployment-curl-test.png)

5. 通过ONNX部署Keras模型：

    * 转换模型到ONNX：

    ```
    import onnxmltools

    onnx_model = onnxmltools.convert_keras(model, model.name)
    ``` 

    * 发布ONNX模型：

    ```
    publish_resp = client.publish(onnx_model,
                                  name='keras-mnist-onnx',
                                  mining_function='classification',
                                  x_test=x_test,
                                  y_test=y_test,
                                  description='A tf.Keras classification model in ONNX')
    pprint(publish_resp)
    ```

    结果如下：

    ```
    {'model_name': 'keras-mnist-onnx', 'model_version': '1'}
    ```

    * 测试ONNX模型：登陆DaaS Web客户端，查看`keras-onnx-mnist`模型信息：

    ![DaaS-onnx-model-overview](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-keras-in-daas/daas-onnx-model-overview.png)

    模型输入字段`input_1`，维数为(N,28,28,1)，输出字段`dense`，维数为(N,10)。切换到`测试`标签页，我们看到DaaS自动存储了一条测试数据，点击`提交`命令，测试该条数据，如图：

    ![DaaS-onnx-model-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-keras-in-daas/daas-onnx-model-test.png)

    我们看到，该ONNX模型和原生Keras模型测试结果是一致的。

    * 自定义部署ONNX模型：参考以上Keras模型，流程相同，就不再这里赘述。关于ONNX格式详情以及部署，可以参考文章《[使用ONNX部署深度学习和传统机器学习模型](https://github.com/aipredict/ai-deployment/blob/master/deploy-ml-dl-using-onnx/README.md)》


## 总结
通过以上的演示，我们可以看到，DaaS在部署深度学习模型时的优势：既可以一键式创建默认部署，又可以灵活的自定义部署，满足用户多样的部署需求。关于其他深度学习框架的部署，比如Pytorch，MXNet等，会在后续的文章中介绍。

## 参考
* DaaS-Client：[https://github.com/autodeployai/daas-client](https://github.com/autodeployai/daas-client)
* AutoDeployAI：[https://www.autodeploy.ai/](https://www.autodeploy.ai/)

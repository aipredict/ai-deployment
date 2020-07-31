#  如何上线部署Pytorch深度学习模型到生产环境中

## 目录
* [Pytorch模型部署准备](#Pytorch模型部署准备)
* [Pytorch自定义运行时](#Pytorch自定义运行时)
* [默认部署Pytorch模型](#默认部署Pytorch模型)
* [自定义部署Pytorch模型](#自定义部署Pytorch模型)
* [通过ONNX部署Pytorch模型](#通过ONNX部署Pytorch模型)
* [试用DaaS(Deployment-as-a-Service)](#试用DaaS(Deployment-as-a-Service))
* [参考](#参考)


## Pytorch模型部署准备
[Pytorch](https://pytorch.org/)和[TensorFlow](https://www.tensorflow.org/)是目前使用最广泛的两种深度学习框架，在上一篇文章《[自动部署深度神经网络模型TensorFlow（Keras）到生产环境中](https://github.com/aipredict/ai-deployment/blob/master/deploy-keras-in-daas/README.md)》中我们介绍了如何通过AutoDeployAI的AI模型部署和管理系统DaaS（Deployment-as-a-Service）来自动部署TensorFlow模型，本篇我们将介绍如果通过DaaS来自动部署Pytorch深度神经网络模型，同样我们需要：

* 安装Python [DaaS-Client](https://github.com/autodeployai/daas-client)
* 初始化DaasClient
* 创建项目

完整的代码，请参考Github上的Notebook：[deploy-pytorch.ipynb](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/deploy-pytorch.ipynb)

## Pytorch自定义运行时
DaaS是基于Kubernetes的AI模型自动部署系统，模型运行在Docker Container中，在DaaS中被称为运行时（Runtime），有两类不同的运行时，分别为网络服务运行环境（Environment）和任务运行环境（Worker）。Environment用于创建网络服务（Web Service），而Worker用于执行任务（Job）的部署，比如模型评估和批量预测等。DaaS默认自带了四套运行时，分别针对Environment和Workder基于不同语言Python2.7和Python3.7，自带了大部分常用的机器学习和深度学习类库，但是因为Docker镜像（Image）大小的缘故，暂时没有包含Pytorch库。

DaaS提供了自定义运行时功能，允许用户把自定义Docker镜像注册为Runtime，满足用户使用不同模型类型，模型版本的定制需求。下面，我们以部署Pytorch模型为例，详细介绍如何创建自定义运行时:

1. 构建Docker镜像：

一般来说，有两种方式创建Image，一种是通过Dockerfile构建（docker build），一种是通过Container生成（docker commit），这里我们使用第一种方式。无论那一种方式，都需要选定一个基础镜像，这里为了方便构建，我们选择了Pytorch官方镜像`pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime`。

为了创建网络服务运行时，除了包含模型运行的依赖类库外，还需要额外安装网络服务的一些基础库，完整的列表请参考[requirements-service.txt](https://github.com/autodeployai/daas-microk8s/blob/master/requirements-service.txt)。下载requirements-service.txt文件到当前目录，创建Dockerfile：

```dockerfile
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

RUN mkdir -p /daas
WORKDIR /daas

COPY requirements-service.txt /daas

RUN pip install -r requirements-service.txt && rm -rf /root/.cache/pip
```

构建Image：

```bash
docker build -f Dockerfile -t pytorch:1.0 .
```

2. 推送Docker镜像到Kubernetes中：

构建好的Docker镜像必须推送到安装DaaS的Kubernetes环境能访问的地方，不同的Kubernetes环境有不同的Docker镜像访问机制，比如本地镜像，私有或者公有镜像注册表（Image Registry）。下面以[Daas-MicroK8s](https://github.com/autodeployai/daas-microk8s)为例，它使用的是MicroK8s本地镜像缓存（Local Images Cache）：

```bash
docker save pytorch:1.0 > pytorch.tar
microk8s ctr image import pytorch.tar
```

3. 创建Pytorch运行时：

登陆DaaS Web页面后，点击顶部菜单`环境 / 运行时定义`，下面页面会列出所有的有效运行时，可以看到DaaS自带的四种运行时：

![DaaS-runtimes](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/daas-runtimes.png)

点击按钮`创建运行时`，创建基于`pytorch:1.0`镜像的Environment运行时:

![DaaS-runtimes](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/daas-create-runtime.png)


## 默认部署Pytorch模型

### 训练Pytorch模型。

使用torchvision中的`MNIST`数据来识别用户输入的数字，以下代码参考官方实例[Image classification (MNIST) using Convnets](https://github.com/pytorch/examples/blob/master/mnist/main.py)：

首先，定义一个无参函数返回用户定义模型类（继承自torch.nn.Module）的一个实例，函数中包含所有的依赖，可以独立运行，也就是说包含引入的第三方库，定义的类、函数或者变量等等。这是能自动部署Pytorch模型的关键。

```python
# Define a function to create an instance of the Net class
def create_net():
    import torch
    import torch.nn as nn  # PyTorch's module wrapper
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output
    return Net()
```

为了快速训练出模型，修改epochs=3

```python
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

use_cuda = torch.cuda.is_available()
batch_size = 64
test_batch_size = 1000
seed = 1234567
lr = 1.0
gamma = 0.7
log_interval = 10
epochs = 3

torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'batch_size': batch_size}
if use_cuda:
    kwargs.update({'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True},
                  )

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

model = create_net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch, log_interval)
    test(model, device, test_loader)
    scheduler.step()
```

### 发布Pytorch模型

模型训练成功后，通过客户端`publish`函数，发布模型到DaaS服务器端。通过设置测试数据集`x_test`和`y_test`，DaaS会自动侦测模型输入数据格式（类型和维数），挖掘模式（分类或者回归），评估模型，并且自动存储`x_test`中的第一行数据作为样例数据，以方便模型测试使用。参数`source_object`指定为上面定义的`create_net`函数，该函数代码会被自动存储到DaaS系统中。

```
batch_idx, (x_test, y_test) = next(enumerate(test_loader))

# Publish the built model into DaaS
publish_resp = client.publish(model,
                              name='pytorch-mnist',
                              x_test=x_test,
                              y_test=y_test,
                              source_object=create_net,
                              description='A Pytorch MNIST classification model')
pprint(publish_resp)
```


结果如下：

```python
{'model_name': 'pytorch-mnist', 'model_version': '1'}
```

### 测试Pytorch模型

调用`test`函数，指定runtime为之前创建的pytorch：
```
test_resp = client.test(publish_resp['model_name'], 
                        model_version=publish_resp['model_version'],
                        runtime='pytorch')
pprint(test_resp)
```

返回值`test_resp`是一个字典类型的结果，记录了测试API信息，如下：

```
The runtime "pytorch" is starting
Waiting for it becomes available... 

{'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOjEwMDAsInVzZXJuYW1lIjoiYWRtaW4iLCJyb2xlIjoiYWRtaW4iLCJleHAiOjE1OTYwNzkyNzksImlhdCI6MTU5NjAzMjQ3OX0.kLO5R-yiTY6xOo14sAxZGwetQqiq5hDfPs5WZ7epSkDWKeDvyLkVP4VzWQxxlPyUX6SgGeCx0pq-of6SYVLPcOmR54a6W7b4ZfKgllKrssdMqaStclv0S2OFHeVXDIoy4cyoB99MjNaXOc6FCbNB4rae0ufu-eZLLYGlHbvV_c3mJtIIBvMZvonU1WCz6KDU2fEyDOt4hXsqzW4k7IvhyDP2geHWrkk0Jqcob8qag4qCYrNHLWRs8RJXBVXJ1Y9Z5PdhP6CGwt5Qtyf017s7L_BQW3_V9Wq-_qv3_TwcWEyCBTQ45RcCLoqzA-dlCbYgd8seurnI3HlYJZPOcrVY5w',
 'endpoint_url': 'https://192.168.64.7/api/v1/test/deployment-test/pytorch/test',
 'payload': {'args': {'X': [{'tensor_input': [[[[...], [...], ...]]]}],
                      'model_name': 'pytorch-mnist',
                      'model_version': '1'}}}
```
`tensor_input`是一个维数为(1, 1, 28, 28)的嵌套数组，以上未列出完整的数据值。

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
{'result': [{'tensor_output': [[-21.444242477416992,
                                -20.39040756225586,
                                -17.134702682495117,
                                -16.960391998291016,
                                -20.394105911254883,
                                -22.380189895629883,
                                -29.211040496826172,
                                -1.311301275563892e-06,
                                -20.16324234008789,
                                -13.592040061950684]]}],
 'stderr': [],
 'stdout': []}
```
测试结果除了预测值，还包括标准输出和标准错误输出的日志信息，方便用户的查看和调试。

验证预测结果，与本地模型预测结果进行比较：

```
import numpy as np

desired = model(x_test[[0]]).detach().numpy()
actual = response.json()['result'][0]['tensor_output']
np.testing.assert_almost_equal(actual, desired)
```

### 正式部署Pytorch模型

测试成功后，可以进行正式的模型部署。与测试API `test` 类似，同样需要指定runtime为之前创建的pytorch。为了提升部署的性能和稳定性，可以为运行环境指定CPU核数、内存大小以及部署副本数，这些都可以通过 `deploy` 函数参数设定。

```
deploy_resp = client.deploy(model_name=publish_resp['model_name'], 
                            deployment_name=publish_resp['model_name'] + '-svc',
                            model_version=publish_resp['model_version'],
                            runtime='pytorch')
pprint(deploy_resp)
```

返回结果：

```
The deployment "pytorch-mnist-svc" created successfully
Waiting for it becomes available... 

{'access_token': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOjEwMDAsInVzZXJuYW1lIjoiYWRtaW4iLCJyb2xlIjoiYWRtaW4iLCJwcm9qZWN0TmFtZSI6Ilx1OTBlOFx1N2Y3Mlx1NmQ0Ylx1OGJkNSIsInByb2plY3RMYWJlbCI6ImRlcGxveW1lbnQtdGVzdCIsImlhdCI6MTU5NjAyODU2N30.iBGyYxCjD5mB_o2IbMkSKRlx9YVvfE3Ih-6LOE-cmp9VoDde-t3JLcDdS3Fg7vyVSIbre6XmYDQ_6IDjzy8XEOzxuxxdhwFPnW8Si1P-fbln5HkPhbDukImShM5ZAcfmD6fNWbz2S0JIgs8rM15d1WKGTC3n9yaXiVumWV1lTKImhl1tBF4ay_6YdCqKmLsrLX6UqbcZA5ZTqHaAG76xgK9vSo1aOOstKLTcloEkswpuMtkYo6ByouLznqQ_yklAYTthdrKX623OJdO3__DOkULq8E-am_c6R7FtyRvYwr4O5BKeHjKCxY6pHmc6PI4Yyyd_TJUTbNPX9fPxhZ4CRg',
 'endpoint_url': 'https://192.168.64.7/api/v1/svc/deployment-test/pytorch-mnist-svc/predict',
 'payload': {'args': {'X': [{'tensor_input': [[[[...],[...],...]]]}]}}}
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
{'result': [{'tensor_output': [[-21.444242477416992,
                                -20.39040756225586,
                                -17.134702682495117,
                                -16.960391998291016,
                                -20.394105911254883,
                                -22.380189895629883,
                                -29.211040496826172,
                                -1.311301275563892e-06,
                                -20.16324234008789,
                                -13.592040061950684]]}]}
```

除了通过DaaS-Client客户端程序，模型测试和模型部署，也可以在DaaS Web客户端完成，这里就不再赘述。


## 自定义部署Pytorch模型

在上面的默认模型部署中，我们看到模型的输入数据是维数为(, 1, 28, 28)的张量（Tensor），输出结果是(, 10)的张量，客户端调用部署REST API时，必须进行数据预处理和结果后处理，包括读取图像文件，转换成需要的张量格式，并且调用和模型训练相同的数据变换，比如上面的归一化操作（Normalize），最后通过张量结果计算出最终识别出的数字。

为了减轻客户端的负担，我们希望这些操作都能在部署服务器端完成，客户端直接输入图像，服务器端直接返回最终的识别数字。在DaaS中，可以通过模型自定义部署功能来满足以上需求，它允许用户自由添加任意的数据预处理和后处理操作，下面我们详细介绍如何自定义部署上面的Pytorch模型。

登陆DaaS Web客户端，查看`pytorch-mnist`模型信息：

![DaaS-model-overview](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/daas-model-overview.png)

切换到`实时预测`标签页，点击命令`生成自定义实时预测脚本`，生成预定义脚本：

![DaaS-model-realtime](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/daas-model-realtime.png)

我们看到函数`create_net`内容会被自动写入到生成的预测脚本中，点击命令`高级设置`，选择网络服务运行环境为`pytorch`：

![DaaS-model-realtime](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/daas-advanced-settings.png)

点击`作为API测试`命令，页面切换到测试页面，修改`preprocess_files`函数，引入模型训练时的图像处理操作：

```
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
        
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        import numpy as np
        from PIL import Image
        for key, file in files.items():
            img = Image.open(file)
            normed = transform(img)
            record[key] = normed.numpy()

    return args
```

完成后，输入函数名`predict`，选择请求正文基于`表单`，输入名称`tensor_input`，选择`文件`，点击上传测试图像`test.png`（该图像为上面测试使用的数据），点击`提交`，右侧响应页面将会显示预测结果：

![DaaS-custom-realtime-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/daas-custom-realtime-test.png)

可以看到，结果与默认部署输出相同。继续修改`postprocess`函数为：

```
def postprocess(result):
    """postprocess the predicted results"""
    import numpy as np
    return [int(np.argmax(np.array(result).squeeze(), axis=0))]
```

重新`提交`，右侧响应页面显示结果为：

![DaaS-custom-realtime-test-final](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/daas-custom-realtime-test-final.png)

测试完成后，可以创建正式的部署，切换到`部署`标签页，点击命令`添加网络服务`，输入服务名称`pytorch-mnist-custom-svc`，网络服务运行环境选择`pytorch`，其他使用默认选项，点击`创建`。进入到部署页面后，点击`测试`标签页，该界面类似之前的脚本测试界面，输入函数名`predict`，请求正文选择基于`表单`，输入名称`tensor_input`，类型选择`文件`，点击上传测试的图片后，点击提交：

![DaaS-deployment-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/daas-deployment-test.png)

到此，正式部署已经测试和创建完成，用户可以使用任意的客户端程序调用该部署服务。点击以上界面中的`生成代码`命令，显示如何通过curl命令调用该服务，测试如下：

![DaaS-deployment-curl-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/daas-deployment-curl-test.png)

## 通过ONNX部署Pytorch模型

除了通过以上的原生部署，Pytorch库本身支持导出ONNX格式，所以通过ONNX来部署Pytorch模型是另一个选择，ONNX部署的优势是模型部署不再需要依赖Pytorch库，也就是不需要创建上面的pytorch运行时。可以使用DaaS默认自带的运行时`Python 3.7 - Function as a Service`，它包含了ONNX Runtime CPU版本用于支持ONNX模型预测。

### 转换Pytorch模型到ONNX：

```python
# Export the model
torch.onnx.export(model,                     # model being run
                  x_test[[0]],               # model input (or a tuple for multiple inputs)
                  'mnist.onnx',              # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['tensor_input'],   # the model's input names
                  output_names = ['tensor_output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}}
                  )
```

### 发布ONNX模型：

```python
publish_resp = client.publish('mnist.onnx',
                              name='pytorch-mnist-onnx',
                              x_test=x_test,
                              y_test=y_test,
                              description='A Pytorch MNIST classification model in ONNX')
pprint(publish_resp)
```

结果如下：

```python
{'model_name': 'pytorch-mnist-onnx', 'model_version': '1'}
```

### 测试ONNX模型

上面，我们通过客户端的`test`函数来进行模型测试，这里我们使用另一个方式，在DaaS Web页面中测试模型。登陆DaaS Web客户端，进入`pytorch-mnist-onnx`模型页面，切换到`测试`标签页，我们看到DaaS自动存储了一条测试数据，点击`提交`命令，测试该条数据，如图：

![DaaS-onnx-model-test](https://raw.githubusercontent.com/aipredict/ai-deployment/master/deploy-pytorch-in-daas/daas-onnx-model-test.png)

我们看到，该ONNX模型和原生Pytorch模型测试结果是一致的。

### 默认部署和自定义部署ONNX模型

关于在DaaS Web界面中如何为ONNX模型创建默认部署和自定义部署，请参考文章《[使用ONNX部署深度学习和传统机器学习模型](https://github.com/aipredict/ai-deployment/blob/master/deploy-ml-dl-using-onnx/README.md)》，流程相同，就不再这里赘述。


## 试用DaaS(Deployment-as-a-Service)
本文中，我们介绍了在DaaS中如何原生部署Pytorch模型，整个流程非常简单，对于默认部署，只是简单调用几个API就可以完成模型的部署，而对于自定义部署，DaaS提供了方便的测试界面，可以随时程序修改脚本进行测试，调试成功后再创建正式部署。在现实的部署中，为了获取更高的预测性能，用户需要更多的修改自定义预测脚本，比如更优的数据处理，使用GPU等。DaaS提供了简单易用的部署框架允许用户自由的定制和扩展。

如果您想体验DaaS模型自动部署系统，或者通过我们的云端SaaS服务，或者本地部署，请发送邮件到 autodeploy.ai#outlook.com（# 替换为 @），并说明一下您的模型部署需求。

## 参考
* DaaS-Client：[https://github.com/autodeployai/daas-client](https://github.com/autodeployai/daas-client)
* AutoDeployAI：[https://www.autodeploy.ai/](https://www.autodeploy.ai/)

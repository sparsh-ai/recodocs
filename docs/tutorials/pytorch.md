---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="3EoZeq-emMnV" -->
# PyTorch Fundamentals
<!-- #endregion -->

<!-- #region id="FEDVRuAGVmOd" -->
## Perceptron
<!-- #endregion -->

<!-- #region id="ciyc50m4Vol6" -->
### Imports
<!-- #endregion -->

```python id="Cl3ky79EVqLT"
import numpy as np
import matplotlib.pyplot as plt
import torch
```

<!-- #region id="AVsRFvFyVtCR" -->
### Simple Perceptron
<!-- #endregion -->

<!-- #region id="DF2RO0hZVx7J" -->
#### Prepare the dataset
<!-- #endregion -->

```python id="Hdu5EA4oV0V4"
data = """0.77,-1.14,0\n-0.33,1.44,0\n0.91,-3.07,0\n-0.37,-1.91,0\n-1.84,-1.13,0\n-1.50,0.34,0\n-0.63,-1.53,0\n-1.08,-1.23,0\n0.39,-1.99,0\n-1.26,-2.90,0\n-5.27,-0.78,0\n-0.49,-2.74,0\n1.48,-3.74,0\n-1.64,-1.96,0\n0.45,0.36,0\n-1.48,-1.17,0\n-2.94,-4.47,0\n-2.19,-1.48,0\n0.02,-0.02,0\n-2.24,-2.12,0\n-3.17,-3.69,0\n-4.09,1.03,0\n-2.41,-2.31,0\n-3.45,-0.61,0\n-3.96,-2.00,0\n-2.95,-1.16,0\n-2.42,-3.35,0\n-1.74,-1.10,0\n-1.61,-1.28,0\n-2.59,-2.21,0\n-2.64,-2.20,0\n-2.84,-4.12,0\n-1.45,-2.26,0\n-3.98,-1.05,0\n-2.97,-1.63,0\n-0.68,-1.52,0\n-0.10,-3.43,0\n-1.14,-2.66,0\n-2.92,-2.51,0\n-2.14,-1.62,0\n-3.33,-0.44,0\n-1.05,-3.85,0\n0.38,0.95,0\n-0.05,-1.95,0\n-3.20,-0.22,0\n-2.26,0.01,0\n-1.41,-0.33,0\n-1.20,-0.71,0\n-1.69,0.80,0\n-1.52,-1.14,0\n3.88,0.65,1\n0.73,2.97,1\n0.83,3.94,1\n1.59,1.25,1\n3.92,3.48,1\n3.87,2.91,1\n1.14,3.91,1\n1.73,2.80,1\n2.95,1.84,1\n2.61,2.92,1\n2.38,0.90,1\n2.30,3.33,1\n1.31,1.85,1\n1.56,3.85,1\n2.67,2.41,1\n1.23,2.54,1\n1.33,2.03,1\n1.36,2.68,1\n2.58,1.79,1\n2.40,0.91,1\n0.51,2.44,1\n2.17,2.64,1\n4.38,2.94,1\n1.09,3.12,1\n0.68,1.54,1\n1.93,3.71,1\n1.26,1.17,1\n1.90,1.34,1\n3.13,0.92,1\n0.85,1.56,1\n1.50,3.93,1\n2.95,2.09,1\n0.77,2.84,1\n1.00,0.46,1\n3.19,2.32,1\n2.92,2.32,1\n2.86,1.35,1\n0.97,2.68,1\n1.20,1.31,1\n1.54,2.02,1\n1.65,0.63,1\n1.36,-0.22,1\n2.63,0.40,1\n0.90,2.05,1\n1.26,3.54,1\n0.71,2.27,1\n1.96,0.83,1\n2.52,1.83,1\n2.77,2.82,1\n4.16,3.34,1"""

with open('perceptron_toydata.txt', 'w') as f:
    f.write(data)
```

```python colab={"base_uri": "https://localhost:8080/"} id="760TA99XWBl8" outputId="702bb920-45c3-4b40-9b8d-22da69efe09a"
data = np.genfromtxt('perceptron_toydata.txt', delimiter=',')
X, y = data[:, :2], data[:, 2]
y = y.astype(np.int)

print('Class label counts:', np.bincount(y))
print('X.shape:', X.shape)
print('y.shape:', y.shape)

# Shuffling & train/test split
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, y = X[shuffle_idx], y[shuffle_idx]

X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]

# Normalize (mean zero, unit variance)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="Qb9e2-LjWQ9R" outputId="46976b92-566a-4d18-ef99-e80348d41f0b"
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend()
plt.show()
```

<!-- #region id="xHGwQwp5W_Nw" -->
#### Defining the Perceptron model
<!-- #endregion -->

```python id="wGyJ983uXCEx"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def custom_where(cond, x_1, x_2):
    return (cond.long() * x_1) + ((1-cond.long()) * x_2)


class Perceptron():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, 
                                   dtype=torch.float32, device=device)
        self.bias = torch.zeros(1, dtype=torch.float32, device=device)

    def forward(self, x):
        linear = torch.add(torch.mm(x, self.weights), self.bias)
        predictions = custom_where(linear > 0., 1, 0).float()
        return predictions
        
    def backward(self, x, y):  
        predictions = self.forward(x)
        errors = y - predictions
        return errors
        
    def train(self, x, y, epochs):
        for e in range(epochs):
            
            for i in range(y.size()[0]):
                # use view because backward expects a matrix (i.e., 2D tensor)
                errors = self.backward(x[i].view(1, self.num_features), y[i]).view(-1)
                self.weights += (errors * x[i]).view(self.num_features, 1)
                self.bias += errors
                
    def evaluate(self, x, y):
        predictions = self.forward(x).view(-1)
        accuracy = torch.sum(predictions == y).float() / y.size()[0]
        return accuracy
```

<!-- #region id="qNtagj4zXMUr" -->
#### Training the Perceptron
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="RgM5T97NXPMk" outputId="ecef04a3-c75a-4184-e97d-37f4cbeb9f3a"
ppn = Perceptron(num_features=2)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

ppn.train(X_train_tensor, y_train_tensor, epochs=5)

print('Model parameters:')
print('  Weights: %s' % ppn.weights)
print('  Bias: %s' % ppn.bias)
```

<!-- #region id="WQjLffD8XQH5" -->
#### Evaluating the model
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="HPBujF2_X152" outputId="3a9bcd77-9e67-4a0a-e556-ff721555bde0"
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

test_acc = ppn.evaluate(X_test_tensor, y_test_tensor)
print('Test set accuracy: %.2f%%' % (test_acc*100))
```

<!-- #region id="Uq9DpvWkX29h" -->
#### 2D Decision Boundary
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 211} id="Rj-WVopfX-yb" outputId="c87d5ba9-968b-4b55-aecc-38fbaa043279"
w, b = ppn.weights, ppn.bias

x_min = -2
y_min = ( (-(w[0] * x_min) - b[0]) 
          / w[1] )

x_max = 2
y_max = ( (-(w[0] * x_max) - b[0]) 
          / w[1] )


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))

ax[0].plot([x_min, x_max], [y_min, y_max])
ax[1].plot([x_min, x_max], [y_min, y_max])

ax[0].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
ax[0].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')

ax[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='class 0', marker='o')
ax[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='class 1', marker='s')

ax[1].legend(loc='upper left')
plt.show()
```

<!-- #region id="fMwkQGCOYAkn" -->
## Deploying PyTorch to Production
<!-- #endregion -->

<!-- #region id="S-g8Pw6Valrb" -->
**torchscript, torchserve, Flask, ONNX**
<!-- #endregion -->

<!-- #region id="AIH22088bZ3x" -->
In the past, going from research to production was a challenging task that required a team of software engineers to move PyTorch models to a framework and integrate them nto a (often non-Python) production environment. Today, PyTorch includes built-in tools and external libraries to support rapid deployment to a variety of production environments.
<!-- #endregion -->

<!-- #region id="f-J0XMNtbaIc" -->
We will focus on deploying your model for inference, not training, and we’ll explore how to deploy your trained PyTorch models into a variety of applications. First, I’ll describe the various built-in capabilities and tools within PyTorch that you can use for deployment. Tools like TorchServe and TorchScript allow you to easily deploy your PyTorch models to the cloud and to mobile or edge devices.

Depending on the application and environment, you may have several options for deployment, each with its own trade-offs. I’ll show you examples of how you can deploy your PyTorch models in multiple cloud and edge environments. You’ll learn how to deploy to web servers for development and production at scale, to iOS and Android mobile devices, and to Internet of Things (IoT) devices based on ARM processors, GPUs, and field-programmable gate array (FPGA) hardware.
<!-- #endregion -->

<!-- #region id="octSI6UkbkWI" -->
Table below summarizes the various resources available for deployment and indicates how to appropriately use each one.
<!-- #endregion -->

<!-- #region id="fy9AjJSbbylc" -->
| Resource | Use |
| -------- | --- |
| Python API | Perform fast prototyping, training, and experimentation; program Python runtimes. |
| TorchScript | Improve performance and portability (e.g., load and run a model in C++); </br>program non-Python runtimes or strict latency and performance requirements. |
| TorchServe | A fast production environment tool with model store, A/B testing, monitoring, and RESTful API. |
| ONNX | Deploy to systems with ONNX runtimes or FPGA devices. |
| Mobile libraries | Deploy to iOS and Android devices. |
<!-- #endregion -->

<!-- #region id="AhZUOMzHcHYN" -->
 For our examples, we’ll deploy an image classifier using a VGG16 model pretrained with ImageNet data. That way, each section can focus on the deployment approach used and not the model itself. For each approach, you can replace the VGG16 model with one of your own and follow the same workflow to achieve results with your own designs.
<!-- #endregion -->

```python id="6rFvzHR9eT5I"
import numpy as np
from torchvision.models import vgg16

model = vgg16(pretrained=True)

model_parameters = filter(lambda p: 
      p.requires_grad, model.parameters())

params = sum([np.prod(p.size()) for 
      p in model_parameters])
print(params)
```

```python id="iHjAwVv0jIdX"
import torch
torch.save(model.state_dict(), "./vgg16_model.pt")
```

<!-- #region id="jQ56otMTe7ni" -->
The VGG16 model has 138,357,544 trainable parameters. As we go through each approach, keep in mind the performance at this level of complexity. You can use this as a rough benchmark when comparing the complexity of your models.

After we instantiate the VGG16 model, it requires minimal effort to deploy it in a Python application.
<!-- #endregion -->

<!-- #region id="Sn3EMcuZe--U" -->
> Tip: Python is not always used in production environments due to its slower performance and lack of true multithreading. If your production environment uses another language (e.g., C++, Java, Rust, or Go), you can convert your models to TorchScript code.
<!-- #endregion -->

<!-- #region id="jdTEXC9xfJXk" -->
TorchScript is a way to serialize and optimize your PyTorch model code so that your PyTorch models can be saved and executed in non-Python runtime environments with no dependency on Python. TorchScript is commonly used to run PyTorch models in C++ and with any language that supports C++ bindings.

TorchScript represents a PyTorch model in a format that can be understood, compiled, and serialized by the TorchScript compiler. The TorchScript compiler creates a serialized, optimized version of your model that can be used in C++ applications. To load your TorchScript model in C++, you would use the PyTorch C++ API library called LibTorch.

There are two ways to convert your PyTorch models to TorchScript. The first one is called tracing, which is a process in which you pass in an example input and perform the conversion with one line of code. It’s used in most cases. The second is called scripting, and it’s used when your model has more complex control code. For example, if your model has conditional if statements that depend on the input itself, you’ll want to use scripting. Let’s take a look at some reference code for each case.

Since our VGG16 example model does not have any control flow, we can use tracing to convert our model to TorchScript, as shown in the following code:
<!-- #endregion -->

```python id="hZXq7ugAfxen"
import torch

model = vgg16(pretrained=True)
example_input = torch.rand(1, 3, 224, 224)
torchscript_model = torch.jit.trace(model,
                            example_input)
torchscript_model.save("traced_vgg16_model.pt")
```

<!-- #region id="ejtVptEegAsH" -->
If our model used control flow, we would need to use the annotation method to convert it to TorchScript. Let’s consider the following model:
<!-- #endregion -->

```python id="Lii7cb_4gQif"
import torch.nn as nn

class ControlFlowModel(nn.Module):
  def __init__(self, N):
    super(ControlFlowModel, self).__init__()
    self.fc = nn.Linear(N,100)

  def forward(self, input):
    if input.sum() > 0:
      output = input
    else:
      output = -input
    return output

model = ControlFlowModel(10)
torchcript_model = torch.jit.script(model)
torchscript_model.save("scripted_vgg16_model.pt")
```

<!-- #region id="FqrTTQnsgTtf" -->
Now we can use our model in a C++ application, as shown in the following C++ code:
<!-- #endregion -->

<!-- #region id="GOyE5f-QgqQG" -->
```cpp
include <torch/script.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app" >> \
      "<path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module model;
  model = torch::jit::load(argv[1]);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( \
      torch::ones({1, 3, 224, 224}));

  at::Tensor output = model.forward(inputs).toTensor();
  std::cout \
    << output.slice(/*dim=*/1, \
        /*start=*/0, /*end=*/5) \
    << '\N';
  }

}
```
<!-- #endregion -->

<!-- #region id="Z1ElKSnKhBYS" -->
In this section, we used TorchScript to increase the performance of our model when it’s used in a C++ application or in a language that binds to C++. However, deploying PyTorch models at scale requires additional capabilities, like packaging models, configuring runtime environments, exposing API endpoints, logging and monitoring, and managing multiple model versions. Fortunately, PyTorch provides a tool called TorchServe to facilitate these tasks and rapidly deploy your models for inference at scale.
<!-- #endregion -->

<!-- #region id="FPLXvE9JhQud" -->
TorchServe is an open-source model-serving framework that makes it easy to deploy trained PyTorch models. It was developed by AWS engineers and jointly released with Facebook in April 2020, and it is actively maintained by AWS. TorchServe supports all the features needed to deploy models to production at scale, including multimodel serving, model versioning for A/B testing, logging and metrics for monitoring, and a RESTful API for integration with other systems.
<!-- #endregion -->

<!-- #region id="Xg7Cq3yuZQi5" -->
![image.png](data:image/png;base64,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)
<!-- #endregion -->

<!-- #region id="jv0S9C7GhfV1" -->
The client application interfaces with TorchServe through multiple APIs. The Inference API provides the main inference requests and predictions. The client application sends input data through the RESTful API request and receives the prediction results. The Management API allows you to register and manage your deployed models. You can register, unregister, set default models, configure A/B testing, check status, and specify the number of workers for a model. The Metrics API allows you to monitor each model’s performance.

TorchServe runs all model instances and captures server logs. It processes the frontend APIs and manages the model storage to disk. TorchServe also provides a number of default handlers for common applications like object detection and text classification. The handlers take care of converting data from the API into a format that your model will process. This helps speed up deployment since you don’t have to write custom code for these common applications.
<!-- #endregion -->

<!-- #region id="jzKrDmpTh_CC" -->
To deploy your models via TorchServe, you will need to follow a few steps. First you need to install TorchServe’s tools. Then you’ll package your model using the model archiver tool. Once your models are archived, you’ll then run the TorchServe web server. Once the web server is running, you can use its APIs to request predictions, manage your models, perform monitoring, or access server logs.
<!-- #endregion -->

```python id="vjp7DZDsiHWJ"
!pip install torchserve torch-model-archiver
!pip install image_classifier captum
```

<!-- #region id="TDErgaNni0hY" -->
TorchServe has the ability to package all model artifacts into a single-model archive file. To do so, we will use the torch-model-archiver command-line tool that we installed in the previous step. It packages model checkpoints as well as the state_dict into a .mar file that the TorchServe server uses to serve the model.

You can use the torch-model-archiver to archive your TorchScript models as well as the standard “eager-mode” implementations, as shown in the following code.
<!-- #endregion -->

```python id="CQ2cD5c_jAGG"
!torch-model-archiver --model-name vgg16 \
  --version 1.0 --serialized-file traced_vgg16_model.pt --handler \
  image_classifier
```

```python id="-pzHqEg_o-r3"
!mkdir -p /content/models && mv ./*.mar /content/models
```

<!-- #region id="7mbTM7uXjyzA" -->
TorchServe includes a built-in web server that is run from the command line. It wraps one or more PyTorch models in a set of REST APIs and provides controls for configuring the port, host, and logging. The following command starts the web server with all models in the model store located in the /models folder:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="B9m6cczWrnSY" outputId="1fc789b9-adb1-4888-b271-7fed9d49e496"
%%writefile config.properties
inference_address=https://0.0.0.0:8091
management_address=https://0.0.0.0:8092
metrics_address=https://0.0.0.0:8093
```

```python colab={"base_uri": "https://localhost:8080/"} id="E1z42ss7n06P" outputId="97f22199-ddc3-4201-c8ef-1cdaecb405e2"
!nohup torchserve --model-store ./models --start --models all --ts-config ./config.properties --ncs --foreground &
```

```python colab={"base_uri": "https://localhost:8080/"} id="LYMkyNgyuemQ" outputId="b690ea2d-27c0-4b2e-a21e-4c917d59b808"
!tail nohup.out
```

```python colab={"base_uri": "https://localhost:8080/"} id="IyFjEuEwvRxX" outputId="03064fa4-231e-4738-f65a-24333c5907d9"
!wget -O hotdog.jpg -q --show-progress https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTk5onR1hxG2h_yGFkgZvLVu7b7IY2PIuekKaagBG0nYFsqktcIwjYu6a7LT6OjTfEHWAU&usqp=CAU
```

```python colab={"base_uri": "https://localhost:8080/"} id="fN9mCqqzwn2r" outputId="10b34086-e242-42e5-a62b-056c2f670bc3"
!curl --insecure https://localhost:8091/ping
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZNW6rFTZuzcA" outputId="216ecda0-3ba1-4a79-de79-73bbfaec87dc"
!curl --insecure https://localhost:8091/predictions/vgg16 -T hotdog.jpg
```

```python colab={"base_uri": "https://localhost:8080/"} id="cM4HAOmJn4_m" outputId="9bde44ef-22f4-4983-8b57-41fe29f60320"
!torchserve --stop
```

<!-- #region id="peYYCsZ0yQn2" -->
You can configure metrics using the Metrics API and monitor and log your models’ performance when deployed.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ysiXL3KT0QOl" outputId="e6f40f54-71d6-4166-8cef-14ed10affff9"
!curl --insecure https://127.0.0.1:8093/metrics
```

<!-- #region id="O5fAtWKT0UwD" -->
The default metrics endpoint returns Prometheus-formatted metrics. Prometheus is a free software application used for event monitoring and alerting that records real-time metrics in a time series database built using an HTTP pull model. You can query metrics using curl requests or point a Prometheus Server to the endpoint and use Grafana for dashboards.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="oLK9D-Bs060e" outputId="13f073e0-ab7e-4bf3-d3c3-f427ff7b36eb"
!ls -al ./logs
```

<!-- #region id="DvMWsigQ00Jg" -->
Metrics are logged to a file. TorchServe also supports other types of server logging, including access logs and TorchServe logs. Access logs record the inference requests and the time it takes to complete the requests. As defined in the properties file, the access logs are collected in the <log_location>/access_log.log file. TorchServe logs collect all the logs from TorchServe and its backend workers.

TorchServe supports capabilities beyond the default settings for metrics and logging. Metrics and logging can be configured in many different ways. In addition, you can create custom logs. For more information on metric and logging customization and other advanced features of TorchServe, refer to the TorchServe documentation.
<!-- #endregion -->

<!-- #region id="b4VuOcQI08YV" -->
The NVIDIA Triton Inference Server is becoming more popular and is also used to deploy AI models at scale in production. Although not part of the PyTorch project, you may want to consider the Triton Inference Server as an alternative to TorchServe, especially when deploying to NVIDIA GPUs.

The Triton Inference Server is open source software and can load models from local storage, GCP, or AWS S3. Triton supports running multiple models on single or multiple GPUs, low latency and shared memory, and model ensembles. Some possible advantages of Triton over TorchServe include:

- Triton is out of beta.
- It is the fastest way to infer on NVIDIA hardware (common).
- It can use int4 quantization.
- You can port directly from PyTorch without ONNX.

Available as a Docker container, Triton Inference Server also integrates with Kubernetes for orchestration, metrics, and auto-scaling. For more information, visit the NVIDIA Triton Inference Server documentation.
<!-- #endregion -->

<!-- #region id="vsyxzX1m1Mvf" -->
If your platform doesn’t support PyTorch and you cannot use TorchScript/C++ or TorchServe for your deployment, it may be possible that your deployment platform supports the Open Neural Network Exchange (ONNX) format. The ONNX format defines a common set of operators and a common file format so that deep learning engineers can use models across a variety of frameworks, tools, runtimes, and compilers.

ONNX was developed by Facebook and Microsoft to allow model interoperability between PyTorch and other frameworks, such as Caffe2 and Microsoft Cognitive Toolkit (CTK). ONNX is currently supported by inference runtimes from a number of providers, including Cadence Systems, Habana, Intel AI, NVIDIA, Qualcomm, Tencent, Windows, and Xilinx.
<!-- #endregion -->

<!-- #region id="gdSZDxYV3VM1" -->
An example use case is edge deployment on a Xilinx FPGA device. FPGA devices are custom chips that can be programmed with specific logic. They are used by edge devices for low-latency or high-performance applications, like video. If you want to deploy your new innovative model to an FPGA device, you would first convert it to ONNX format and then use the Xilinx FPGA development tools to generate an FPGA image with your model’s implementation.

Let’s take a look at an example of how to export a model to ONNX, again using our VGG16 model. The ONNX exporter can use tracing or scripting. We learned about tracing and scripting, described in the earlier section on TorchScript. We can use tracing by simply providing the model and an example input. The following code shows how we’d export our VGG16 model to ONNX using tracing:
<!-- #endregion -->

```python id="cXNIAeV53gCP"
!pip install onnx
```

```python id="zqG53qe83WnT"
import onnx

model = vgg16(pretrained=True)
example_input = torch.rand(1, 3, 224, 224)
onnx_model = torch.onnx.export(model, 
                               example_input, 
                               "vgg16.onnx")
```

<!-- #region id="liFkPfnE3sEM" -->
We define an example input and call torch.onnx.export(). The resulting file, vgg16.onnx, is a binary protobuf file that contains both the network structure and the parameters of the VGG16 model we exported.

If we want to verify that our model was converted to ONNX properly, we can use the ONNX checker, as shown in the following code:
<!-- #endregion -->

```python id="gwZPQCjo3sV9"
model = onnx.load("vgg16.onnx")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)
```

<!-- #region id="WOp5RRno3twm" -->
This code uses the Python ONNX library to load the model, run the checker, and print out a human-readable version of the model.
<!-- #endregion -->

<!-- #region id="s9mdnqB13y4K" -->
Before deploying to full-scale production, you may want to deploy your models to a development web server. This enables you to integrate your deep learning algorithms with other systems and quickly build prototypes to demonstrate your new models. One of the easiest ways to build a development server is with Python using Flask.

Flask is a simple micro web framework written in Python. It is called a “micro” framework because it does not include a database abstraction layer, form validation, upload handling, various authentication technologies, or anything else that might be provided with other libraries. We won’t cover Flask in depth in this book, but I’ll show you how to use Flask to deploy your models in Python.

We’ll also expose a REST API so that other applications can pass in data and receive predictions. In the following examples, we’ll deploy our pretrained VGG16 model and classify images. First we’ll define our API endpoints, request types, and response types. Our API endpoint will be at /predict, which takes in POST requests (including the image file). The response will be in JSON format and contain a class_id and class_name from the ImageNet dataset.
<!-- #endregion -->

<!-- #region id="1xT7m0L04V5S" -->
Since our model will return a number indicating the class, we’ll need a lookup table to convert this number to a class name. We create a dictionary called imagenet_class_index by reading in the JSON conversion file. We then instantiate our VGG16 model and define our image transforms to preprocess a PIL image by resizing it, center-cropping it, converting it to a tensor, and normalizing it. These steps are required prior to sending the image into our model.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DvXu2jFJ4PEo" outputId="4ce7dc4c-bf15-4d78-864a-203669fa55f5"
!wget -q --show-progress "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
```

```python id="yEt2UCvW82ge"
import socket
print(socket.gethostbyname(socket.getfqdn(socket.gethostname())))

import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

imagenet_class_index = json.load(
    open("./imagenet_class_index.json"))

model = models.vgg16(pretrained=True)

image_transforms = transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(
          [0.485, 0.456, 0.406],
          [0.229, 0.224, 0.225])])

def get_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    tensor = image_transforms(image)
    outputs = model(tensor.unsqueeze(0))
    _, y = outputs.max(1)
    predicted_idx = str(y.item())
    return imagenet_class_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
  img_bytes = file.read()
  class_id, class_name = \
    get_prediction(image_bytes=img_bytes)
  return jsonify({'class_id': class_id,
                 'class_name': class_name})

import threading
threading.Thread(target=app.run, kwargs={'host':'0.0.0.0','port':5062}).start() 
```

```python colab={"base_uri": "https://localhost:8080/"} id="U-SB5eog5AXy" outputId="9ca0e503-c513-42f1-eaa9-3c0f8561f93d"
import requests

resp = requests.post(
    "http://localhost:5062/predict",
    files={"file": open('hotdog.jpg','rb')})

print(resp.json())
```

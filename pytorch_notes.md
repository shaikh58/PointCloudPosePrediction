### Misc. notes
- import torch
- For experiments, ensure random seed is fixed to reproduce results - torch.manual_seed(1729) 
- GPU support - torch.cuda.is_available() tells you if a GPU is available
- Move CPU defined data to GPU - a.to(torch.device('cuda'))
- Pytorch automatically tracks gradients for variables; at inference time, disable this by using with torch.no_grad() or @torch.no_grad

### Tensors:
- Uses tensor data structure for all operations - torch.tensor(some_list), torch.from_numpy(array)
- Supports numpy-like functions such as zeros, and can convert from numpy arrays - torch.rand(), torch.ones(), torch.zeros()
- Support slicing, broadcasting, concatenation - 
torch.cat([a,b]), a[:,1], torch.ones((2,1))*torch.ones((1,2))
- Supports reshaping (uses x.view()), transposing e.g. a.transpose(3,1,2) specifies the order of the dimensions in the result
- Unsqueeze - e.g. is shape of array is (3,226,226) and we want (1,3,226,226), use a.unsqueeze(0)

### Optimizer: torch.optim
- Ensure requires_grad is set to True for each optimization variable e.g. torch.tensor([1.], requires_grad=True)
- SGD: torch.optim.SGD(model.parameters(), learning_rate)
- Reset gradients to zero at every backprop step - optim.zero_grad() - prevents gradient accumulation 
- Backprop and gradient descent step - loss.backward(), optim.step()

### NN module: torch.nn
- Common layer types are provided e.g. torch.nn.Linear, Conv2D, RNN etc. All layers inherit from nn.Module
- Loss types e.g. loss = nn.CrossEntropyLoss()
- These keep track of parameters, gradients and have methods for forward and backward pass
- Can define extra computations in the forward pass method of a custom Model class e.g. frozen layers
- To freeze a layer named linear1 (i.e. no gradient updates, only transformations), use: 
````self.linear1.requires_grad_(False)````

Defining a custom model (need to define the forward pass):

    import torch.nn as nn

    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)
            self.conv_stack = nn.Sequential(self.conv1, nn.ReLU() self.conv2, nn.ReLU())

        def forward(self, x):
            return self.conv_stack(x)

    model = CustomModel()
    # model is a callable
    output = model(input)

Creating a model quickly without forward pass definition:

    model = nn.Sequential(nn.Conv2D(),nn.ReLU())
    device = torch.device("cuda")
    # move everything to GPU
    data = data.to(device)
    model.to(device)
    # can run on multiple GPUs in parallel
    model = nn.DataParallel(model)


### Datasets: torch.utils.data
- Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the sample
- Define a dataset with all the required transformations and pre-processing; for CV tasks, use torchvision.transforms
- Use inbuilt DataLoader to customize the loading process with automatic batching, shuffling etc.

Creating our custom dataset:

    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms

    class RandomDataset(Dataset):
        def __init__(self, size, length, transform=None, target_transform=None):
            self.len = length
            self.data = torch.randn(length, size)
            # can define custom data preprocessing steps e.g. normalization
            self.transform = transform
            self.target_transform = target_transform

        # recall: this method overloads the square bracket operator (__setitem__ is setter)
        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len

    # create custom transformations pipeline and pass into the dataset creator
    my_transform = transforms.Compose([
        transforms.toTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    my_loader = DataLoader(dataset=RandomDataset(5,100, my_transform), batch_size=30, shuffle=True, num_workers=1)

    # use the data loader to load batches during training loop
    for i, data in enumerate(my_loader):
        inputs, labels = data
        # rest of training loop

### Training loop
- Gets a batch of data from DataLoader, zeros gradients, does a forward pass, calculates loss, backpropagates gradients, performs gradient descent step, reports loss/validation accuracy
- See https://pytorch.org/tutorials/beginner/introyt/trainingyt.html for full code example
- Note: Use model.eval() to set to evaluation mode, disabling dropout and using population statistics for batch normalization
- Also use with torch.no_grad() to speed up inference for validation accuracy during the training loop

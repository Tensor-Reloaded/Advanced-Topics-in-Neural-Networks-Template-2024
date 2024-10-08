import os
from multiprocessing import freeze_support
from typing import Optional

import torch
import torch.nn.functional as F
from timed_decorator.simple_timed import timed
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from tqdm import tqdm


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.conv3(F.relu(self.bn3(out), inplace=True))
        out += shortcut
        return out


class PreActResNet_C10(nn.Module):
    """Pre-activation ResNet for CIFAR-10"""

    def __init__(self, block, num_blocks, num_classes):
        super(PreActResNet_C10, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18_C10(num_classes):
    return PreActResNet_C10(PreActBlock, [2, 2, 2, 2], num_classes)


def get_model():
    return PreActResNet18_C10(10)


class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset, runtime_transforms: Optional[v2.Transform], cache: bool):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset
        self.runtime_transforms = runtime_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        if self.runtime_transforms is None:
            return image, label
        return self.runtime_transforms(image), label


def get_dataset(data_path: str, is_train: bool):
    initial_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.491, 0.482, 0.446),
            std=(0.247, 0.243, 0.261)
        ),
    ])
    cifar10 = CIFAR10(root=data_path, train=is_train, transform=initial_transforms, download=True)
    runtime_transforms = None
    if is_train:
        runtime_transforms = v2.Compose([
            v2.RandomCrop(size=32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomErasing()
        ])
    return CachedDataset(cifar10, runtime_transforms, True)


@torch.jit.script
def accuracy(output: Tensor, labels: Tensor):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements


def train(model, train_loader, criterion, optimizer, device):
    model.train()

    all_outputs = []
    all_labels = []

    for data, labels in train_loader:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(data)
        loss = criterion(output, labels)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        output = output.softmax(dim=1).detach().cpu().squeeze()
        labels = labels.cpu().squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4)


@torch.inference_mode()
def val(model, val_loader, device):
    model.eval()

    all_outputs = []
    all_labels = []

    for data, labels in val_loader:
        data = data.to(device, non_blocking=True)
        output = model(data)

        output = output.softmax(dim=1).cpu().squeeze()
        labels = labels.squeeze()
        all_outputs.append(output)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4)


def do_epoch(model, train_loader, val_loader, criterion, optimizer, device):
    acc = train(model, train_loader, criterion, optimizer, device)
    acc_val = val(model, val_loader, device)
    # torch.cuda.empty_cache()
    return acc, acc_val


def main(device: torch.device = get_default_device(), data_path: str = './data',
         checkpoint_path: str = "./checkpoints"):
    print(f"Using {device}")
    os.makedirs(checkpoint_path, exist_ok=True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset = get_dataset(data_path, is_train=True)
    val_dataset = get_dataset(data_path, is_train=False)

    model = get_model()
    model = model.to(device)
    model = torch.jit.script(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.00001,
                                fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10,
                                                           threshold=0.001, threshold_mode='rel')
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 50
    val_batch_size = 500
    num_workers = 0
    persistent_workers = (num_workers != 0) and False
    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=(device.type == 'cuda'), num_workers=num_workers,
                              batch_size=batch_size, drop_last=True, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    epochs = tuple(range(200))
    best_val = 0.0
    with tqdm(epochs) as tbar:
        for _ in tbar:
            acc, acc_val = do_epoch(model, train_loader, val_loader, criterion, optimizer, device)
            scheduler.step(acc)

            if acc_val > best_val:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, "best.pth"))
                best_val = acc_val
            tbar.set_description(f"Acc: {acc}, Acc_val: {acc_val}, Best_val: {best_val}")


@timed(stdout=False, return_time=True)
def infer(model, val_loader, device, tta, dtype, inference_mode):
    model.eval()
    all_outputs = []
    all_labels = []

    inference_mode = torch.inference_mode if inference_mode else torch.no_grad

    enable_autocast = device.type != 'cpu' and dtype != torch.float32
    # Autocast is slow for cpu, so we disable it.
    # Also, if the device type is mps, autocast might not work (?) and disabling it might also not work (?)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enable_autocast), inference_mode():
        for data, labels in val_loader:
            data = data.to(device, non_blocking=True)

            output = model(data)
            if tta:
                # Horizontal rotation:
                output += model(v2.functional.hflip(data))
                # Vertical rotation:
                output += model(v2.functional.vflip(data))
                # Horizontal rotation + Vertical rotation:
                output += model(v2.functional.hflip(v2.functional.vflip(data)))

            output = output.softmax(dim=1).cpu().squeeze()
            labels = labels.squeeze()
            all_outputs.append(output)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels)

    return round(accuracy(all_outputs, all_labels), 4)


def create_model(device: torch.device, checkpoint_path: str, model_type: str):
    model = get_model()
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, "best.pth"), map_location=device, weights_only=True))
    model.eval()

    if model_type == 'raw model':
        return model
    if model_type == 'scripted model':
        return torch.jit.script(model)
    if model_type == 'traced model':
        return torch.jit.trace(model, torch.rand((5, 3, 32, 32), device=device))
    if model_type == 'frozen model':
        return torch.jit.freeze(torch.jit.script(model))
    if model_type == 'optimized for inference':
        return torch.jit.optimize_for_inference(torch.jit.script(model))
    if model_type == 'compiled model':
        if os.name == 'nt':
            print("torch.compile is not supported on Windows. Try Linux or WSL instead.")
            return model
        return torch.compile(model)


def predict(device: torch.device = get_default_device(), data_path: str = './data',
            checkpoint_path: str = "./checkpoints"):
    val_dataset = get_dataset(data_path, is_train=False)

    val_batch_size = 500

    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, num_workers=0, batch_size=val_batch_size,
                            drop_last=False)

    use_tta = (False, True)
    dtypes = (torch.bfloat16, torch.half, torch.float32)
    model_types = (
        'raw model', 'scripted model', 'traced model', 'frozen model', 'optimized for inference', 'compiled model')

    for tta in use_tta:
        for dtype in dtypes:
            for model_type in model_types:
                if model_type == 'optimized for inference' and dtype != torch.float32:
                    print(f'Model type {model_type} does not work on {dtype}')
                    continue
                inference_mode = True
                if model_type == 'compiled model':
                    # torch.compile may have problems with torch.inference mode
                    inference_mode = False
                try:
                    model = create_model(device, checkpoint_path, model_type)
                    acc_val, elapsed = infer(
                        model, val_loader, device, tta=tta, dtype=dtype, inference_mode=inference_mode)

                    print(f"Device {device.type}, val acc: {acc_val}, tta: {tta}, dtype: {dtype}, "
                          f"model type: {model_type}, took: {elapsed / 1e9}s")
                except Exception as _:
                    # Debug only
                    # import traceback
                    # traceback.print_exc()
                    # print()

                    print(f"Model type {model_type} failed on {dtype} on {device.type}")
            print()


if __name__ == "__main__":
    freeze_support()
    main()
    predict()

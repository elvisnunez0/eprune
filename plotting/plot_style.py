import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-bright")
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = "16"

LINE_WIDTH = 2

MPL_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
TR_FISHER_COLOR = MPL_COLOR_CYCLE[2]
PT_METRIC_COLOR = MPL_COLOR_CYCLE[4]

ARCH_NAME_MAP = {
    "resnet18": "ResNet18",
    "resnet20": "ResNet20",
    "resnet32": "ResNet32",
    "resnet44": "ResNet44",
    "resnet56": "ResNet56",
    "resnet110": "ResNet110",
}

LR_NAME_MAP = {
    "exp": "Exponential",
    "cosine": "Cosine",
    "linear": "Linear",
    "fixed": "Fixed",
}

DATASET_NAME_MAP = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}

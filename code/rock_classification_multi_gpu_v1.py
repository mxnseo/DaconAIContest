
import os
import glob
import random
import cv2
import numpy as np
import timm
from collections import defaultdict
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
from liveplot import LivePlot
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings(action='ignore') # 경고 메시지 무시

CFG = {
    'IMG_SIZE': 224, # model = resnet101의 input size=224
    'EPOCHS': 50,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 96, # multi gpu(DDP) 사용해서 배치 사이즈 *3 (실제로는 한 보드 당 32 부담)
    'SEED': 41
}

def seed_everything(seed): # 시드 고정
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def encode_labels(paths): # 이미지 경로에서 클래스 이름 추출 후 라벨 인코딩
    """pandas의 DataFrame 이후 LabelEncoder에서 버전 충돌이 많이 일어나서 수정"""
    label_map = {}
    labels = []
    for p in paths:
        label_name = os.path.basename(os.path.dirname(p))
        if label_name not in label_map:
            label_map[label_name] = len(label_map)
        labels.append(label_map[label_name])
    return labels, label_map

def stratified_split(paths, labels, test_ratio=0.3, seed=41): # 클래스 비율 유지한 채 train/val 분할
    """기존 baseline에서 train_test_split 함수를 sklearn 없이 수정"""
    label_to_indices = defaultdict(list)
    for i, lbl in enumerate(labels):
        label_to_indices[lbl].append(i)
    train_idx, val_idx = [], []
    random.seed(seed)
    for indices in label_to_indices.values():
        random.shuffle(indices)
        split = int(len(indices) * (1 - test_ratio))
        train_idx.extend(indices[:split])
        val_idx.extend(indices[split:])
    return [paths[i] for i in train_idx], [labels[i] for i in train_idx], [paths[i] for i in val_idx], [labels[i] for i in val_idx]

class PadSquare(ImageOnlyTransform):
    def __init__(self, value=0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.value = value

    def apply(self, image, **params):
        h, w, _ = image.shape
        max_dim = max(h, w)
        top = (max_dim - h) // 2
        bottom = max_dim - h - top
        left = (max_dim - w) // 2
        right = max_dim - w - left
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.value)

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels=None, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        if self.labels is not None:
            return image, self.labels[idx]
        return image

    def __len__(self):
        return len(self.img_paths)

# 증식 추가
train_transform = A.Compose([
    A.RandomResizedCrop(CFG['IMG_SIZE'], CFG['IMG_SIZE'], scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.3),  # 확률적으로 crop, 이미지를 임의의 위치에서 자르고 다시 224x224로 resize
    PadSquare(value=(0, 0, 0)), # 전체 이미지 중 80%~100% 크기에서 crop, 가로:세로 비율도 랜덤
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.4, p=1.0), # 명암 대비, 밝기 대비
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3), p=1.0), # 이미지 선명하게
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0) # 어두운 영역 대비 증가 (적응적 히스토그램 평활화)
    ], p=0.9),  # 위 효과 중 하나를 90% 확률로 적용
    A.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3),
    ToTensorV2()
])


test_transform= A.Compose([
    PadSquare(value=(0, 0, 0)),
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.5,)*3, std=(0.5,)*3),
    ToTensorV2()
])


def calculate_metrics_per_class(true_labels, pred_labels, num_classes):
    """
    클래스별 정밀도/재현율/F1-score 계산
    """
    metrics = []
    for cls in range(num_classes):
        tp = sum((p == cls and t == cls) for p, t in zip(pred_labels, true_labels))
        fp = sum((p == cls and t != cls) for p, t in zip(pred_labels, true_labels))
        fn = sum((p != cls and t == cls) for p, t in zip(pred_labels, true_labels))
        tn = sum((p != cls and t != cls) for p, t in zip(pred_labels, true_labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append({
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1
        })

    return metrics

def display_classwise_metrics(metrics, label_map):
    """
    클래스별 성능 지표 출력
    """
    print("\nClass-wise Metrics:")
    print("Class\t\tPrecision\tRecall\t\tAccuracy\tF1")
    print("-" * 60)
    for cls_id, metric in enumerate(metrics):
        class_name = label_map.get(cls_id, str(cls_id))
        print(f"{class_name: <10}\t{metric['precision']:.4f}\t\t{metric['recall']:.4f}\t\t{metric['accuracy']:.4f}\t\t{metric['f1']:.4f}")
    print("-" * 60)

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, preds, targets = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validating", disable=(dist.get_rank()!=0)):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            val_loss.append(criterion(output, labels).item())
            preds += output.argmax(1).cpu().tolist()
            targets += labels.cpu().tolist()
    val_f1 = f1_score(targets, preds, average='macro')
    return np.mean(val_loss), val_f1, preds, targets

def inference(model, test_paths, transform, device, label_map):
    model.eval()
    dataset = CustomDataset(test_paths, None, transform)
    loader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False)
    preds = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Testing"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds += outputs.argmax(1).cpu().tolist()
    inv_map = {v: k for k, v in label_map.items()}
    return [inv_map[p] for p in preds]


def train(local_rank, world_size):
    
    # 분산 환경 초기화 및 GPU 설정
    seed_everything(CFG['SEED'])
    dist.init_process_group(backend='gloo')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # 학습 데이터 로딩 및 라벨 인코딩
    all_img_paths = glob.glob('/home/jetson/work/dacon/open/train/*/*')
    all_labels, label_map = encode_labels(all_img_paths)
    train_paths, train_labels, val_paths, val_labels = stratified_split(all_img_paths, all_labels, test_ratio=0.3)

    
    # dataset 및 dataloader 정의
    train_dataset = CustomDataset(train_paths, train_labels, train_transform)
    val_dataset = CustomDataset(val_paths, val_labels, test_transform)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4)

    # 모델 로딩 및 DDP 래핑
    model = timm.create_model('resnet101', pretrained=True, num_classes=7).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # 옵티마이저, 스케쥴러 정의
    optimizer = optim.Adam(model.parameters(), lr=CFG["LEARNING_RATE"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5)
    
    
    # 클래스 불균형 보정 가중치
    class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels), y=train_labels) # 클래스 수가 적을 수록 가중치 비율 높게 줌
    
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # master 보드 (rank==0)에서 liveplot 정의 후 학습 시각화
    lp = LivePlot() if dist.get_rank() == 0 else None
    best_score = 0

    for epoch in range(1, CFG['EPOCHS'] + 1): # 에폭, 학습 시작
        model.train()
        train_sampler.set_epoch(epoch) # epoch마다 셔플
        train_loss, train_preds, train_targets = [], [], []

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}", disable=(dist.get_rank()!=0)):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_preds += output.argmax(1).detach().cpu().tolist()
            train_targets += labels.detach().cpu().tolist()

        val_loss, val_f1, val_preds, val_targets = validation(model, criterion, val_loader, device)
        train_f1 = f1_score(train_targets, train_preds, average='macro')

        if dist.get_rank() == 0: # 모델 추론
            os.makedirs('model', exist_ok=True)
            lp.update_all(epoch, np.mean(train_loss), val_loss, train_f1, val_f1)
            print(f"[Epoch {epoch}] Train Loss: {np.mean(train_loss):.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            torch.save(model.module.state_dict(), f"model/model_epoch_{epoch}.pth")
            scheduler.step(val_f1)

            metrics = calculate_metrics_per_class(val_targets, val_preds, num_classes=7)
            display_classwise_metrics(metrics, label_map)

            if val_f1 > best_score:
                best_score = val_f1
                torch.save(model.module.state_dict(), "best_model.pth")
                print(f"Best model updated at epoch {epoch}")

    if dist.get_rank() == 0:
        lp.show()
        print("Running inference on test set...")
        test_img_dir = '/home/jetson/work/dacon/open/test'
        test_paths = sorted(glob.glob(os.path.join(test_img_dir, '*')))
        predictions = inference(model.module, test_paths, test_transform, device, label_map)

        with open('final.csv', 'w') as f:
            f.write('ID,rock_type\n')
            for path, pred in zip(test_paths, predictions):
                fname = os.path.splitext(os.path.basename(path))[0]
                f.write(f"{fname},{pred}\n")
        print("✅ final.csv 저장 완료!")

    dist.destroy_process_group()


# torchrhn으로 실행될 때 main 실행 -> 분산 학습
if __name__ == '__main__':
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    train(local_rank, world_size)

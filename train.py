import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.optim.optimizer import Optimizer
import torchvision.utils as vutils
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib
import gc 

# configuration
N_FOLDS = 5
N_EPOCHS = 10
SEED = 42
TARGET_COL = 'target'
BATCH_SIZE = 32
IMG_SIZE = 640
LR = 1e-4
MAX_LR = 5e-4
PRECISION = 16
NUM_WORKERS = 12
GPUS = 1

# set seed
def set_seed(seed):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    seed_everything(seed)
    return random_state
random_state = set_seed(SEED)

# set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU not available, going to use CPU instead.")

# load train and test data
train = pd.read_csv('./seti-breakthrough-listen/train_labels.csv')
test = pd.read_csv('./seti-breakthrough-listen/sample_submission.csv')
def get_train_file_path(image_id):
    return "./seti-breakthrough-listen/train/{}/{}.npy".format(image_id[0], image_id)
def get_test_file_path(image_id):
    return "./seti-breakthrough-listen/test/{}/{}.npy".format(image_id[0], image_id)
train['file_path'] = train['id'].apply(get_train_file_path)
test['file_path'] = test['id'].apply(get_test_file_path)

# class Dataset
class TrainDataset(Dataset):
    def __init__(self, df, test=False, transform=None):
        self.df = df
        self.test = test
        self.file_names = df['file_path'].values
        if not self.test:
            self.labels = df[TARGET_COL].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = np.load(file_path)[[0, 2, 4]] 
        image = image.astype(np.float32)
        image = np.vstack(image).T 
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = image[np.newaxis,:,:]
            image = torch.from_numpy(image).float()
        if not self.test:
            label = torch.unsqueeze(torch.tensor(self.labels[idx]).float(),-1)
            return image, label
        else:
            return image

# data transfroms function
def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
            A.Resize(IMG_SIZE,IMG_SIZE),
            A.VerticalFlip(p=0.5), # data augmentation
            A.HorizontalFlip(p=0.5), # data augmentation
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            ToTensorV2(),
        ])
        
# class DataModule
class DataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df,  batch_size = BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def setup(self, stage=None):
        
        self.train_dataset = TrainDataset(
          self.train_df,
          transform=get_transforms(data='train')
        )
        
        self.val_dataset = TrainDataset(          
          self.val_df,
          transform=get_transforms(data='valid')
        )
        
        self.test_dataset = TrainDataset(
          self.test_df,
          transform=get_transforms(data='valid'),
          test = True
        )

    def train_dataloader(self):
        return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=NUM_WORKERS,
          drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
          self.val_dataset,
          batch_size=self.batch_size,
          num_workers=NUM_WORKERS,
          drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=NUM_WORKERS,
          drop_last=True,
        )
        
# Mixup
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# MADGRAD
import math
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any

class MADGRAD(Optimizer):

    def __init__(
        self, params: _params_t, lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0, eps: float = 1e-6,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps must be non-negative")

        defaults = dict(lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return False

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:

        loss = None
        if closure is not None:
            loss = closure()

        if 'k' not in self.state:
            self.state['k'] = torch.tensor([0], dtype=torch.long)
        k = self.state['k'].item()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"] + eps
            decay = group["weight_decay"]
            momentum = group["momentum"]

            ck = 1 - momentum
            lamb = lr * math.pow(k + 1, 0.5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                    state["s"] = torch.zeros_like(p.data).detach()
                    if momentum != 0:
                        state["x0"] = torch.clone(p.data).detach()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError("momentum != 0 is not compatible with sparse gradients")

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                # Apply weight decay
                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")

                    grad.add_(p.data, alpha=decay)

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_val = grad._values()

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    # Compute x_0 from other known quantities
                    rms_masked_vals = grad_sum_sq_masked._values().pow(1 / 3).add_(eps)
                    x0_masked_vals = p_masked._values().addcdiv(s_masked._values(), rms_masked_vals, value=1)

                    # Dense + sparse op
                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    rms_masked_vals = grad_sum_sq_masked._values().pow_(1 / 3).add_(eps)

                    s.add_(grad, alpha=lamb)
                    s_masked._values().add_(grad_val, alpha=lamb)

                    # update masked copy of p
                    p_kp1_masked_vals = x0_masked_vals.addcdiv(s_masked._values(), rms_masked_vals, value=-1)
                    # Copy updated masked p to dense p using an add operation
                    p_masked._values().add_(p_kp1_masked_vals, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments
                    grad_sum_sq.addcmul_(grad, grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    # Update s
                    s.data.add_(grad, alpha=lamb)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)


        self.state['k'] += 1
        return loss
    
# Model
class Predictor(pl.LightningModule):

    def __init__(self,
                 MODEL: str,
                 n_classes: int, 
                 n_training_steps=None, 
                 steps_per_epoch=None):
        super().__init__()
        self.n_classes = n_classes
        self.model = timm.create_model(MODEL, pretrained=True, in_chans=1)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, n_classes)
        self.n_training_steps = n_training_steps
        self.steps_per_epoch = steps_per_epoch

    def forward(self, x):
        output = self.model(x)
        return output


    def training_step(self, batch, batch_idx):
        gc.collect()

        x, y = batch
        
        x, y_a, y_b, lam = mixup_data(x, y.view(-1, 1))
        
        output = self(x)
        labels = y
        loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        try:
            auc=roc_auc_score(labels.detach().cpu(), output.sigmoid().detach().cpu())        
            self.log("auc", auc, prog_bar=True, logger=True)
        except:
            pass
        return {"loss": loss, "predictions": output, "labels": labels}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        labels = y
        loss = criterion(output, y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return {"predictions": output, "labels": labels}
      

    def validation_epoch_end(self, outputs):

        preds = []
        labels = []
        
        for output in outputs:
          
          preds += output['predictions']
          labels += output['labels']

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        val_auc=roc_auc_score(labels.detach().cpu(), preds.sigmoid().detach().cpu())
        self.log("val_auc", val_auc, prog_bar=True, logger=True)
        

    def test_step(self, batch, batch_idx):
        x = batch        
        output = self(x).sigmoid()
        return output   

    def configure_optimizers(self):

        optimizer = MADGRAD(self.parameters(), lr=LR)

        scheduler = OneCycleLR(
          optimizer,
          epochs = N_EPOCHS,
          max_lr = MAX_LR,
          total_steps = self.n_training_steps,
          steps_per_epoch = self.steps_per_epoch
        )

        return dict(
          optimizer=optimizer,
          lr_scheduler=scheduler
        )

# model training config 
t_steps_per_epoch=(len(train)// N_EPOCHS) // BATCH_SIZE
total_training_steps = t_steps_per_epoch * N_EPOCHS
criterion=nn.BCEWithLogitsLoss()

def train_teacher_model(train,fold_id):
    '''
    input: train,fold_id
    output: trained_teacher_model
    '''
    teacher_model = Predictor(steps_per_epoch=t_steps_per_epoch,
                              n_training_steps=total_training_steps ,
                              n_classes=1,MODEL='efficientnet_b0')
    data_module = DataModule(
            train[train['fold']!=fold_id], # train fold
            train[train['fold']==fold_id], # val fold
            train[train['fold']==fold_id], # test data, same as val for now
            batch_size=BATCH_SIZE,
            )
    early_stopping_callback = EarlyStopping(
            monitor='val_auc',
            mode="max", 
            patience=3)
    checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-teacher-model-checkpoint-ep{epoch:02d}-{val_auc:.3f}",
            save_top_k = 1,
            verbose = True,
            monitor = "val_auc",
            mode="max",
            )
    trainer = pl.Trainer(
            checkpoint_callback= True,
            callbacks=[early_stopping_callback,checkpoint_callback],
            max_epochs=N_EPOCHS,
            gpus=GPUS,
            #accelerator="dp",
            precision=PRECISION,
            progress_bar_refresh_rate=5
            )
    trainer.fit(teacher_model, data_module)
    
    trained_teacher_model = Predictor.load_from_checkpoint(
        trainer.callbacks[-1].best_model_path,
        n_classes=1,
        MODEL='efficientnet_b0')
    trained_teacher_model.eval()
    trained_teacher_model.freeze()
    trained_teacher_model = trained_teacher_model.to(device)
    return trained_teacher_model

def train_student_model(train,fold_id,test):
    '''
    input: train,fold_id,test
    output: trained_student_model
    '''
    student_model = Predictor(steps_per_epoch=t_steps_per_epoch,
                                  n_training_steps=total_training_steps,
                                  n_classes=1,
                                  MODEL='efficientnet_b1')
    test_sample_n = int(len(test)/N_FOLDS)
    data_module = DataModule(
        train[train['fold']!=fold_id].append(test.sample(test_sample_n)),
        train[train['fold']==fold_id], # val fold
        train[train['fold']==fold_id], # test data, same as val for now
        batch_size=BATCH_SIZE,
        )
    
    early_stopping_callback = EarlyStopping(
            monitor='val_auc',
            mode="max", 
            patience=3)
    
    checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-student-model-checkpoint-ep{epoch:02d}-{val_auc:.3f}",
            save_top_k = 1,
            verbose = True,
            monitor = "val_auc",
            mode="max",
            )
    
    trainer = pl.Trainer(
            checkpoint_callback= True,
            callbacks=[early_stopping_callback,checkpoint_callback],
            max_epochs=N_EPOCHS,
            gpus=GPUS,
            #accelerator="dp",
            precision=PRECISION,
            progress_bar_refresh_rate=5
            )
    trainer.fit(student_model, data_module)
    
    trained_student_model = Predictor.load_from_checkpoint(
        trainer.callbacks[-1].best_model_path,
        n_classes=1,
        MODEL='efficientnet_b1')
    trained_student_model.eval()
    trained_student_model.freeze()
    trained_student_model = trained_student_model.to(device)
    return trained_student_model
    

def get_predictions(df,trained_model):
    '''
    input: df,trained_model
    output: predictions (np.array)
    '''
    predictions = []
    val_dataset = TrainDataset(df,transform=get_transforms(data='valid'),test = True)
    dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12)
    for item in tqdm(dataloader, position=0, leave=True):
        prediction = trained_model(item.to(device))
        predictions.append(prediction.flatten().sigmoid())
    predictions = torch.cat(predictions).detach().cpu()
    final_preds = predictions.squeeze(-1).numpy()
    return final_preds #(np.array)

def train_loop():
    '''
    input : None
    output : oof_train_dict ,oof_test_dict
    '''
    oof_train_dict = {} #[F1,F2,F3,F4,F5]
    oof_test_dict = {} #[F1,F2,F3,F4,F5]
    train["fold"] = -1
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # RUN KFOLD training
    for fold_id, (_, val_idx) in enumerate(skf.split(train["id"], train["target"])):
        # tag fold_id to val_idx
        train.loc[val_idx, "fold"] = fold_id
        
        # train teacher_model
        trained_teacher_model =  train_teacher_model(train,fold_id)
        
        # teacher model predict
        oof_train_dict[fold_id] = get_predictions(train[train['fold']==fold_id],trained_teacher_model)
        oof_test_dict[fold_id] = get_predictions(test,trained_teacher_model)

        # use trained_teacher_model predict pseudo label
        test['target'] = oof_test_dict[fold_id]
        
        # use pseudo_label train student model
        trained_student_model = train_student_model(train,fold_id,test)
        
        # trained_student_model predict
        oof_train_dict[fold_id] = get_predictions(train[train['fold']==fold_id],trained_student_model)
        oof_test_dict[fold_id] = get_predictions(test,trained_student_model)
        
        oof_train_df = train[train['fold']==fold_id] # this is dataframe
        oof_train_df['preds'] = oof_train_dict[fold_id] # fill np.array values to dataframe's columns
        oof_train_dict[fold_id] = oof_train_df # fill dataframe to oof_train_dict
        
        oof_test_df = test # this is dataframe
        oof_test_df['target'] = oof_test_dict[fold_id] # fill np.array values to dataframe's columns
        oof_test_dict[fold_id] = oof_test_df # fill dataframe to oof_test_dict
        
    return oof_train_dict ,oof_test_dict # both is dataframe [df1,df2,df3,df4,df5]

if __name__ == '__main__':
    oof_train_dict ,oof_test_dict = train_loop()
    joblib.dump(oof_train_dict ,'oof_train_dict.pkl')
    joblib.dump(oof_test_dict ,'oof_test_dict.pkl')
    print('all done!')
        
        
        
        
        
        
        
        
        
        
        
        

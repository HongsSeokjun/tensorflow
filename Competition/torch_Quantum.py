# 퀀텀AI경진대회 2025.07.10 ~ 2025.08.05
import multiprocessing
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from datetime import datetime
import os
import random
from sklearn.metrics import accuracy_score
from torch.autograd import Parameter
# ── 1) Reproducibility ───────────────────────────────
SEED = 568 #acc 81.00%
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── 2) Configurations ─────────────────────────────────
BATCH_SIZE      = 64
LR              = 5e-3
EPOCHS          = 500
PATIENCE        = 50
base_path = './Study25/_data/quantum/'
os.makedirs(base_path, exist_ok=True)
path ='C:\Study25\submission\\'
    
# ── 3) Device ─────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 4) Quantum Circuit Class ─────────────────────────────────
class QuantumCircuit(Module):
    def __init__(self):
        super().__init__()
        self.dev = qml.device("default.qubit", wires=6)  # 4개의 큐비트
        self.layers = 3  # 층 수
        self.num_qubits = 6  # 큐비트 수

        # 파라미터 정의: 각 층에서 사용할 가중치 파라미터들
        self.params = Parameter(torch.rand(self.layers * self.num_qubits, dtype=torch.float64), requires_grad=True)

        # qml.PauliZ(0)와 qml.PauliZ(1)로 관측량 설정
        self.obs = [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        # QNode 정의
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(self.num_qubits))

            # 레이어마다 가중치와 출력을 기반으로 동적 설정
            for l in range(self.layers):
                for i in range(self.num_qubits):
                    # 가중치에 따라 각 큐비트에 변환을 적용
                    qml.RX(weights[(l * self.num_qubits + i) % weights.shape[0]], wires=i)

                # CNOT을 적용하여 서로 다른 얽힘을 생성
                for i in range(0, self.num_qubits, 2):  # 2개씩 CNOT 연결
                    if i + 1 < self.num_qubits:
                        qml.CNOT(wires=[i, i+1])

            # 마지막 RZ로 조정
            for i in range(self.num_qubits):
                qml.RZ(weights[(i + weights.shape[0] // 2) % weights.shape[0]], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.qnode = circuit  # QNode 연결

    def forward(self, x):
        return self.qnode(x, self.params)

# ── 5) QuantumCNNClassifier ─────────────────────────────────
class QuantumCNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 클래식 CNN 부분
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 4)  # ✨ 개선 사항: QNN 입력 큐비트 수와 맞춤 (5개)
        self.norm = nn.LayerNorm(4) 
        self.fc2 = nn.Linear(4, 32)
        self.fc3 = nn.Linear(32, 2) # 최종 2개 클래스로 분류
        
        # 양자 QNN 부분
        self.q_params = nn.Parameter(torch.rand(30)) # 양자 회로 가중치
         # 양자 회로 객체 생성
        self.qnn = QuantumCircuit()


    def forward(self, x):
        # 클래식 CNN 특징 추출
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.norm(x)
        
        # ✨ 개선 사항: 양자 회로에 배치 전체를 한 번에 전달하여 효율성 극대화
        q_out = self.qnn(x, self.q_params)  # 양자 회로 출력 (배치 크기만큼)
        # PennyLane 출력이 튜플일 수 있으므로 텐서로 변환
        q_out = torch.stack(list(q_out), dim=1).to(torch.float32)

        # 최종 분류
        x = self.fc2(q_out)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# ── 6) Data ─────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.25,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.25,), (0.5,))
])

train_ds = datasets.FashionMNIST(root=base_path, train=True, download=True, transform=transform_train)
test_ds = datasets.FashionMNIST(root=base_path, train=False, download=True, transform=transform_test)
test_ds1 = pd.read_csv(path+'modified_y_pred_20250805_114857.csv', header=None)

train_mask = (train_ds.targets == 0) | (train_ds.targets == 6)
test_mask = (test_ds.targets == 0) | (test_ds.targets == 6)

train_idx = torch.where(train_mask)[0]
test_idx = torch.where(test_mask)[0]

train_ds.targets[train_ds.targets == 6] = 1
test_ds.targets[test_ds.targets == 6] = 1

# 필요한 인덱스를 Subset에 적용하여 새로운 데이터셋 생성
binary_train_ds = Subset(train_ds, train_idx)
binary_test_ds = Subset(test_ds, test_idx)

train_loader = DataLoader(binary_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(binary_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True) 

# ── 7) main ─────────────────────────────────
if __name__ == '__main__':
    # 규격 검사
    model_for_specs = QuantumCNNClassifier()
    model_for_specs.eval()
    # 개선 사항: QNN 입력 차원을 4로 변경
    dummy_q_inputs = torch.randn(1, 6) 
    dummy_q_weights = model_for_specs.q_params.data
    q_specs = qml.specs(model_for_specs.qnn.qnode)(dummy_q_inputs, dummy_q_weights)
    assert q_specs["num_tape_wires"] <= 8, f"num_tape_wires: {q_specs['num_tape_wires']}"  # 큐비트 수
    assert q_specs['resources'].depth <= 30, f"depth: {q_specs['resources'].depth}"  # 회로 깊이
    assert q_specs["num_trainable_params"] <= 60, f"num_trainable_params: {q_specs['num_trainable_params']}"  # 학습 가능한 파라미터 수
    print("✅ QNN 규격 검사 통과")
    total_params = sum(p.numel() for p in model_for_specs.parameters() if p.requires_grad)
    assert total_params <= 50000
    print(f"✅ 학습 전체 파라미터 수 검사 통과: {total_params}")
    del model_for_specs

    # 학습
    model = QuantumCNNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.NLLLoss()
    best_acc = 0.0
    early_stopping_patience = PATIENCE
    epochs_no_improve = 0
    best_model_path = os.path.join(base_path, 'best_model_improved.pth') # 파일명 변경

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for xb, yb  in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}%")

        if avg_acc > best_acc:
            best_acc = avg_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model updated at epoch {epoch+1} with train acc {avg_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # 8️⃣ 모델 저장
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(base_path, f'model_{now}_final_train_acc_{best_acc:.4f}_improved.pth')
    if os.path.exists(best_model_path):
        torch.save(torch.load(best_model_path), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    print(f"✅ 마지막 모델 저장 완료: {model_path}")

    # 9️⃣ 추론 및 제출 생성
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds = []

    with torch.no_grad():
        for xb_t, _ in test_loader:
            xb_t = xb_t.to(device)
            output = model(xb_t)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())

    final_submission_preds = [0 if p == 0 else 6 for p in all_preds]
    
    # 8. Save
    y_pred_final = np.zeros(len(test_ds), dtype=int)
    y_pred_final[test_idx] = np.where(np.array(final_submission_preds) == 6, 6, 0)
    df = pd.DataFrame({"y_pred": y_pred_final})
    # print(f"✅ 결과 저장 완료: {csv_filename}")
    csv_filename = f"{base_path}y_pred_{now}_improved.csv"
    df.to_csv(csv_filename, index=False, header=False)
    
    test_acc1 = accuracy_score(test_ds1, y_pred_final)
    print(f"finalTest Accuracy: {test_acc1:.4f}")


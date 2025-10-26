from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
# from template_construct.embed_tool import get_embeddings


class QueryErasedDataset(Dataset):
    def __init__(self, data_path, target_embedding_model='qwen3-embedding-4b'):
        """
        加载(原始查询, 擦除后查询)对数据集
        """
        self.data = pd.read_parquet(
            data_path, columns=['query', 'erased', 'erased_embedding'])
        # self.target_embeddings = get_embeddings(
        #     self.data['erased'].tolist(),
        #     model=target_model
        # )
        self.target_embeddings = self.data['erased_embedding'].tolist()
        self.target_embedding_model = target_embedding_model
        self.target_embedding_dim = len(self.target_embeddings[0])

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_query = self.data.iloc[idx]['query']
        target_embedding = torch.tensor(
            self.target_embeddings[idx], dtype=torch.float32)

        # Tokenize raw query
        # encoding = self.tokenizer(
        #     raw_query,
        #     padding='max_length',
        #     truncation=True,
        #     max_length=256,
        #     return_tensors='pt'
        # )

        return {
            'query': raw_query,
            # 'attention_mask': encoding['attention_mask'].squeeze(),
            'target_embedding': target_embedding
        }


class MARLOEmbeddingModel(nn.Module):
    def __init__(self, base_model, target_embedding_dim=2560):
        super(MARLOEmbeddingModel, self).__init__()
        # 尝试加载sentence-transformers库的模型
        self.model = SentenceTransformer(base_model)
        self.use_sentence_transformer = True
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.adapter = nn.Linear(
            self.embedding_dim, target_embedding_dim)

    def forward(self, texts, **kwarg):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return self.adapter(embeddings.clone())

    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts, **kwargs):
        """便捷的编码方法"""
        return self.forward(texts=texts, **kwargs)


def train_model(model_path, data_path, epoch_num=10):
    from tqdm import tqdm  # 导入进度条库

    # 数据集和加载器
    dataset = QueryErasedDataset(data_path)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 划分训练集、验证集、测试集 (85%训练, 5%验证, 10%测试)
    train_indices, temp_indices = train_test_split(
        range(len(dataset)), test_size=0.15, random_state=42)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.67, random_state=42)

    # 创建子数据集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 模型和优化器
    model = MARLOEmbeddingModel(
        base_model=model_path,
        target_embedding_dim=dataset.target_embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CosineEmbeddingLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 移动模型到设备
    # 训练循环
    model.train()
    best_val_loss = float('inf')

    for epoch in range(epoch_num):
        # 训练阶段
        model.train()
        train_loss = 0
        train_bar = tqdm(
            train_loader, desc=f'Train Epoch {epoch+1}/{epoch_num}', leave=False)

        for batch in train_bar:
            optimizer.zero_grad()
            embeddings = model(batch['query'])
            target_embeddings = batch['target_embedding'].to(
                device)  # 移动目标嵌入到设备
            target = torch.ones(embeddings.size(0)).to(device)  # 移动目标标签到设备

            loss = criterion(embeddings, target_embeddings, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader, desc=f'Val Epoch {epoch+1}/{epoch_num}', leave=False)
            for batch in val_bar:
                embeddings = model(batch['query'])
                target_embeddings = batch['target_embedding'].to(
                    device)  # 移动目标嵌入到设备
                target = torch.ones(embeddings.size(0)).to(device)  # 移动目标标签到设备

                loss = criterion(embeddings, target_embeddings, target)
                val_loss += loss.item()
                val_bar.set_postfix(loss=f'{loss.item():.4f}')

        avg_val_loss = val_loss / len(val_loader)

        print(
            f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), './model/best_marlo_model.pth')
            print(f'Best model saved with val loss: {best_val_loss:.4f}')

    # 最终测试
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing', leave=False)
        for batch in test_bar:
            embeddings = model(batch['query'])
            target_embeddings = batch['target_embedding'].to(
                device)  # 移动目标嵌入到设备
            target = torch.ones(embeddings.size(0)).to(device)  # 移动目标标签到设备

            loss = criterion(embeddings, target_embeddings, target)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Final Test Loss: {avg_test_loss:.4f}')


if __name__ == "__main__":
    train_model(model_path='sentence-transformers/all-MiniLM-L6-v2',
                data_path='./data/ignore/20250909_AIMI全量_AIO_erased_embedding.parquet',
                epoch_num=1)

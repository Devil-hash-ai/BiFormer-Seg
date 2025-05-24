import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoderModule(nn.Module):
    def __init__(self, model_name: str = "microsoft/BiomedVLP-CXR-BERT-general", out_dim: int = 256, freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.out_dim = out_dim
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze BERT if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.project = nn.Linear(self.encoder.config.hidden_size, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, texts):
        # Tokenize input text
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = tokenized['input_ids'].to(self.project.weight.device)
        attention_mask = tokenized['attention_mask'].to(self.project.weight.device)

        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        projected = self.project(cls_token)
        return self.norm(projected).unsqueeze(1)  # shape: [B, 1, out_dim]


class PromptEmbeddingRefiner(nn.Module):
    def __init__(self, input_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.LayerNorm(input_dim))
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.GELU())
        self.refiner = nn.Sequential(*layers)

    def forward(self, x):
        return self.refiner(x)


class TextPromptEncoder(nn.Module):
    def __init__(self, model_name="microsoft/BiomedVLP-CXR-BERT-general", out_dim=256, refine_layers=3):
        super().__init__()
        self.base_encoder = TextEncoderModule(model_name=model_name, out_dim=out_dim)
        self.refiner = PromptEmbeddingRefiner(input_dim=out_dim, num_layers=refine_layers)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.base_encoder.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        projected = self.base_encoder.project(cls).unsqueeze(1)
        return self.refiner(projected)  # [B, 1, out_dim]


if __name__ == "__main__":
    sample_texts = ["COVID-19 opacities in the left lower lung.", "No signs of pneumonia detected."]
    model = TextEncoderModule().eval()
    refined_encoder = TextPromptEncoder().eval()

    print("Simple TextEncoderModule Output:")
    output1 = model(sample_texts)
    print(output1.shape)

    print("Refined TextPromptEncoder Output:")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")
    tokenized = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    output2 = refined_encoder(tokenized['input_ids'], tokenized['attention_mask'])
    print(output2.shape)
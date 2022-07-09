from transformers import LayoutLMForSequenceClassification

def layoutlm_classify_model(pretrain_path, num_label=7):
    model = LayoutLMForSequenceClassification.from_pretrained(pretrain_path, num_labels=num_label)
    return model
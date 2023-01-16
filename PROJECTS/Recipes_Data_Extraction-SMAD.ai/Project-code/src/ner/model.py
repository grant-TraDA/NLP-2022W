from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)


def load_pipeline(model_name: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )
    return ner_pipeline

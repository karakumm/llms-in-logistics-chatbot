This repository contains a code for bachelor thesis on Application of LLMs for a Chatbot System in the Logistics Industry


## About
Digital transformation is a crucial strategic plan element for various industries in today's fast-paced and dynamic market, including the supply chain. Providing excellent customer service is an essential aspect of logistics management; thus, integrating artificial intelligence into customer service is an effective solution. To enhance external communication with customers, AI chatbots that use Large Language Models can lead to satisfactory user experience, although this subject has not yet been thoroughly explored. This study investigates the application of LLMs in developing chatbot solutions for customer service within the logistics industry. The research aimed to employ an effective fine-tuning approach and compare leading LLMs, considering customer satisfaction crucial for business growth. Various pre-trained models, including Llama, Vicuna, and Mistral, were evaluated using parameter-efficient techniques like QLoRA for domain-specific fine-tuning. A dataset with logistics-related queries was employed to evaluate the models, utilizing multiple evaluation methods such as human reviews, cosine similarity, ROUGE, SacreBLEU, and Perplexity measurements, and the innovative LLM-as-a-Judge approach. This study may assist logistics companies in making informed decisions regarding integrating LLMs into customer service.

### Prerequisites
```Shell
pip install torch transformers sentence_transformers
pip install datasets evaluate huggingface_hub
pip install peft trl
``` 

### Training and Evaluation
Prepare YAML configuration file beforehand to use `train_qlora.py` for LLM's fine-tuning using Quantized LoRA.

```Shell
python train_qlora.py --config configuration.yaml
```

`validation.ipynb` includes code for testing the merged model on test dataset. 
`evaluation.ipynb` includes evaluation metrics measurement (cosine distance, ROUGE, SacreBLEU, Perplexity), visualizations, and LLM-as-a-Judge implementation.

## Results

|                       | **Vicuna 7b** | **Llama-13b** | **Vicuna-13b** | **Vicuna-33b** | **Mistral-7b** | **Zephyr-7b** |
|-----------------------|---------------|---------------|----------------|----------------|----------------|---------------|
| **Mean Cosine Dist**  | 0.828         | 0.619         | 0.536          | 0.565          | 0.567          | 0.626         |
| **SacreBLEU**         | 24.77         | 16.97         | 10.1           | 11.17          | 6.87           | 62.15         |
| **ROUGE**             | 0.47          | 0.34          | 0.31           | 0.36           | 0.27           | 0.74          |
| **Perplexity**        | 47.86         | 16.1          | 21.43          | 22.6           | 19.4           | 63.12         |

*Table: Scores Results*


## Acknowledgements
This repository is built using several Hugging Face libraries (`transformers`, `peft`, `datasets`).

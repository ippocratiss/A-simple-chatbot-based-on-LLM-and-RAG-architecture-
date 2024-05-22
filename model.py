# Neeeded packages
import torch
import transformers 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# This is the LLM pre-trained model we use. 
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

# Define the tokenizers for the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Quantization settings 
use_4bit = True # 4-bit precision
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4" # Type of quantization 
use_nested_quant = False # Nested quantization
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Define the quantized LLM model: model_name + quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)


# Useful functions to e.g load the paper, do text splitting, etc.

def bf(text): # bold-face text 
  return "\033[1m" + text + "\033[0m"
    
def load_doc(file): # the input can be an online link 
    loader=PyPDFLoader(file)
    pages  = loader.load_and_split()
    print(bf('Document loaded OK.'))
    return loader.load()

def doc(s): # text splitting
    text_splitter = CharacterTextSplitter()
    splits = text_splitter.split_text(s)
    return text_splitter.create_documents(splits) #this should return the list of documents.

